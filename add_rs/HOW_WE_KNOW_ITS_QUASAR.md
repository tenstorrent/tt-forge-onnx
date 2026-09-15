# How we know the Add really executes on Quasar

Every claim below has a log in `add_rs/logs/`. Runs dated 2026-09-09.

## 1. The chip the driver opened is a Quasar

UMD builds its cluster from the simulator's SoC descriptor, not from anything forge asserts:

- `craq-sim/src/_out/release_qsr/soc_descriptor.yaml:59` — `arch_name: QUASAR`;
  32 `functional_workers` at `2-2 … 9-5`; `worker_l1_size: 4194304`;
  `dram_bank_size: 1073741824` over 2 banks. md5-identical to tt-metal's
  `tt_metal/soc_descriptors/quasar_32_arch.yaml`.
- `UMD | Creating Simulation device (cluster.cpp:266)` and
  `EmulationDriver | TTSimTTDevice chip_id=0 PCI vendor_id=0x1e52 device_id=0xfeed`
  — no silicon in the path.

## 2. tt-metal itself asserts the arch is Quasar (negative control)

`logs/proof_device_is_quasar.log`

`Mul` is deliberately *not* routed to the Quasar op library, so it reaches mainline
`ttnn::multiply`, whose program factory builds a `DataMovementKernel`:

```
TT_FATAL @ tt_metal/impl/kernels/kernel.hpp:418:
    MetalContext::instance(context_id_).get_cluster().arch() != ARCH::QUASAR
    DataMovementKernel is not supported on Quasar. Use QuasarDataMovementKernel instead.
```

That predicate reads the **live cluster arch**, not forge's enum and not our system
descriptor. It only fires on Quasar; the same `Mul` passes on Wormhole. In the same
process, `Add` returned `pcc=0.999985`.

## 3. The device code was built for the Quasar Tensix

From the JIT compile line in `logs/kernel_perturbation3.log`:

```
riscv-tt-elf-g++ ... -mcpu=tt-qsr32-tensix -DARCH_QUASAR
  -DNUM_DRAM_BANKS=2 -DNUM_L1_BANKS=32 -DPCIE_NOC_X=10 -DPCIE_NOC_Y=8
  -I .../tt_metal/tt-llk/tt_llk_quasar/ -I .../hw/ckernels/quasar/metal/...
  .../tt_metal/hw/firmware/src/tt-2xx/trisck.cc
```

Quasar ISA, Quasar LLKs, `tt-2xx` firmware, and the 2 DRAM banks / 32 L1 banks from
section 1.

## 4. The cores actually ran (device-side prints)

`logs/kernel_perturbation.log`, with `TT_METAL_DPRINT_CORES=all`. Prints arrive from
processors named `DM0…DM7` and `N0TR0…N3TR3` — names that exist only in
`tt_metal/llrt/hal/tt-2xx/quasar/qa_hal_tensix.cpp:246`, i.e. the Quasar HAL, and which
match Quasar's 4 Tensix ("Neo") × 4 TRISC layout. Wormhole would report
`BRISC/NCRISC/TRISC0-2`.

## 5. The numbers come from that kernel (perturbation test)

`logs/kernel_perturbation3.log`. The Quasar Add program factory names its compute
kernel at `binary_ng_metal_v2_factory.cpp:96`:
`ttnn/.../quasar/binary_ng/device/kernels_dfb/compute/eltwise_binary_no_bcast_dfb.cpp`.

Comment out the one line that writes the result out —
`pack_tile(i, dfb_out_id);` (line 85) — change nothing else, rebuild with an empty
`TT_METAL_CACHE` (`JIT cache stats: 0/24 hits`), and the host sees:

| kernel | device[0:4] | vs `a+b` |
|---|---|---|
| unmodified | `0.7344, 1.125, 0.2656, 0.8477` | `pcc=0.999985` PASS |
| `pack_tile` removed | `0.0, 0.0, 0.0, 0.0` | `pcc=nan` FAIL |
| restored (`logs/add_after_restore.log`) | `0.7344, 1.125, 0.2656, 0.8477` | `pcc=0.999985` PASS |

The host-visible answer is produced by that kernel packing into the output dataflow
buffer on the device. Nothing on the host is computing it.

## Traps worth knowing

- Two other copies of a similarly-named kernel
  (`quasar/binary/device/kernels/compute/eltwise_binary_kernel.cpp`, source and
  `build/install`) are **dead code** — nothing references them. Editing either changes
  nothing, which looks exactly like "the device isn't running my kernel." Take the path
  from the program factory, not from the filename.
- Kernel include paths in the factory are repo-relative, so runs must start from
  `$TT_METAL_HOME`.
- `DPRINT << ... << ENDL()` is a hard error in this tree; use `DPRINT("fmt", ...)`.

## Caveat

This is craq-sim presenting as a Quasar through the driver layer: correctness, not
performance. fp32 still hangs (Quasar tilize completes without setting `RUN_MSG_DONE`),
which is why everything above is bf16.
