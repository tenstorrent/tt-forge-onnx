# Quasar

There is no Quasar silicon to plug in, so Quasar support in tt-forge-onnx is split
into two independent capabilities. Work out which one you need before reading further.

| I want to... | Use | Needs |
|---|---|---|
| Compile for Quasar from any machine | `MLIRConfig().set_target_arch(Arch.QUASAR)` | nothing |
| Compile against a specific real Quasar topology | `MLIRConfig().set_system_desc_path(...)` | a captured `.ttsys` |
| Actually **run** a graph on Quasar | craq-sim, the QSR simulator | a locally built `libttsim.so` |

## Compiling for Quasar with no device attached

By default forge reads the system descriptor off whatever device is physically
present, so the compile target is whatever hardware you happen to be sitting in front
of — and compiling with no hardware is impossible. Two `MLIRConfig` options move the
descriptor out of the device and into the compile request:

```python
import forge
from forge.config import CompilerConfig, MLIRConfig

cfg = CompilerConfig(mlir_config=MLIRConfig().set_target_arch(forge._C.Arch.QUASAR))
compiled = forge.compile(model, sample_inputs=inputs, compiler_cfg=cfg)
```

`set_target_arch` accepts `WORMHOLE_B0`, `BLACKHOLE` and `QUASAR` — the three the
`ttir-to-ttnn` pipeline has a mock descriptor for. It maps to the pipeline option
`mock-system-desc-arch=<arch>`.

When either option is set, `emit_mlir` skips stamping `ttcore.system_desc` on the
module, and tt-mlir's `TTCoreRegisterDevice` pass builds the descriptor from the
pipeline options instead — it only fabricates one when the module carries none. That
same skip is what avoids `TTSystem::get_system()`, which is the call that would
otherwise open a device.

Two caveats:

* The mock descriptor is **nominal**. It carries the arch's grid, L1 size and DRAM
  geometry, not a specific board's harvesting. Use `set_system_desc_path` when the
  real topology matters — see below.
* Quasar rejects `experimental_weight_dtype`. Its format set has no `bf8_b`/`bf4_b`,
  and without the guard in `mlir_config.cpp` the failure surfaces much later as a
  tt-metal host format-validator throw.

Covered by `forge/test/mlir/test_target_arch.py`, which the
`test-compile-arch-sub.yml` CI job runs on a plain runner with no accelerator. That
job is the only PR coverage for Quasar, since no Quasar runner exists.

### Capturing a real descriptor

`set_system_desc_path` needs a `.ttsys` flatbuffer. Capture one on the machine that
has the device — including a simulated one, so this is how you get a *real* Quasar
descriptor rather than the nominal mock:

```python
import forge

forge._C.runtime.experimental.save_system_desc("quasar_system_desc.ttsys")
```

This opens the attached device, which is the point. It is the equivalent of
`ttrt query --save-artifacts` without needing ttrt installed. Then, from anywhere:

```python
cfg = CompilerConfig(mlir_config=MLIRConfig().set_system_desc_path("quasar_system_desc.ttsys"))
```

`system_desc_path` wins over `target_arch` when both are set, mirroring the tt-mlir
pipeline where a non-empty `system-desc-path` beats `mock-system-desc-arch`.

## Running on the craq-sim simulator

craq-sim presents a virtual QSR device to UMD, so nothing on the forge side is
Quasar-specific: graphs go through the ordinary compile-and-verify path and land on a
device that happens to be simulated. It needs no emulator and no NDA hardware.

### Build the simulator

Public ttsim releases ship wormhole (`libttsim_wh.so`) and blackhole
(`libttsim_bh.so`) builds only. The Quasar one has to be built locally, from a
craq-sim checkout:

```bash
TT_VERSION=2 ./make.py src/_out/release_qsr/libttsim.so
```

### Run

```bash
source ./scripts/quasar_sim_env.sh          # QUASAR_SIM_DIR overrides the default path
pytest -svv forge/test/mlir/test_quasar_sim.py
```

The script stages the simulator, exports the environment, and sanity-checks that the
`.so` is really a QSR build — a stale wormhole/blackhole one loads far enough to be
confusing and then fails with `MissingSpecification: libttsim_pci_mem_wr_bytes`.

Two things it handles that are easy to get wrong by hand:

* **The SoC descriptor must be copied *and renamed*.** UMD derives the path from the
  `.so`'s parent directory and expects the generic name `soc_descriptor.yaml`
  (`get_soc_descriptor_path_from_simulator_path` in
  `umd/device/simulation/simulation_chip.cpp`). The arch-specific
  `quasar_32_arch.yaml` filename is never consulted.
* **Do not set `TT_METAL_MOCK_CLUSTER_DESC_PATH`.** It expects a cluster descriptor,
  which is a different schema; pointing it at the SoC descriptor makes UMD fail with
  "Invalid YAML". The script unsets an inherited value.

### Run Quasar tests in their own pytest process

tt-metal's `RunTimeOptions` (inside the `MetalContext` singleton) and forge's
`TTSystem` are both construct-once-per-process. **The first test to touch a device
fixes hardware-vs-simulator for the entire session**, and nothing can change it
afterwards. Mixing Quasar tests with ordinary ones would silently run one group
against the wrong target.

`forge/test/mlir/test_quasar_sim.py` is therefore deliberately left out of
`pytest.ini`'s `testpaths`, so a bare `pytest` never collects it. Its skip is decided
at collection time via a module-level `pytestmark`, not a fixture, because the root
conftest's autouse property-recorder fixture already probes the device and there is no
ordering guarantee that would let a fixture here run first.

### It is slow, and there are levers

Expect a cycle-accurate simulator to be slow: execution, not compilation, dominates.
A stack sample of a run that looks stuck usually shows

```
libttsim_clock_all_devices (libttsim.so)
tt::umd::TTSimTTDevice::after_read (libtt-umd.so)
```

which is the host polling a completion flag, with each poll clocking simulated time
forward. That is forward progress, not a deadlock — `py-spy dump --pid <pid> --native`
is the quickest way to tell the two apart, and it also distinguishes "still compiling"
from "executing on device".

Keep the Quasar test file small for this reason. The accelerator knobs below are the
lever if runtime becomes the blocker. Check what your `libttsim.so` actually reads
before setting any of them — craq-sim's README documents `*_DRAM_TELEPORT` and
`*_L1_TELEPORT` too, but neither is present in the QSR build here:

```bash
strings "$TT_METAL_SIMULATOR" | grep '^TT_METAL_SIMULATOR'
```

| Variable | Effect |
|---|---|
| `TT_METAL_SIMULATOR_PARALLEL_TENSIX_TILE_CLOCK=1` | clock whole Tensix tiles in parallel |
| `TT_METAL_SIMULATOR_PARALLEL_CLOCK_THREADS=N` | cap on parallel clock lanes (`0`/unset = auto, `1` = serial) |
| `TT_METAL_SIMULATOR_PARALLEL_CHIP_CLOCK=1` | clock multiple simulated chips in parallel |
| `TT_METAL_SIMULATOR_CQ_WAIT_CLOCKS=N` | clock pumping while waiting on command-queue progress |

`quasar_sim_env.sh` deliberately sets **none** of them. They are marked experimental
opt-in upstream and were documented against a fast-dispatch Blackhole run rather than
slow-dispatch Quasar, so turning them on by default would change results on a guess.
Set them explicitly after sourcing the environment, and re-check numerics when you do.

### Debugging a hang

An unresponsive simulated core spins at 100% CPU indefinitely with no output at all —
indistinguishable from "still simulating", which on a cycle-accurate simulator is
genuinely slow. `quasar_sim_env.sh` sets two variables that make the difference visible.
Both are read by `libttsim.so` itself, so they work with any tt-metal:

| Variable | Effect |
|---|---|
| `TTSIM_HANG_WATCHDOG_CLOCKS=N` | fails the run after `N` simulated clocks with pending work but no RISC-V progress and no Tensix retirement |
| `TTSIM_PROGRESS_HEARTBEAT_CLOCKS=N` | prints chip-cycle progress, active RISC-V PCs, pending Tensix FIFOs and outstanding NoC counts every `N` clocks |

The watchdog classifies **true deadlocks** only. A firmware loop that keeps retiring
instructions is not "hung" by that definition, so still wrap long runs in a wall-clock
`timeout`.

Note that tt-metal's own `TT_METAL_SIM_CORE_WAIT_TIMEOUT_MS` is *not* an alternative
here: it lives on the unpinned `quasar-sim-core-wait-diagnostic` branch and is inert
against the pinned tt-metal. The two TTSIM variables above supersede it and need no
tt-metal patch at all.

## Op status

Score each op three ways; only the third counts:

1. does it exist under `ttnn/.../experimental/quasar/`
2. does the tt-mlir runtime dispatch forge's op to it
3. does it actually run

Most "op X is broken on Quasar" turns out to be "Quasar's op X falls through to the
*mainline* op X", which is refused rather than merely slow: mainline program factories
construct `DataMovementKernel`, whose constructor `TT_FATAL`s on Quasar with "Use
QuasarDataMovementKernel instead". Attribute from the stack, not the test name.

| Op | Status |
|---|---|
| Add, Mul, Sub, Div | run and verify |
| relu | runs and verifies (PCC 0.95), but *rewritten*, not dispatched — see below |
| to_layout, reshape, transpose/permute, reductions, pools, linear, matmul | dispatched to the Quasar op library |
| Greater / Less / Equal / GE | **blocked, Metal ask** — comparison SFPU kernels are `#ifndef ARCH_QUASAR` |
| conv2d | **blocked, Metal ask** — `conv_bmm_tilize_metal2` deadlock, tt-metal #48552 |

The two genuine asks for Metal are conv2d and a unary path; everything else that fails
is dispatch work on our side.

relu is worth calling out because it is the one op with no dispatch target at all:
Quasar has no unary family under `experimental/quasar/`, only binary and binary_ng. It
is emitted as `add(x, 0)` with relu fused as an LHS activation — `relu(x) + 0 ==
relu(x)`, and adding `0.0f` is exact in bf16. `max(x, 0)` was tried first and fails,
because Quasar's tensor-scalar maximum runs on the unary clamp path and delegates
straight back to mainline `ttnn::prim::unary`. This is the correctness fix, not the
performance one: the hand-written Quasar ResNet-50 folds relu into the preceding
add/conv and removes the op entirely, which needs a binary+activation pattern in
tt-mlir's fusing pass that does not exist yet.

## Where Quasar support lives, and how it gets lost

Quasar support is **not** in tt-mlir or tt-metal `main`. It lives on branches that
`third_party/tt-mlir` is pinned to:

| Repo | Branch | Carries |
|---|---|---|
| tt-mlir | `lelanchelian/quasar-forge-onnx-bringup` | the Quasar mock system descriptor, the `TTNNCollectPerfMetrics` skip, and the TTNN runtime op dispatch |
| tt-metal | `lelanchelian/quasar-forge-onnx-op-slicing` | the Quasar `to_memory_config` fix and the `scaleout_tools` PCH patch |
| tt-metal | `lelanchelian/quasar-sim-core-wait-diagnostic` | `TT_METAL_SIM_CORE_WAIT_TIMEOUT_MS` (not pinned) |

At the previous pin, `createDefaultQuasarSystemDesc` returned **zero chip
descriptors**, so `mock-system-desc-arch=quasar` could not compile anything, and
`TTNNCollectPerfMetrics` emitted a hard error for arch `quasar`. Both fail the compile
rather than degrading.

> **Hazard.** Both the parent build and the tt-mlir build force-checkout their
> submodules and `ExternalProject` to the pinned SHAs. This has already silently
> reverted the Quasar work once, leaving a built `ttmlir-opt` whose behaviour no source
> on disk explained. If you edit tt-mlir or tt-metal in place, be on a **named, pushed**
> branch, and bump the pin.

Pins to bump when the branches move:

* tt-metal: `TT_METAL_VERSION` in `third_party/tt-mlir/third_party/CMakeLists.txt`.
  For local iteration, `-DTTMLIR_TTMETAL_SOURCE_DIR=<path>` overrides it and skips the
  `GIT_TAG` enforcement.
* tt-mlir: the `third_party/tt-mlir` gitlink in this repo.
