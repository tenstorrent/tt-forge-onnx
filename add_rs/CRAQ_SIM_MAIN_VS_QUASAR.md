# craq-sim: main branch vs quasar branch, tested (2026-09-10)

Question: should the Quasar simulator be built from craq-sim `main` instead of the
`quasar` branch, and does that let conv2d run?

Answer: **no.** A `main` build cannot run anything through tt-metal.

## What was built

A second, parallel build — no override needed, since `scripts/quasar_sim_env.sh`
already takes `QUASAR_SIM_DIR`:

| | quasar branch | main |
|---|---|---|
| checkout | `/proj_sw/user_dev/ctr-lelanchelian/craq-sim` @ `f9ff0702` | `/proj_sw/user_dev/ctr-lelanchelian/craq-sim-main` @ `f9a2b18e` (worktree) |
| build | `./make.py src/_out/release_qsr/libttsim.so` | same, exit 0 |
| `libttsim.so` | 658280 bytes | 293608 bytes |
| QSR sanity (`libttsim_pci_mem_wr_bytes`) | present | present |

So main *does* build a QSR target, and it *does* open a device:

```
UMD | Creating Simulation device (cluster.cpp:266)
EmulationDriver | TTSimTTDevice chip_id=0 PCI vendor_id=0x1e52 device_id=0xfeed
```

## Then it dies during device bring-up

Both ops fail at the identical simulated clock and address:

```
[3099] ERROR: UnimplementedFunctionality: t_tile_mmio_wr32: addr=0x1842200
```

| op | quasar-branch sim | main-branch sim |
|---|---|---|
| `reshape` | PASS `pcc=0.999996` | `UnimplementedFunctionality` at clock 3099 |
| `conv2d`  | (see conv_quasar_sim.log) | `UnimplementedFunctionality` at clock 3099 |

Same message, same clock, same address for both ops — this is a **device bring-up**
failure, not an op problem. tt-metal's Quasar firmware performs an MMIO write to a
tile register that main's model does not decode, so nothing gets as far as compiling
or running an op. conv2d is therefore not reachable on a main build at all.

Logs: `logs/smoke_main_sim.log`, `logs/main_sim_raw.log`, `logs/conv_on_main_sim.log`.

## Why — the model is roughly half the size

|  | quasar branch | main |
|---|---|---|
| `src/tile.cpp` | 8438 lines | 5931 |
| `src/riscv_impl.h` | 4139 lines | 1798 |

~4,850 fewer lines of tile and RISC-V model on main, which matches the 2.2x
difference in `.so` size. `t_tile_mmio_wr32` exists on both, but main's version
lacks the register decode for the address tt-metal writes.

This is consistent with the branch history: `main` and `quasar` diverged on
2026-06-10 and both are still active; `quasar` carries 345 commits `main` does not,
almost entirely Quasar model work (28 mention DFB, 15 SFPU, 8 tilize).

## Conclusion

- Keep building the simulator from the `quasar` branch.
- Two builds coexist fine via `QUASAR_SIM_DIR`; the main build is kept at
  `craq-sim-main` for reference but is not usable.
- The remaining upgrade worth doing is **within** the quasar branch: we are 11
  commits behind its tip, including a DRAM single-bank stream corruption fix and a
  stale/zero instruction-fetch fix.
