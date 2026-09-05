# Running a single op on Quasar

Step-by-step runbook for putting one op through forge onto the craq-sim Quasar
simulator. There is no Quasar silicon, so the simulator is the only way to *run*
rather than merely compile.

For what Quasar is, how the architecture reaches the compiler, and the current op
status, see [Quasar](quasar.md). This page is only the mechanics.

> **Expect the run to wedge.** As of 2026-09-04 compilation succeeds in about 9
> seconds and execution then stalls. That is a known open blocker below forge, not a
> mistake in your setup — see step 4. Follow this runbook to reproduce it, to test a
> fix, or as the basis for a bisect.

## 0. Check the stack is on the Quasar branches

Quasar support lives on branches, and a build can force-checkout the submodule back
to its pin. Verify before anything else:

```bash
cd /proj_sw/user_dev/ctr-lelanchelian/tt-forge-onnx

git -C third_party/tt-mlir rev-parse --short HEAD          # expect 6f17e2407b

grep -q "l1Size = 4194304" \
  third_party/tt-mlir/lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp \
  && echo "quasar sysdesc OK" || echo "REVERTED - re-checkout the branch"
```

Detached HEAD is fine as long as the SHA matches — that is just how a submodule
gitlink checks out. If the descriptor check says `REVERTED`:

```bash
git -C third_party/tt-mlir checkout quasar-forge-onnx-bringup
```

Branches this depends on, all on `origin`:

| Repo | Branch |
|---|---|
| tt-mlir | `lelanchelian/quasar-forge-onnx-bringup` |
| tt-metal | `lelanchelian/quasar-forge-onnx-op-slicing` |

## 1. Build the QSR simulator

Skip if you already have one. Public ttsim releases ship wormhole and blackhole
builds only; the Quasar one must be built locally, and `TT_VERSION=2` is required —
plain `make.py` gives you WH/BH.

```bash
cd /proj_sw/user_dev/ctr-lelanchelian/craq-sim
TT_VERSION=2 ./make.py src/_out/release_qsr/libttsim.so
```

## 2. Stage the simulator and export the environment

```bash
cd /proj_sw/user_dev/ctr-lelanchelian/tt-forge-onnx
source env/activate
source ./scripts/quasar_sim_env.sh        # MUST be sourced, not executed
```

Set `QUASAR_SIM_DIR` first if your `libttsim.so` is not at the default
`/proj_sw/user_dev/ctr-lelanchelian/craq-sim/src/_out/release_qsr`.

The script exists because three things here are easy to get wrong by hand:

* It copies `quasar_32_arch.yaml` to `soc_descriptor.yaml` **beside the `.so`**. UMD
  derives the descriptor path from the library's own directory and only accepts that
  generic name; the arch-specific filename is never consulted.
* It **unsets** `TT_METAL_MOCK_CLUSTER_DESC_PATH`. That variable expects a *cluster*
  descriptor, a different schema — pointing it at the SoC descriptor makes UMD fail
  with "Invalid YAML". It is not needed for a single-chip simulator.
* It checks the `.so` really is a QSR build. A stale WH/BH one loads far enough to be
  confusing and then fails with `MissingSpecification: libttsim_pci_mem_wr_bytes`.

Confirm the output says:

```
ARCH_NAME / CHIP_ARCH          = quasar / quasar
```

## 3. Run one op

In its own pytest process, under a wall-clock timeout:

```bash
export TTSIM_PROGRESS_HEARTBEAT_CLOCKS=10000000
export TTSIM_PROGRESS_DETAIL=1
export TTSIM_PROGRESS_TENSIX_DETAIL_STUCK_ONLY=1

timeout 3600 python -u -m pytest -svv \
  forge/test/mlir/test_quasar_sim.py::test_add 2>&1 | tee /tmp/qsr_add.log
```

Swap the op by changing the test id: `::test_mul`, `::test_sub`, `::test_div`,
`::test_relu`. All take the same path.

Two things that are not optional:

* **Its own process.** tt-metal's `RunTimeOptions` and forge's `TTSystem` are both
  construct-once-per-process, so the first test to touch a device fixes
  hardware-vs-simulator for the entire session. Never mix Quasar tests with ordinary
  ones. (`test_quasar_sim.py` is deliberately left out of `pytest.ini`'s `testpaths`,
  so a bare `pytest` never collects it.)
* **A timeout, but not a short one.** craq-sim's own Quasar op CI allows 240 minutes.
  A sub-hour timeout will kill a healthy run, and a killed run is indistinguishable
  from a hang.

## 4. What you will see

Compilation succeeds quickly:

```
INFO | forge.compile:forge_compile_from_context:425 - Compilation completed.
INFO | forge.compiled_graph_state:__call__:334 - Running model ModelProto forward on device...
```

Then execution wedges, with `pending_tensix=4` in every heartbeat and never
returning. Four Tensix pipes stall symmetrically with both operands unpacked and the
math unit never consuming them. Full evidence and the ruled-out hypotheses are in
[Quasar](quasar.md).

## 5. Tell hung from slow

They look identical from outside — both sit at 100% CPU printing nothing.

```bash
py-spy dump --pid $(pgrep -f '[t]est_quasar_sim' | head -1) --native
```

* `libttsim_clock_all_devices` under `TTSimTTDevice::after_read`, inside `verify()`
  → compiled fine, executing on device.
* Still inside `forge.compile` → compiling, which should take seconds.

Then read the heartbeat. `pending_tensix` and the Tensix PCs frozen across several
samples means the stall; advancing means it is genuinely working.

Note `pgrep -f '[t]est_quasar_sim'` uses the bracket trick so the pattern does not
match its own command line. Never put a `pkill -f <pattern>` in the same shell
command that later names the same target — the bracket protects the pattern, not the
rest of the line, and it will kill the job you are starting.

## 6. Going further

The simulator's own diagnostics, all read by `libttsim.so` so they need no tt-metal
patch:

```bash
export TTSIM_STALLWAIT_TRACE=1          # which wait gate is unmet
export TTSIM_TENSIX_STALL_TRACE=1
export TTSIM_TENSIX_STALL_TRACE_TILE=0
export TTSIM_SEM_TRACE=1                # seminit / sempost / semget / semwait
export TTSIM_SEM_TRACE_MAX_CLOCK=30000000
```

`quasar_sim_env.sh` already sets `TTSIM_HANG_WATCHDOG_CLOCKS` and
`TTSIM_PROGRESS_HEARTBEAT_CLOCKS`. Be aware the watchdog only classifies *true*
deadlocks — a loop that keeps retiring RISC-V instructions is not "hung" by its
definition, which is exactly the current failure, so it stays silent.

Check what your simulator build actually reads before setting anything else; craq-sim's
README documents knobs that are not in every build:

```bash
strings "$TT_METAL_SIMULATOR" | grep '^TT_METAL_SIMULATOR'
strings "$TT_METAL_SIMULATOR" | grep '^TTSIM_' | sort -u
```

## Compiling for Quasar without any of this

If you only need to compile — no simulator, no hardware:

```python
import forge
from forge.config import CompilerConfig, MLIRConfig

cfg = CompilerConfig(mlir_config=MLIRConfig().set_target_arch(forge._C.Arch.QUASAR))
compiled = forge.compile(model, sample_inputs=inputs, compiler_cfg=cfg)
```

That path works today and is covered by `forge/test/mlir/test_target_arch.py`, which
CI runs on a plain runner with no accelerator.
