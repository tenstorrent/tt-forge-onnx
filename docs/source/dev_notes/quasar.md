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
  real topology matters.
* Quasar rejects `experimental_weight_dtype`. Its format set has no `bf8_b`/`bf4_b`,
  and without the guard in `mlir_config.cpp` the failure surfaces much later as a
  tt-metal host format-validator throw.

Covered by `forge/test/mlir/test_target_arch.py`, which the
`test-compile-arch-sub.yml` CI job runs on a plain runner with no accelerator. That
job is the only PR coverage for Quasar, since no Quasar runner exists.

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

### Debugging a hang

An unresponsive simulated core otherwise hangs with no output at all. The script sets
`TT_METAL_SIM_CORE_WAIT_TIMEOUT_MS` (default 180000), which turns that into
`Device 0: Timeout (180000 ms) waiting for physical cores to finish: 2-2.` It is only
honoured by tt-metal builds carrying the `quasar-sim-core-wait-diagnostic` change.

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
