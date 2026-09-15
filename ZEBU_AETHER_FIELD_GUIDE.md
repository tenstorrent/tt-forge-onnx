# Quasar Execution Field Guide — ZeBu, aether, and the layers between

What actually happens between `forge.compile()` and a Tensix core doing arithmetic,
and why the same binary can be right on one target and wrong on the next.

- **Grounded in** tt-metal, tt-mlir, UMD and craq-sim as checked out in this workspace
- **Measurements** from runs on craq-sim and the ZeBu emulator, 11–15 Sep 2026
- **Caveat** line numbers move; treat citations as pointers and re-check before quoting

> Text version of the artifact at https://claude.ai/artifact/A4MBpgw6bb3fhpYxzn6nUb
> (originally written 12 Sep). **Section 0 lists what has changed since** — the
> original's headline measurements are now out of date, and one of its conclusions
> was wrong.

---

## 0. What has changed since this was first written

Four claims in the 12 Sep version have been superseded by later measurement. They
are corrected in place throughout, but worth stating plainly because two of them
were the document's conclusions.

**The emulator failures are fixed.** The original reported a 4-D permute at
−0.000173 and a 1×1 convolution at 0.086632 on the emulator. Every ResNet-50 op
now passes there:

| op | emulator pcc | op | emulator pcc |
|---|---|---|---|
| `relu` | 1.000000 | `conv2d` 1×1 | 0.999996 |
| `reshape` | 0.999996 | `conv2d` 3×3 (2×2 spatial) | 0.999992 |
| `multiply` | 0.999991 | `conv2d` 3×3 (4×4 spatial) | 0.999991 |
| `typecast` | 0.999995 | `max_pool2d` 3×3 s2 p1 | 0.999940 |
| `matmul` | 0.999860 | `mean` (dim −2) | 0.999687 |
| `linear` | 0.999711 | `permute` NCHW↔NHWC (sub-tile) | 0.999996 |

**"craq-sim simply does not model whatever the RTL does here" was wrong.** That was
the original's explanation for the emulator-only permute failure. The real causes
were three specific runtime defects, all found by instrumenting logical invariants:

1. **The device tilize** (ROW_MAJOR → TILE) corrupts unaligned tensors on the
   emulator but not on craq-sim.
2. **The tiled reshape only relabels** — it never repacks across tile rows, so
   `[1,2,2,256] → [1,1,4,256]` returned padding from element 512 on.
3. **The device untilize** (TILE → ROW_MAJOR) loses a quarter of a sub-tile-height
   tensor from tile column 16 onward. `[1,1,4,1024]` lost 1024 of 4096 elements;
   `[1,1,4,256]` was exact. This one sits immediately before `ttnn.conv2d` in every
   NCHW convolution graph, and was the root cause of ResNet-50 scoring 0.88 instead
   of 0.9999.

**The "75 hours for ResNet-50" figure was built on an inflated count.** craq-sim
writes a DFB-config log line *per core*, so its 10,623 "launches" are ~10× the real
program count on a one-core emulator. Corrected arithmetic is in §8.

**The emulator has exactly one worker core — now measured, not inferred.** Its own
SoC descriptor reports `grid = 1x1`, `num_dram_channels = 1`, against craq-sim's
`4x8` and 2. Everything that is a property of a Quasar *core* is identical on both.

---

## 1. Start here: two different things are called "the simulator"

This is the single most useful thing to internalise, and the distinction is made by
one line of code that checks a **file extension**.

UMD looks at whatever `TT_METAL_SIMULATOR` points to. If the path ends in `.so` you
get one world; if it is a directory you get a completely different one.

```
simulation_device_factory.cpp:15-21
    if (simulator_path.extension() == ".so")
```

### Path ends in `.so` → craq-sim, a functional model

A ~715 KB shared library `dlopen`ed into **your own process**. UMD's PCIe reads and
writes become direct C function calls. It is an executable specification of the
chip: bit-exact arithmetic, no notion of time.

| | |
|---|---|
| class | `TTSimTTDevice` |
| models | Tensix ISA, RISC-V harts, all 4 MB L1, NOC, overlay |
| does not model | timing, power, NOC contention, cycle counts |
| cost | seconds per op; full ResNet-50 at 32×32 in ~25 s |

### Path is a directory → ZeBu, real RTL on an emulator

UMD spawns `run.sh` from that directory, opens a socket, and waits. The actual
Quasar RTL runs on a Synopsys ZeBu-Server5 box in another building. Every register
access is a network round trip.

| | |
|---|---|
| class | `RtlSimulationTTDevice` |
| models | the real gates — timing, arbitration, everything |
| does not | run fast, or run unattended |
| cost | 20–24 s per program launch, plus 38–66 s model load per process |

### Why you should care

A functional model only models what someone wrote down. It is *deliberately
stricter* than silicon in places and silent about things it doesn't implement in
others. Passing on craq-sim is good evidence about arithmetic and much weaker
evidence about hardware behaviour.

The concrete case: three separate layout-conversion defects were **correct on
craq-sim and wrong on the emulator**. Same runtime binary, same op, same day. See
§0 and §7.

---

## 2. Vocabulary: what a Tensix core is

A Tensix "core" is not a CPU core. It is a small cluster of processors around a
block of scratchpad memory, and the processors mostly *issue commands* to
fixed-function engines rather than compute themselves.

On Wormhole and Blackhole a Tensix holds five "baby" RISC-V cores — two for data
movement, three (the TRISCs) driving unpack / math / pack. **Quasar restructures
this substantially.**

| per Tensix cluster | Wormhole / Blackhole | Quasar |
|---|---|---|
| data-movement cores | 2 (BRISC, NCRISC) | 8 (DM0–DM7), 64-bit |
| compute engines | 1 × 3 TRISCs | 4 "Neos" × 4 TRISCs, 32-bit |
| processors in total | 5 | 24 |
| L1 scratchpad | 1.5 MB | 4 MB, shared by all of them |
| NOCs | 2 | 1 |
| buffer primitive | Circular Buffer (CB) | Dataflow Buffer (DFB) |
| you target… | individual cores | the cluster |

Sources: `temp_quasar_api.hpp:15-39`, `core_config.h:18-44`, and the three SoC
descriptors. **There is no prose spec.** The one document the tree points at for the
Quasar-vs-Blackhole delta, `CHANGES_FROM_BLACKHOLE.md`, does not exist anywhere on
this filesystem — don't spend an afternoon looking for it.

Two rows bite constantly. Quasar reserves **DM0 and DM1** for runtime use, so user
kernels land on DM2–DM7. And the kernel classes are hard-split by arch:
constructing a plain `DataMovementKernel` on Quasar throws —

```
DataMovementKernel is not supported on Quasar. Use QuasarDataMovementKernel instead.
                                                          (kernel.hpp:417-419)
```

That single assertion is how you discover that an op you're calling was never
ported. It is still the most common Quasar failure mode.

### Inside the engine

The Tensix Engine runs its own ISA — not RISC-V. The RISC-Vs issue instructions into
it. Work flows:

**Unpacker** (L1 → SrcA/SrcB, converting format in hardware) → **FPU** (matrix unit,
natively 16×16) or **SFPU** (vector unit, 32 lanes on Quasar) → **Dst** register →
**Packer** (Dst → L1).

Each is driven by a different TRISC, which is why one operation needs several
kernels.

### Why everything is 32×32

A **tile** is 32×32 elements, subdivided into four 16×16 *faces* because the matrix
engine natively multiplies 16×16. The deeper reason is locality: in a row-major
layout the element below you is a whole row away, so the hardware must buffer entire
rows. Tiling puts it 32 elements away.

This is also where a whole class of bugs lives — see §7.

---

## 3. The layers: from Python to a core

Each layer knows strictly less about your model than the one above it. By the time
you reach the bottom there are no ops, only addresses.

```
  forge               ONNX / PyTorch in; builds a graph; picks the compiler config
  tt-mlir compiler    TTIR -> TTNN dialect -> flatbuffer. Chooses layouts,
                      inserts permutes and layout changes
  ----------------- compile time ends - run time begins -----------------
> tt-mlir runtime     Walks the flatbuffer, calls a C++ function per op.
                      THIS IS WHERE QUASAR DISPATCH LIVES
  ttnn                The op library: matmul, conv2d, permute. Picks a
                      program factory per op
  tt-metal            Programs, kernels, buffers. Compiles kernels JIT,
                      writes them into L1
  UMD                 User Mode Driver. Cluster -> Chip -> TTDevice.
                      Reads and writes addresses. Knows nothing of ops
  device              silicon | craq-sim .so | ZeBu over a socket
```

The compile/run boundary is worth memorising: it explains most confusion about where
a failure belongs. The compiler emits *identical* IR for Quasar as for other
architectures — every Quasar-specific decision happens below that line, in the
runtime.

Two things confusingly share the name **TTNN**: the MLIR *dialect* (compile time, in
tt-mlir) and the C++ *op library* (run time, in tt-metal). They are not the same
thing.

> **Exception worth knowing.** The compile/run boundary holds for op *lowering*, but
> not for the optimiser. The optimised pipeline runs an op-model constraint query
> that reaches a mainline program factory and TT_FATALs on Quasar
> (`OperationValidationAndFallback`). So the optimised pipeline cannot compile
> conv2d at all, and the default pipeline is the only one that gets through. That is
> a compiler-side gap, above the line.

---

## 4. The emulator path: how a register write reaches ZeBu

When the simulator path is a directory, this is what stands between your Python and a
flip-flop. Every step can fail independently, which is why emulation bring-up feels
fragile.

1. **Your process opens the device.** tt-metal constructs a UMD `Cluster` with
   `ChipType::SIMULATION`, handing it the directory. — `tt_cluster.cpp:398-417`

2. **A socket server starts — before anything is spawned.** UMD opens an NNG `pair1`
   TCP socket and listens. `NNG_SOCKET_LOCAL_PORT` fixes the port; otherwise it picks
   a random one in 50000–59999. — `simulation_host.cpp:48-96`
   (NNG = nanomsg-next-generation, v1.8.0)

3. **`run.sh` is launched and detached.** From the simulator directory. It picks a CI
   or dev variant, then `ssh`es to `soc-l-04` to reach the RTL testbench repo.
   — `rtl_sim_communicator.cpp:94-118`

4. **aether sets up the emulation environment.** aether is the Quasar SoC RTL /
   verification repo. `setup_emu_env.sh <config>` selects the ZeBu model for a grid
   config, then `make -C verification/emu test TESTCASE=test_umd_remote
   AETHER_CONFIG=1x3`. — `quasar-1x3_run_dev.sh:38-44`

5. **A batch scheduler reserves emulator modules.** ZeBu hardware is scarce and
   licensed by token, so the job queues. Physical modules get assigned
   (`HERO:LEAF_zs5_U8.M1`, `U8.M2`) and there is a one-job-at-a-time limit
   (`HERO:JOBCOUNT_zs5/1`). Scheduled by **Altair VOV**, not LSF — see §9.

6. **The testbench dials back and the model loads.** `test_umd_remote` connects to
   `NNG_SOCKET_ADDR`, and your process unblocks. **This wait is the 38–66 s you see
   before anything happens** — and if the hardware is held by someone else, it is
   however long the queue is. — `rtl_sim_communicator.cpp:127-138`

7. **Reads and writes become AXI transactions.** UMD encodes each access as a
   FlatBuffer `DEVICE_COMMAND`; the far end turns it into AXI (512-bit data, 56-bit
   address) into the Quasar RTL. — `simulation_device.fbs:3-41`,
   `xtor_amba_master_axi4_svs`

### Waiting on the queue looks identical to a hang

Both present as `Waiting for ack msg from remote...` forever. The way to tell them
apart is the remote job's own log:

```
ssh soc-l-04
tail .../work/aether/verification/emu/emu_out/1x3/test_umd_remote_1x3_0/emu.log

vovsh(...) Job 050853127 waiting for 'SW HERO:JOBCOUNT_zs5/1'
```

That is someone else holding the hardware — nothing is wrong with your setup.

---

## 5. Two descriptors, two different questions

These are constantly confused, and confusing them produces errors that look like
nothing to do with configuration.

| | SoC descriptor | Core descriptor |
|---|---|---|
| owned by | UMD | tt-metal |
| answers | what tiles exist on this chip, and where | of those, which are the user's |
| lives in | `tt_metal/soc_descriptors/` — **or the simulator directory** | `tt_metal/core_descriptors/` |
| key fields | `grid`, `functional_workers`, `dram`, `worker_l1_size` | `compute_with_storage_grid_range`, `dispatch_cores` |

Two consequences that cost real time:

**In simulator mode the SoC descriptor does not come from the repo.** It is read from
`$TT_METAL_SIMULATOR/soc_descriptor.yaml`, beside the `.so` or inside the emulator
directory. Editing the repo copy does nothing.

**The core descriptor is chosen *from* the SoC descriptor's grid size.** If either
grid dimension is ≤ 2 you get a small-simulation file, otherwise the full one
(`core_descriptor.cpp:63-84`). These two files must stay consistent — shrinking
`functional_workers` alone desynchronises them and dies in the allocator with
`No core coordinate found at location: (0, 1, TENSIX, LOGICAL)`.

### The emulator's actual descriptor

`emu-quasar-1x3/soc_descriptor.yaml`, in full — this is the whole machine:

```yaml
grid:  { x_size: 1, y_size: 3 }
dram:                [[0-0]]
functional_workers:  [0-1]
router_only:         [0-2]
worker_l1_size:      4194304
dram_bank_size:      1073741824
arch_name: QUASAR
```

"1×3" is literally the NOC grid: **three nodes — one DRAM, one worker, one router.**
That is the minimum topology that is still a working Quasar. Everything that is a
property of a Quasar *core* matches craq-sim exactly (L1 size, all three alignments,
unreserved bases, dst capacity 16 tiles, 64 CBs, 4 compute threads, 6 DM threads, the
five supported data types, the six tile sizes). Only the *count* differs.

The reason is ZeBu: it emulates RTL gate by gate, so 32 workers isn't practical. They
kept the core faithful and cut the topology. **Core-accurate, topology-reduced.**

### A useful lever on craq-sim

To change how many cores your ops actually use, edit
`compute_with_storage_grid_range` in the *core* descriptor — not the worker list.
Setting `end: [2, 2]` in `quasar_simulation_8x4_arch.yaml` confines craq-sim to a
single core (verified: one core in the DFB writes instead of 32) at full simulator
speed. That makes "is this a core-count problem?" a 20-second question.

This cannot widen the emulator — its grid comes from the aether config, and `2x3`
and `9x4_DM` configs exist if you need more than one core there.

---

## 6. Runtime modes: slow and fast dispatch

"Dispatch" is how a compiled program — kernel binaries, buffer configs, runtime args,
and the go signal — gets from the host into each core's L1, and how completion comes
back.

**Fast dispatch** puts firmware on dedicated cores and lets the device feed itself
from a ring buffer. It is faster and it *costs you cores* — that is what
`dispatch_cores` reserves.

**Slow dispatch** has the host do every step over UMD: write the binaries into L1,
write the launch message, then busy-poll each core's L1 until it reports done.

Simulation and emulation use slow dispatch, and the reason is more specific than
"because it's a simulator". The 1×3 emulator has exactly **one** Tensix core and
`dispatch: []` — there is no spare core to host a prefetcher and dispatcher, so the
fast-dispatch pipeline cannot be built at all. It is enforced by configuration, not
by code.

There is also a practical reason: in slow dispatch every step is a synchronous UMD
round trip, and over a socket to ZeBu that is milliseconds each. UMD accordingly
makes the completion wait **infinite** on a simulator rather than timing out — which
is why a hang there hangs forever.

---

## 7. The bug factory: layouts, and the padding nobody asked for

Two orthogonal choices, routinely conflated:

- **Tensor layout** — how elements map to pages. `ROW_MAJOR` (one row per page) or
  `TILE` (one 32×32 tile per page).
- **Memory layout** — how pages are distributed. `INTERLEAVED` (round-robin across
  banks) or *sharded* (height / width / block, pinned to specific cores).

Now the part that generates bugs. In `TILE` layout the alignment *is* the tile shape,
and the last two dimensions are rounded up to it. A tensor whose logical height is
**49** is physically **64** — 15 rows of padding that no one requested and that
`logical_shape` never mentions.

More precisely: a TILE tensor addresses a row at
`prod(leading) * roundup(H, 32) + h`, so the implicit padding after a sub-tile H is
**part of the address**. Every op must then decide, correctly, whether it means the
logical extent or the padded one.

All three defects found in this bring-up were exactly that, and **none was visible in
a single-op test**:

| defect | symptom | why probes missed it |
|---|---|---|
| tilize corrupts unaligned tensors | emulator-only wrong data | craq-sim's tilize is correct |
| tiled reshape only relabels | `[1,2,2,256] → [1,1,4,256]` wrong from element 512 | the error surfaced two ops downstream |
| untilize loses data past tile column 16 | `[1,1,4,1024]` lost a quarter | every probe was ≤ 8 tiles wide |

That last row is the lesson: the ops were all covered, the **widths** weren't.
`conv2d_1x1_sp2` is 256→1024, so its untilize is 8 tile columns and exact;
`conv1x1_down` is 1024→256 and its untilize is 32 wide and broken. Probe shapes have
to span the model's real channel counts, not just its real operators.

### The technique that actually finds these

`Tensor::to_vector<T>()` returns elements in *logical row-major order regardless of
physical layout*. That gives you exact invariants to assert inside the runtime:

- a **reshape** must preserve logical order element for element
- a **layout change** must preserve it too — in **both** directions
- a **permute** has an exact CPU reference

Instrument every op in a failing chain and one run names the guilty one, instead of
guessing and re-running. This is now wired in behind env vars:

```
TTMLIR_OP_TRACE=1        one flushed line per op, works in a release build
TTMLIR_LAYOUT_CHECK=1    to_layout logical identity, both directions
TTMLIR_RESHAPE_CHECK=1   reshape logical identity
TTMLIR_PERMUTE_CHECK=1   permute against an exact CPU reference
TTMLIR_CONV_CHECK=1      conv internals: weight, activation, matmul, tail, spec
```

`TTMLIR_LAYOUT_CHECK` is what found the untilize bug, in one run, after a day of
narrowing by hand.

**One trap in writing these checks.** A check that compares an op against its own
operands cannot see a corrupt operand — both sides share the error and agree. The
conv matmul check read 0.999997 while the activation feeding it was a quarter wrong.
Always compare against something *outside* the op.

---

## 8. What each target costs, and what it tells you

Same forge stack, same runtime binary, both targets, as of 15 Sep 2026.

| op | craq-sim | ZeBu emulator |
|---|---|---|
| `relu` | 1.000000 | 1.000000 |
| `matmul` | 0.999824 | 0.999860 |
| `linear` | 0.999728 | 0.999711 |
| `permute` NCHW→NHWC (sub-tile) | 0.999996 | 0.999996 |
| `conv2d` 1×1 | 0.999996 | 0.999996 |
| `conv2d` 3×3 | 0.999993 | 0.999992 |
| `max_pool2d` | 0.999940 | 0.999940 |
| `mean` (dim −2) | 0.999830 | 0.999687 |

Both targets now agree to five or six digits on every op. That agreement is the
point: it means the fixes were to real defects rather than per-target workarounds.

### Cost, and the consequence

- ZeBu model load: **38–66 s**, paid once per process
- Marginal cost: **20–24 s per program launch**
- Queue wait: **0 to 30+ minutes**, depending on who else is using the hardware
- Remote session cap: **`EMULATOR_TIMEOUT=5000`** (1.4 h) in
  `quasar-1x3_run_dev.sh`, overridable

**Counting launches correctly.** craq-sim logs `Writing DFB config` *per core*. For
the same 16-op graph it emitted 317 lines across 32 cores where the emulator emitted
32 across one. So divide craq-sim's count by ~10 to get a one-core emulator estimate.
The original version of this guide missed that and over-stated ResNet-50 by 10×.

### Can the emulator run the whole of ResNet-50?

**Not yet — five attempts, none finished.** Best reached 38 of 549 ops. Measured
rates and what they project to:

| configuration | measured | projects to |
|---|---|---|
| random weights, 8×8 | 17 ops in 18 min | ~7.5 h |
| pretrained + real image | 38 ops in 65 min | ~15.7 h |
| random weights, 8×8, after the fused conv | not measured | ~2.6 h |

The fused im2col convolution (one matmul with `K = kh*kw*C_in` instead of one matmul
per kernel tap) cut program launches **10,623 → 3,621** on craq-sim. Applied to the
7.5 h figure that projects to roughly **2.6 h**, inside a raised cap — but it has
never been given a clean run.

**Input size is not the lever for a pretrained model.** ResNet-50 is 25.6M
parameters = **51.1 MB in bf16**, plus 52 const-eval programs to fold BatchNorm, and
none of that shrinks with the image. Going 224 → 8 does nothing about it. This is why
the pretrained run spent 65 minutes without leaving the upload prologue.

So: emulation is for single ops and small compositions; whole models belong on
craq-sim. If you must batch emulator probes, run them in **one** process so the model
load is paid once — but put the heavy ones last, because a heavy decomposition can
leave the next model in the session reading memory nobody wrote (ZeBu DRAM is not
zero-initialised, so that reads as `inf` rather than as zero).

### End-to-end, for calibration

Pretrained ResNet-50, a real photograph, on craq-sim at 32×32 — whole model, 549 ops,
25.3 s:

```
device_vs_golden_pcc=0.999532   device_vs_bf16cpu_pcc=0.999622
top1_golden=257  top1_device=257  MATCH=True
```

Same top-1 and same top-5 set as CPU. Note 96×96 is the smallest input that still
identifies the breed correctly (Samoyed 98.24%); at 32 it becomes Great Pyrenees, so
that run is a faithful *numerical* match rather than a correct classification.

---

## 9. Corrections to the setup doc

Checked against this environment. Each of these will otherwise cost you an afternoon.

**"Clean up hung jobs with `module load lsf`, then `bjobs` / `bkill`."**
The Quasar ZeBu job is scheduled by **Altair VOV**, not LSF. In the emulation logs the
only `lsf` strings are git branch names. `bjobs` will show nothing — a 44-minute
orphan was live while `bjobs` reported "No unfinished job found". Use the web UI at
`soc-zebu-01:8766`, or list your own processes on `soc-l-04`. LSF is real, but for the
older Blackhole flow in UMD's `README.emu.md`.

**"`TT_UMD_SIMULATOR_PATH` is what Metal needs."**
**`TT_METAL_SIMULATOR`** is the only variable production code reads.
`TT_UMD_SIMULATOR` is read *only* by UMD's own gtests; `TT_UMD_SIMULATOR_PATH` only by
tt-llk. Set the wrong one and a gtest run reports `[ PASSED ] 0 tests` with exit
code 0 — a silent success. Always check the test count, not the exit code.

**"`TT_METAL_SLOW_DISPATCH_MODE=0` turns slow dispatch off."**
It does not. The handler ignores the value entirely — any setting of the variable
enables slow dispatch. Unset it instead. — `rtoptions.cpp:776-783`

**"`storage_cores` configures storage-only cores."**
No such field exists anywhere in this tree. The word survives only inside the name
`compute_with_storage_grid_range`.

---

## 10. Habits worth having before you debug

- **Read the stack, not the test name.** "conv2d is broken" has meant, at different
  times: mainline permute, a matmul factory, a readback untilize, a dropped fused
  activation, a tiled reshape, and a device untilize. The op that fails is frequently
  not the op that is wrong.

- **Check whether your probe can fail.** A test with all-positive weights cannot
  detect a missing ReLU. A tensor whose dimensions are all 32 cannot detect an axis
  error. A probe 8 tile columns wide cannot detect a bug that starts at column 16.
  All three happened here, and all three reported PASS.

- **Byte-identical results mean your change wasn't in the path.** If two different
  edits produce exactly the same wrong number, stop editing and go measure. This was
  right three times in a row.

- **Test composition, not just ops.** A defect that made every single-op probe pass
  showed up immediately in one real ResNet block.

- **Compare against the right floor.** Device-vs-fp32 measures correctness *and*
  precision at once. Run the same model on CPU in bf16 first: that is what bf16
  costs, and the device's residual should be judged against it, not against zero.
  bf16 on CPU costs ~1% and stays flat with depth; an error that *grows* with depth
  is a defect, not rounding.

- **Measure against a consistent build.** `deploy_only.sh` rebuilds only the runtime,
  so the installed compiler drifts from source. A full `--target install` in the
  middle of a measurement campaign rewrote the graphs and made every passing probe
  fail, with identical execution traces and different numbers. Do the full install
  *before* a campaign, never during one.

- **Distrust "it passes on the simulator"** for anything about hardware behaviour. It
  is strong evidence about arithmetic and weak evidence about the machine.

---

*Compiled from the tt-metal, tt-mlir, UMD and craq-sim trees in this workspace, plus
emulator run logs of 11–15 Sep 2026. Measurements are cited where they exist;
unverified items are flagged rather than smoothed over. Full measurement record in
`add_rs/OP_STATUS_2026-09-10.md`.*
