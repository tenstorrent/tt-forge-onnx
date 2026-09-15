# ZeBu / aether runbook — build, then run one op

Everything below was verified against this workspace on **2026-09-15**. Paths and
port numbers are real, not placeholders. Companion to
`ZEBU_AETHER_FIELD_GUIDE.md`, which explains *why* each step exists.

---

## Part 1 — What ZeBu and aether actually are

### ZeBu

**ZeBu is hardware.** A Synopsys ZeBu-Server5 emulator — a rack of FPGAs, in another
building, that the real Quasar RTL is compiled onto and executed gate by gate.

It is not a simulator in the software sense. There is no model of the chip written in
C++; there is the actual RTL, running. That is what makes it trustworthy about
hardware behaviour and unbearably slow about everything else:

- ~20–24 s per program launch
- 38–66 s to load the model, once per process
- physically shared, licensed by token, one job at a time (`HERO:JOBCOUNT_zs5/1`)

You never touch ZeBu directly. Your process talks to it over a socket.

### aether

**aether is a git repo** — `git@yyz-gitlab.local.tenstorrent.com:tensix/soc/aether.git`,
the Quasar SoC RTL and verification environment. It lives on `soc-l-04`, cloned at
`/proj_sw/user_dev/$USER/work/aether`, pinned to tag `aether-main-v2026.W21.0`.

aether's job is to stand between your socket and the ZeBu box:

- `bin/setup_emu_env.sh <config>` picks the compiled ZeBu model for a grid config
- `make -C verification/emu test TESTCASE=test_umd_remote AETHER_CONFIG=1x3` submits
  the job and runs the testbench
- the testbench, `test_umd_remote`, dials **back** to your socket and then relays
  every read and write into the RTL as AXI transactions

So the full chain, which is worth holding in your head because each link fails
differently:

```
  your python (container, listens on :5555)
    <- docker port-forward from  bgd-lab-NN:<debuda_port>
    <- ssh from your container to soc-l-04, which runs run.sh
    <- aether on soc-l-04 submits to the Altair VOV scheduler
    <- ZeBu-Server5 runs the Quasar RTL
```

### The one-line summary

> **ZeBu** is the emulator hardware. **aether** is the repo that knows how to put
> Quasar RTL on it and expose it over a socket. **UMD** turns your reads and writes
> into that socket traffic. You run ordinary Python.

### And what it is *not*: craq-sim

Don't confuse the two. `TT_METAL_SIMULATOR` decides which you get, purely by
**file extension** (`simulation_device_factory.cpp:15-21`):

| | craq-sim | ZeBu |
|---|---|---|
| `TT_METAL_SIMULATOR` | path to a **`.so`** | path to a **directory** |
| what it is | functional C++ model, `dlopen`ed in-process | real RTL on FPGAs |
| workers | 32 (`grid 4x8`) | **1** (`grid 1x1`) |
| DRAM channels | 2 | 1 |
| full ResNet-50 | ~25 s | never finished (best: 38 of 549 ops) |
| use it for | whole models, fast iteration, arithmetic | single ops, hardware behaviour |

Develop on craq-sim. Confirm on ZeBu.

---

## Part 2 — Build

Two independent things get built. **You usually only need the second.**

### 2a. The emulator config directory — one-time, already done

```
/proj_sw/user_dev/ctr-lelanchelian/tt-umd-simulators/build/emu-quasar-1x3/
    run.sh                  entry point UMD spawns
    quasar-1x3_run_dev.sh   ssh's to soc-l-04, drives aether
    soc_descriptor.yaml     the machine: 1 DRAM + 1 worker + 1 router
```

This is *not* a compiled artifact — it is scripts plus a YAML. Nothing to rebuild
unless you want a different grid (`2x3` and `9x4_DM` configs exist).

Verify it is intact:

```bash
ls /proj_sw/user_dev/$USER/tt-umd-simulators/build/emu-quasar-1x3/
cat /proj_sw/user_dev/$USER/tt-umd-simulators/build/emu-quasar-1x3/soc_descriptor.yaml
```

### 2b. The tt-mlir runtime — this is what you rebuild

Your Quasar op dispatch lives in `third_party/tt-mlir/runtime/lib/ttnn/`. After
editing it:

```bash
cd /proj_sw/user_dev/$USER/tt-forge-onnx/third_party/tt-mlir
source env/activate

cmake --build build --target TTMLIRRuntime          # ~40-90 s incremental
```

**Then deploy it, and this step is mandatory.** `cmake --build` leaves the *installed*
library stale, and forge loads the installed one — so skipping this makes your change
silently do nothing:

```bash
SRC=build/runtime/lib/libTTMLIRRuntime.so
DST=build/install/lib/libTTMLIRRuntime.so
cp -f "$SRC" "$DST.new" && mv -f "$DST" "$DST.bak" 2>/dev/null; mv -f "$DST.new" "$DST"
[ "$(stat -c %s $SRC)" = "$(stat -c %s $DST)" ] && echo "deploy OK" || echo "SIZE MISMATCH"
```

Three things about that copy, each learned the hard way:

- **Copy to a temp name and `mv`.** A plain `cp` over a `.so` that another process
  has mmap'd can truncate it to 0 bytes on NFS **and exit 0**. The next run then dies
  with "file too short".
- **Always check the size.** It is the only cheap proof the copy happened.
- Deploying while a run is live is safe — the running process keeps the old inode.

Confirm what forge will actually load:

```bash
ldd $(python3 -c "import forge,os;print(os.path.dirname(forge.__file__))")/_C.so \
  | grep -E "TTMLIRRuntime|TTMLIRCompiler"
```

### 2c. If you changed the compiler, not just the runtime

`--target TTMLIRRuntime` does not rebuild `libTTMLIRCompiler.so`, and forge loads
that too. If you touched `lib/` rather than `runtime/`:

```bash
cmake --build build --target install -j 24        # 10+ min, rebuilds everything
```

> **Do this before a measurement campaign, never in the middle of one.** A mid-run
> full install rewrote the graphs and made every passing probe fail — identical
> execution traces, different numbers. It cost hours of bisecting the wrong thing.

---

## Part 3 — Run one op on craq-sim first

Always. It is ~20 s per op, free, and it tells you whether the problem is your code
or the emulator.

```bash
cd /proj_sw/user_dev/$USER/tt-forge-onnx
source env/activate
source ./scripts/quasar_sim_env.sh          # must be SOURCED, not executed

cd "$TT_METAL_HOME"                         # required, see note
python $OLDPWD/add_rs/probe_exec_one_op.py conv2d_3x3_sp2
```

Expected last line:

```
[conv2d_3x3_sp2] RESULT: PASS pcc=0.999996 max_abs_err=0.00841 err/scale=0.00563
```

Notes that matter:

- **`cd $TT_METAL_HOME` is not optional.** Several Quasar binary ops register a
  *relative* compiler include path, so kernel compilation fails if you start
  elsewhere.
- `quasar_sim_env.sh` renames the SoC descriptor into the craq-sim directory for you
  and refuses to run if the `.so` isn't a QSR (`TT_VERSION=2`) build.
- 77 op cases are available; `grep 'if op ==' add_rs/probe_exec_one_op.py` lists them.
- To use the newer simulator build: `export QUASAR_SIM_DIR=/proj_sw/user_dev/$USER/craq-sim-qtip/src/_out/release_qsr`
  before sourcing. Both builds agree on every op; they disagreed on the full model.

---

## Part 4 — Run one op on the ZeBu emulator

### Step 1. Check nothing of yours is already holding the hardware

**Do this first, every time.** A killed run orphans the remote job, which keeps its
ZeBu tokens and makes your next run wait forever on `Waiting for ack msg from
remote...`.

```bash
# local
ps -eo pid,comm,args --no-headers | awk '$2 ~ /^python/ && /probe_|run_resnet50_/ {print $1}'

# remote — this is the check that actually works
ssh soc-l-04 "ps -u \$USER -o pid,etime,cmd | grep -E 'zrun|vovsh|verification/emu' | grep -v grep"
```

Kill anything left over, then **wait ~4 minutes** before reconnecting — starting
sooner gives `hero_adapter.tcl:82: child process exited abnormally`.

> `bjobs` is useless here. That is LSF; this job runs under **Altair VOV**. A
> 44-minute orphan was live while `bjobs` said "No unfinished job found". Use the
> remote `ps` above, or the web UI at http://soc-zebu-01:8766/.

> **Never `pkill -f` a pattern that appears in your own command line.** It matches
> your own shell and kills it. This bit me twice. Select on `comm` instead:
> `awk '$2 ~ /^python/'`.

### Step 2. Set the environment

```bash
source /proj_sw/user_dev/$USER/emu-run/env_quasar_emu.sh
```

It prints what it derived. Verified right now:

```
NNG_SOCKET_ADDR       = tcp://bgd-lab-17:53396
NNG_SOCKET_LOCAL_PORT = 5555
TT_METAL_SIMULATOR    = /proj_sw/user_dev/ctr-lelanchelian/tt-umd-simulators/build/emu-quasar-1x3
```

**`NNG_SOCKET_ADDR` is the only thing that changes between reservations** — machine
name and debuda port. The script derives both from the container hostname
(`bgd-lab-17-special-<user>-for-reservation-83726` → job 83726) by asking `ird` on
the host, because `ird` does not exist inside the container.

Check it by hand if a run won't connect:

```bash
ssh bgd-lab-17 "ird list"
#  MACHINE      JOB ID   SSH PORT   DEBUDA PORT
#  bgd-lab-17   83726    48396      53396        <- ADDR = tcp://bgd-lab-17:53396
```

If derivation fails: `export EMU_DEBUDA_PORT=53396` before sourcing.

> **The env-var names are a trap.** `TT_METAL_SIMULATOR` is the only one production
> code reads. `TT_UMD_SIMULATOR` is read only by UMD's gtests; `TT_UMD_SIMULATOR_PATH`
> only by tt-llk. Set just the wrong one and a gtest run prints
> `[ PASSED ] 0 tests` with **exit code 0** — a silent success. Always check the
> test *count*, not the exit code. (The script sets all three, so this only bites
> when you set them by hand.)

> `TT_METAL_SLOW_DISPATCH_MODE` **ignores its value.** Any setting enables slow
> dispatch; `=0` does not disable it. Unset it instead. — `rtoptions.cpp:776-783`

### Step 3. For a forge/ONNX op, override `TT_METAL_HOME`

`env_quasar_emu.sh` points at the standalone `$USER_DEV/tt-metal`, which is right for
UMD and ttnn tests. For forge op probes you need the tt-mlir submodule's tt-metal —
the one your runtime was built against:

```bash
export TT_METAL_HOME=/proj_sw/user_dev/$USER/tt-forge-onnx/third_party/tt-mlir/third_party/tt-metal/src/tt-metal
export PYTHONPATH=$TT_METAL_HOME
```

### Step 4. Run it

```bash
cd /proj_sw/user_dev/$USER/tt-forge-onnx
source env/activate                                   # forge, keeps the emu vars
cd "$TT_METAL_HOME"

TT_METAL_CACHE=/proj_sw/user_dev/$USER/emu-run/cache_emu \
TTMLIR_OP_TRACE=1 \
timeout 5400 python -u /proj_sw/user_dev/$USER/tt-forge-onnx/add_rs/probe_exec_one_op.py \
    conv2d_3x3_sp2 2>&1 | tee /tmp/emu_op.log
```

- **`TT_METAL_CACHE`** separate from craq-sim's, or the two fight over kernel
  artifacts.
- **`TTMLIR_OP_TRACE=1`** prints one flushed line per op. At ~20 s a launch you
  cannot afford to bisect a hang by re-running — this names the op on the first
  attempt. Costs nothing when unset.
- **`timeout`** always. A hang on a simulator target waits *forever* by design: UMD
  makes the completion wait infinite rather than time out.

### Step 5. Read the progress

```bash
grep -c "Writing DFB config" /tmp/emu_op.log     # program launches so far
grep     "optrace"           /tmp/emu_op.log     # which op it is on
grep     "RESULT:"           /tmp/emu_op.log     # the answer
```

Timeline for a healthy single-op run:

| phase | duration | what you see |
|---|---|---|
| queue for hardware | 0 → 30+ min | `Waiting for ack msg from remote...` |
| model load | 38–66 s | `Notification handler thread started` |
| first program | ~7 min | one `Writing DFB config`, then a long pause |
| remaining programs | 20–24 s each | `Writing DFB config` accumulating |
| `conv2d_3x3_sp2` total | ~20 min | 16 ops, 32 launches |

### Step 6. Clean up — always

```bash
ps -eo pid,comm,args --no-headers | awk '$2 ~ /^python/ && /probe_/ {print $1}' | xargs -r kill
sleep 4
ssh soc-l-04 "pkill -u \$USER -f 'verification/emu'; pkill -u \$USER -f zrun; pkill -u \$USER -f vovsh"
sleep 5
ssh soc-l-04 "ps -u \$USER -o pid,cmd | grep -E 'zrun|vovsh|verification/emu' | grep -v grep"
```

Even a **clean exit** can orphan the remote job (SOCEMU-95). Verify the remote is
empty before you walk away — otherwise the next person, or the next you, waits
forever.

---

## Part 5 — Telling a queue wait apart from a hang

Both look identical from your side: `Waiting for ack msg from remote...`, forever.
The remote job's own log distinguishes them:

```bash
ssh soc-l-04 'tail -5 $(ls -t /proj_sw/user_dev/$USER/work/aether/verification/emu/emu_out/1x3/*/emu.log | head -1)'
```

- `Job 050853127 waiting for 'SW HERO:JOBCOUNT_zs5/1'` → **someone else has the
  hardware.** Nothing is wrong. Wait, or come back later.
- `Job ... waiting for 'SW HERO:LEAF_zs5_U8.M2 ...'` → same thing, waiting on
  specific modules.
- `FAIL=1` / `hero_adapter.tcl: child process exited abnormally` → you reconnected
  too soon after a kill. Clean up and wait 4 minutes.
- nothing at all, and no remote process → the job died. Re-run.

Also note `/proj_sw` on `soc-l-04` is a **different mount** from `/proj_sw` in the
lab container. The aether clone that `run_dev.sh` expects at `$USER/work/aether` is
invisible from your container — that looks like a missing clone and is not.

---

## Part 6 — Copy-paste: one op, start to finish

```bash
# ---- 0. clean slate -------------------------------------------------------
ssh soc-l-04 "ps -u \$USER -o pid,cmd | grep -E 'zrun|vovsh|verification/emu' | grep -v grep"
#    if that printed anything: kill it, then sleep 240

# ---- 1. build + deploy the runtime ---------------------------------------
cd /proj_sw/user_dev/$USER/tt-forge-onnx/third_party/tt-mlir
source env/activate
cmake --build build --target TTMLIRRuntime || exit 1
SRC=build/runtime/lib/libTTMLIRRuntime.so; DST=build/install/lib/libTTMLIRRuntime.so
cp -f "$SRC" "$DST.new" && mv -f "$DST.new" "$DST"
[ "$(stat -c %s $SRC)" = "$(stat -c %s $DST)" ] || { echo "DEPLOY FAILED"; exit 1; }

# ---- 2. sanity-check on craq-sim (20 s, free) ----------------------------
cd /proj_sw/user_dev/$USER/tt-forge-onnx
source env/activate && source ./scripts/quasar_sim_env.sh
cd "$TT_METAL_HOME"
python /proj_sw/user_dev/$USER/tt-forge-onnx/add_rs/probe_exec_one_op.py conv2d_3x3_sp2

# ---- 3. same op on ZeBu (~20 min) ----------------------------------------
source /proj_sw/user_dev/$USER/emu-run/env_quasar_emu.sh
export TT_METAL_HOME=/proj_sw/user_dev/$USER/tt-forge-onnx/third_party/tt-mlir/third_party/tt-metal/src/tt-metal
export PYTHONPATH=$TT_METAL_HOME
cd /proj_sw/user_dev/$USER/tt-forge-onnx && source env/activate
cd "$TT_METAL_HOME"
TT_METAL_CACHE=/proj_sw/user_dev/$USER/emu-run/cache_emu TTMLIR_OP_TRACE=1 \
  timeout 5400 python -u /proj_sw/user_dev/$USER/tt-forge-onnx/add_rs/probe_exec_one_op.py \
  conv2d_3x3_sp2 2>&1 | tee /tmp/emu_op.log
grep -E "optrace|RESULT:" /tmp/emu_op.log | tail -20

# ---- 4. clean up ---------------------------------------------------------
ssh soc-l-04 "pkill -u \$USER -f 'verification/emu'; pkill -u \$USER -f zrun; pkill -u \$USER -f vovsh"
sleep 5
ssh soc-l-04 "ps -u \$USER -o pid,cmd | grep -E 'zrun|vovsh' | grep -v grep"
```

---

## Part 7 — Batching, and why order matters

Running N ops as N processes pays the 38–66 s model load N times. `probe_batch.py`
runs several in **one** device session:

```bash
timeout 5400 python -u add_rs/probe_batch.py relu reshape multiply matmul conv2d_3x3_sp2
```

Two rules:

- **Heavy ops last.** A nine-tap decomposition left the *next* model in the session
  reading memory nobody wrote. ZeBu DRAM is not zero-initialised, so that reads back
  as `inf` rather than as zero — `conv2d_3x3_sp4` returned nan/inf purely because it
  ran after the pool. Alone, it gave 0.999991.
- **An `inf` from a later op in a batch is suspect until reproduced alone.**

---

## Part 8 — Debug switches built into the runtime

All env-gated, all no-ops when unset, all work in a release build.

| variable | what it checks |
|---|---|
| `TTMLIR_OP_TRACE=1` | one flushed line per op — use this on every emulator run |
| `TTMLIR_LAYOUT_CHECK=1` | `to_layout` preserves logical order, **both** directions |
| `TTMLIR_RESHAPE_CHECK=1` | reshape preserves logical order |
| `TTMLIR_PERMUTE_CHECK=1` | permute against an exact CPU reference |
| `TTMLIR_CONV_CHECK=1` | conv internals: weight, activation, matmul, tail, output spec |

They rest on one fact: `Tensor::to_vector<T>()` returns **logical row-major order
whatever the physical layout**, so reshape / layout / permute have exact invariants —
no tolerance, nothing to get wrong in a reference implementation.

`TTMLIR_LAYOUT_CHECK=1` found the untilize defect in a single run after a day of
narrowing by hand. If an op is wrong on the emulator and right on craq-sim, run that
first.

**One trap.** A check comparing an op against its own operands cannot see a corrupt
operand — both sides share the error and agree. The conv matmul check read 0.999997
while the activation feeding it was a quarter wrong. Compare against something
*outside* the op.

---

## Part 9 — Quick reference

| | |
|---|---|
| emulator config | `/proj_sw/user_dev/$USER/tt-umd-simulators/build/emu-quasar-1x3` |
| emulator env | `source /proj_sw/user_dev/$USER/emu-run/env_quasar_emu.sh` |
| craq-sim env | `source ./scripts/quasar_sim_env.sh` (from repo root) |
| op probe | `add_rs/probe_exec_one_op.py <case>` — 77 cases |
| batch probe | `add_rs/probe_batch.py <case> <case> ...` |
| runtime source | `third_party/tt-mlir/runtime/lib/ttnn/` |
| build | `cmake --build build --target TTMLIRRuntime` **+ copy to `build/install/lib`** |
| remote host | `soc-l-04` (aether) · `soc-zebu-01:8766` (job UI) |
| aether tag | `aether-main-v2026.W21.0`, config `1x3` |
| remote cap | `EMULATOR_TIMEOUT=5000` in `quasar-1x3_run_dev.sh`, overridable |
| measurements | `add_rs/OP_STATUS_2026-09-10.md` |

### Five ways this goes wrong

1. **Forgot to copy the runtime to `build/install/lib`** → your change does nothing,
   silently. Check the size matches.
2. **Didn't `cd $TT_METAL_HOME`** → kernel compilation can't find relative includes.
3. **Orphan remote job** → next run waits forever. `ps` on `soc-l-04`, not `bjobs`.
4. **Wrong `NNG_SOCKET_ADDR`** after a re-reservation → waits forever. Check
   `ird list` on the host.
5. **`pkill -f` matching your own command line** → kills your shell. Select on
   `comm`.
