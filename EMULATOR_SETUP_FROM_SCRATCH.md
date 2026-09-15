# Fresh machine → ZeBu emulator → one op through tt-forge-onnx

For a **new tt-forge-onnx build with the code already in remote**. Ends with a
single ONNX op executing on the Quasar RTL and printing a pcc.

Marked throughout:
- **[doc]** — from the official setup doc
- **[verified]** — checked against a working setup in this workspace
- **[correction]** — the official doc is wrong or misleading here; details inline

The corrections matter. Four of them cost an afternoon each.

---

## 0. Permissions — what you actually need

**[correction]** The doc lists seven groups. The emulator works with **three** of
them. Verified 2026-09-15: a full ResNet-50 op sweep ran on ZeBu while holding only

```
soc_users_er    4nmsf    risc_users
```

and **not** `hw_er`, **not** `7nmtsmc`. `7nmtsmc` is a TSMC process-node group
(PDK/process data) and gates nothing in the ZeBu path — its NDA takes 2–3 weeks and
you do not need to wait for it. Check with:

```bash
id -Gn | tr ' ' '\n' | grep -E 'soc_users_er|hw_er|4nmsf|7nmtsmc|risc_users'
```

Also needed **[doc]**:

- GitLab accounts on `yyz-gitlab.local.tenstorrent.com` **and**
  `aus-gitlab.local.tenstorrent.com` — sign in with **AD username**, not email,
  then add your SSH key under *User Settings → Edit Profile → SSH Keys*.
- Repo access via a **SOCINFRA** Jira ticket (component `gitlab`, assign to Gregory
  Czajkowski, put your Tenstorrent userid — not GitHub id — in the title):
  - yyz-gitlab groups `tensix/soc`, `tensix/tensix-hw`, `tensix/tenstorrent/vip`;
    projects `tek/tek`, `tensix/tt-umd-simulators`
  - aus-gitlab groups `riscv_external`, `soc`
- `ssh-copy-id soc-l-04` — aether is cloned and driven over ssh, so this must be
  passwordless.

Verify before going further:

```bash
ssh -o BatchMode=yes soc-l-04 'echo OK; hostname'
ssh -T git@yyz-gitlab.local.tenstorrent.com 2>&1 | head -2
```

---

## 1. Reserve a machine

**[doc]**

```bash
ird reserve --docker-image ghcr.io/tenstorrent/tt-llk/tt-llk-ird-ubuntu-22-04:latest \
            --timeout="max" compute --machine yyzc-swc08
```

**[verified]** `ird` exists only on the **host**, not inside the container. From
inside, reach it over ssh — you will need this again in step 4:

```bash
ssh $(hostname | sed -E 's/-special-.*$//') "ird list"
```

```
MACHINE      JOB ID   SSH PORT   DEBUDA PORT
bgd-lab-17   83726    48396      53396
```

Match the row on the reservation id embedded in your container hostname
(`bgd-lab-17-special-<user>-for-reservation-83726` → job 83726). **The debuda port
is the only value that changes between reservations.**

---

## 2. tt-umd-simulators — the emulator config

**[doc]**

```bash
cd /proj_sw/user_dev/$USER
git clone --recurse-submodules \
    git@yyz-gitlab.local.tenstorrent.com:tensix/tt-umd-simulators.git
cd tt-umd-simulators
cmake -B build -G Ninja      # errors — ignore, as the doc says
ninja -C build
```

This produces a **config directory**, not a compiled artifact **[verified]**:

```
/proj_sw/user_dev/$USER/tt-umd-simulators/build/emu-quasar-1x3/
    run.sh                  what UMD spawns
    quasar-1x3_run_dev.sh   ssh's to soc-l-04 and drives aether
    quasar-1x3_run_ci.sh    the CI variant
    soc_descriptor.yaml     the machine itself
```

**[correction]** The doc says to set `EMULATOR_TIMEOUT` in `run.sh`. It is not
there — `run.sh` only dispatches to the CI or dev script. It lives in
**`quasar-1x3_run_dev.sh`**, line 4:

```bash
EMULATOR_TIMEOUT=5000        # 1.4 h; the remote job is killed at this
ENABLE_WAVEFORM=0            # 1 only if you want waveforms
AETHER_CONFIG=1x3
```

Make it overridable rather than editing it each time:

```bash
sed -i 's/^EMULATOR_TIMEOUT=5000/EMULATOR_TIMEOUT=${EMULATOR_TIMEOUT:-5000}/' \
    /proj_sw/user_dev/$USER/tt-umd-simulators/build/emu-quasar-1x3/quasar-1x3_run_dev.sh
```

**Know what you are getting** **[verified]** — the whole emulated machine is three
NOC nodes:

```yaml
grid:  { x_size: 1, y_size: 3 }
dram:                [[0-0]]     # one DRAM
functional_workers:  [0-1]       # ONE worker core
router_only:         [0-2]
worker_l1_size:      4194304     # 4 MB, same as silicon
```

One Tensix. Per-core values match real Quasar exactly; only the count differs. Size
every expectation by rows-on-one-core. Wider configs exist (`2x3`, `2x3_DISPATCH`,
`2x4`, `9x4_DM`) if you need more than one.

---

## 3. aether — usually skip this

**[doc]** says to clone it yourself on soc-l-04.

**[correction]** `quasar-1x3_run_dev.sh` **already clones and tag-checks it for you**
on first run. Two differences from the doc, both verified:

| | doc | actual |
|---|---|---|
| path | `/proj_soc/user_dev/$USER` | **`/proj_sw/user_dev/$USER/work/aether`** |
| tag | `aether-main-v2026.W10.1` | **`aether-main-v2026.W21.0`** |

It is a 37 GB clone; let the script do it. If you want to read it:

```bash
ssh soc-l-04
cd /proj_sw/user_dev/$USER/work/aether
```

**[verified]** `/proj_sw` on `soc-l-04` is a **different mount** from `/proj_sw` in
your container. The aether clone is invisible from the container. That looks like a
missing clone and is not one — do not "fix" it by re-cloning.

---

## 4. Environment

Write this once; it survives re-reservations because it derives the only
per-reservation value itself.

```bash
mkdir -p /proj_sw/user_dev/$USER/emu-run
cat > /proj_sw/user_dev/$USER/emu-run/env_quasar_emu.sh <<'EOF'
#!/usr/bin/env bash
# source me from inside the IRD container
export USER_DEV=/proj_sw/user_dev/$USER

# derive host machine + debuda port from the container hostname via ird on the host
_ctr_host=$(hostname)
: "${EMU_HOST_MACHINE:=$(echo "$_ctr_host" | sed -E 's/-special-.*$//')}"
_job_id=$(echo "$_ctr_host" | sed -nE 's/.*-for-reservation-([0-9]+)$/\1/p')
if [ -z "${EMU_DEBUDA_PORT:-}" ]; then
    EMU_DEBUDA_PORT=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no \
        "$EMU_HOST_MACHINE" "ird list 2>/dev/null" \
        | awk -v job="$_job_id" '$0 ~ "debuda_port=" && $0 ~ job {
              for (i=1;i<=NF;i++) if ($i ~ /^"?ssh_port=/) {
                  sub(/.*debuda_port=/,"",$i); gsub(/"/,"",$i); print $i; exit } }')
fi
export EMU_HOST_MACHINE EMU_DEBUDA_PORT

export NNG_SOCKET_LOCAL_PORT=5555
export NNG_SOCKET_ADDR=tcp://${EMU_HOST_MACHINE}:${EMU_DEBUDA_PORT}

export TT_UMD_SIMULATOR_DIR=$USER_DEV/tt-umd-simulators/build/emu-quasar-1x3
export TT_METAL_SIMULATOR=$TT_UMD_SIMULATOR_DIR      # tt-metal reads THIS
export TT_UMD_SIMULATOR=$TT_UMD_SIMULATOR_DIR        # UMD gtests only
export TT_UMD_SIMULATOR_PATH=$TT_UMD_SIMULATOR_DIR   # tt-llk only
export ARCH_NAME=quasar
export CHIP_ARCH=quasar
export TT_METAL_SLOW_DISPATCH_MODE=1

echo "NNG_SOCKET_ADDR    = $NNG_SOCKET_ADDR"
echo "TT_METAL_SIMULATOR = $TT_METAL_SIMULATOR"
[ -n "$EMU_DEBUDA_PORT" ] || echo "WARNING: no debuda port; set EMU_DEBUDA_PORT by hand"
EOF
chmod +x /proj_sw/user_dev/$USER/emu-run/env_quasar_emu.sh
```

### Three env-var corrections, all of which fail silently

**[correction]** **`TT_METAL_SIMULATOR` is the only variable production code reads.**
The doc's tt-llk section implies `TT_UMD_SIMULATOR_PATH` is what Metal needs — it is
not; that is tt-llk only. `TT_UMD_SIMULATOR` is read only by UMD's own gtests. Set
just the wrong one and a gtest run prints

```
[  PASSED  ] 0 tests          # ...with exit code 0
```

Always check the test **count**, never the exit code.

**[correction]** **`TT_METAL_SLOW_DISPATCH_MODE` ignores its value.** Any setting
enables slow dispatch; `=0` does not disable it. Unset it instead
(`rtoptions.cpp:776-783`). It has to be on anyway: the 1×3 config has one Tensix and
`dispatch: []`, so there is no spare core to host a dispatcher.

**[verified]** The path in `TT_METAL_SIMULATOR` decides *which machine you get*, by
**file extension** (`simulation_device_factory.cpp:15-21`):

| value | you get |
|---|---|
| ends in `.so` | craq-sim — a functional model, in-process, no hardware |
| a **directory** | ZeBu — real RTL over a socket |

---

## 5. Build tt-forge-onnx

**[verified]** from `docs/source/getting_started_build_from_source.md`:

```bash
cd /proj_sw/user_dev/$USER
git clone --recurse-submodules https://github.com/tenstorrent/tt-forge-onnx.git
cd tt-forge-onnx

# toolchain venv (once)
cmake -B env/build env
cmake --build env/build

source env/activate
cmake -G Ninja -B build -DCMAKE_CXX_COMPILER=clang++-17 -DCMAKE_C_COMPILER=clang-17
cmake --build build
```

This builds forge, tt-mlir and tt-metal, and installs `_C.so` into the toolchain
venv (`forge/csrc/CMakeLists.txt:132-134` copies it and symlinks
`forge/forge/_C.so` at it).

Three traps, all from experience in this workspace:

- **[verified]** `third_party/tvm`'s `install.sh` is **not idempotent** and re-runs on
  every `cmake --build`. If it fails, delete the nested copies under
  `third_party/tvm/python/tvm/` (or `git clean` it) and rebuild.
- **[verified]** the tt-metal build needs **ccache** — without it it dies in
  `scaleout_tools` on a clang PCH visibility mismatch. `ccache --version` before you
  start. Also `unset MOLD MOLD_PATH` if set: a cached path under container-local
  `/usr/local` breaks the link.
- **[verified]** `/opt` is container-local overlay while `/proj_sw` is NFS. A build
  can report success and still leave you with no usable forge after a container
  change, because the venv lives in `/opt`. If everything suddenly dies at device
  discovery with `Unknown chip type Quasar`, check
  `stat /opt/ttforge-toolchain/venv/lib/python3.12/site-packages/forge/_C.so` before
  suspecting your code.

### When you change only the tt-mlir runtime

Quasar op dispatch lives in `third_party/tt-mlir/runtime/lib/ttnn/`. Rebuilding just
that is ~60 s, but **the copy step is mandatory** — `cmake --build` leaves the
*installed* library stale and forge loads the installed one, so skipping it makes
your change silently do nothing:

```bash
cd /proj_sw/user_dev/$USER/tt-forge-onnx/third_party/tt-mlir
source env/activate
cmake --build build --target TTMLIRRuntime

SRC=build/runtime/lib/libTTMLIRRuntime.so
DST=build/install/lib/libTTMLIRRuntime.so
cp -f "$SRC" "$DST.new" && mv -f "$DST.new" "$DST"      # temp name + mv, NOT plain cp
[ "$(stat -c %s $SRC)" = "$(stat -c %s $DST)" ] && echo "deploy OK" || echo "SIZE MISMATCH"
```

A plain `cp` over a `.so` another process has mmap'd can truncate it to 0 bytes on
NFS **and exit 0**. Always compare sizes.

---

## 6. Sanity-check on craq-sim first

Twenty seconds, no hardware, no reservation — and it tells you whether a later
failure is your code or the emulator. Worth it every time.

```bash
cd /proj_sw/user_dev/$USER/tt-forge-onnx
source env/activate
source ./scripts/quasar_sim_env.sh        # must be SOURCED, not executed or piped

cd "$TT_METAL_HOME"                       # REQUIRED — see below
python /proj_sw/user_dev/$USER/tt-forge-onnx/add_rs/probe_exec_one_op.py relu
```

```
[relu] RESULT: PASS pcc=1.000000 max_abs_err=0.00097 err/scale=0.00195
```

**[verified]** `cd $TT_METAL_HOME` is not optional: several Quasar binary ops register
a *relative* compiler include path, so kernel compilation fails if you start
anywhere else.

Do not pipe the `source` (`source env.sh | tail`) — a pipe runs it in a subshell and
the exports never reach you. I wasted a diagnostic cycle on exactly that.

---

## 7. Run one op on the emulator

### 7a. Confirm nothing of yours holds the hardware

**Every time.** A killed run — and sometimes a clean exit (SOCEMU-95) — orphans the
remote job, which keeps its ZeBu tokens and makes your next run wait forever.

```bash
ssh soc-l-04 "ps -u \$USER -o pid,etime,cmd | grep -E 'zrun|vovsh|verification/emu' | grep -v grep"
```

**[correction]** The doc says `module load lsf; bjobs; bkill`. **The Quasar job is
scheduled by Altair VOV, not LSF.** `bjobs` reports nothing useful — a 44-minute
orphan was live while it said "No unfinished job found". Use the `ps` above, or
http://soc-zebu-01:8766/ and kill your job by clicking your username. (LSF is real,
but for the older Blackhole flow in UMD's `README.emu.md`.)

If you kill something, **wait ~4 minutes** before reconnecting. Sooner gives
`hero_adapter.tcl:82: child process exited abnormally` → `FAIL=1`.

### 7b. Run it

```bash
cd /proj_sw/user_dev/$USER/tt-forge-onnx
source /proj_sw/user_dev/$USER/emu-run/env_quasar_emu.sh

# forge needs the tt-mlir submodule's tt-metal — the one your runtime links
export TT_METAL_HOME=$PWD/third_party/tt-mlir/third_party/tt-metal/src/tt-metal
export PYTHONPATH=$TT_METAL_HOME

source env/activate
cd "$TT_METAL_HOME"

TT_METAL_CACHE=/proj_sw/user_dev/$USER/emu-run/cache_emu \
TTMLIR_OP_TRACE=1 \
timeout 5400 python -u /proj_sw/user_dev/$USER/tt-forge-onnx/add_rs/probe_exec_one_op.py \
    conv2d_3x3_sp2 2>&1 | tee /tmp/emu_op.log
```

Three flags that are not optional in practice:

- **`TT_METAL_HOME` override** — `env_quasar_emu.sh` points at a standalone tt-metal,
  right for UMD/ttnn tests. forge needs the submodule one.
- **`TT_METAL_CACHE`** separate from craq-sim's, or the two fight over kernel
  artifacts.
- **`TTMLIR_OP_TRACE=1`** — one flushed line per op. At ~20 s a launch you cannot
  bisect a hang by re-running; this names the stalling op on the first attempt. It
  costs nothing when unset, and it is how the one real full-model blocker was found.
- **`timeout`** always. A hang on a simulator target waits *forever* by design — UMD
  makes the completion wait infinite rather than time out.

### 7c. Expected timeline

| phase | duration | what you see |
|---|---|---|
| queue for ZeBu | 0 → 30+ min | `Waiting for ack msg from remote...` |
| model load | 38–66 s | `Notification handler thread started` |
| first program | ~7 min | one `Writing DFB config`, then a pause |
| each later program | 20–24 s | more `Writing DFB config` |
| `conv2d_3x3_sp2` total | ~20 min | 16 ops |

```bash
grep -c "Writing DFB config" /tmp/emu_op.log   # program launches
grep    "optrace"            /tmp/emu_op.log   # which op
grep    "RESULT:"            /tmp/emu_op.log   # the answer
```

```
[conv2d_3x3_sp2] RESULT: PASS pcc=0.999996 max_abs_err=0.00383 err/scale=0.00256
```

### 7d. Clean up — always

```bash
ps -eo pid,comm,args --no-headers | awk '$2 ~ /^python/ && /probe_/ {print $1}' | xargs -r kill
sleep 4
ssh soc-l-04 "pkill -u \$USER -f 'verification/emu'; pkill -u \$USER -f zrun; pkill -u \$USER -f vovsh"
sleep 5
ssh soc-l-04 "ps -u \$USER -o pid,cmd | grep -E 'zrun|vovsh' | grep -v grep"   # must be empty
```

**[verified]** Never `pkill -f` a pattern that appears in your own command line — it
matches your own shell and kills it. This bit me twice in one session. Select on
`comm` instead, as above.

---

## 8. Queue wait vs hang — they look identical

Both show `Waiting for ack msg from remote...` forever. The remote job's own log
distinguishes them:

```bash
ssh soc-l-04 'tail -5 $(ls -t /proj_sw/user_dev/$USER/work/aether/verification/emu/emu_out/1x3/*/emu.log | head -1)'
```

| line | meaning |
|---|---|
| `Job ... waiting for 'SW HERO:JOBCOUNT_zs5/1'` | someone else has the hardware; nothing wrong |
| `Job ... waiting for 'SW HERO:LEAF_zs5_U8.M2 ...'` | same, waiting on specific modules |
| `vtool ... Refreshing handle ... for Emul:zs5` | **healthy and running** |
| `hero_adapter.tcl: child process exited abnormally` | you reconnected too soon after a kill |
| nothing, and no remote process | the job died; re-run |

That last row matters: if the backend is refreshing its handle and your side is still
stuck, the hang is **yours**, not the queue's.

---

## 9. Debug switches in the runtime

Env-gated, no-ops when unset, all work in a release build.

| variable | checks |
|---|---|
| `TTMLIR_OP_TRACE=1` | one line per op — use on every emulator run |
| `TTMLIR_LAYOUT_CHECK=1` | `to_layout` preserves logical order, **both** directions |
| `TTMLIR_RESHAPE_CHECK=1` | reshape preserves logical order |
| `TTMLIR_PERMUTE_CHECK=1` | permute against an exact CPU reference |
| `TTMLIR_CONV_CHECK=1` | conv internals: weight, activation, matmul, tail, spec |

They rest on one fact: `Tensor::to_vector<T>()` returns **logical row-major order
whatever the physical layout**, so reshape / layout / permute have exact invariants.
`TTMLIR_LAYOUT_CHECK=1` found a device-untilize corruption in a single run after a
day of narrowing by hand.

**One trap.** A check comparing an op against its own operands cannot see a corrupt
operand — both sides share the error and agree. Compare against something *outside*
the op.

---

## 10. What to expect, and what not to

**[verified 2026-09-15]** All 13 ResNet-50 ops pass on the emulator, matching
craq-sim to five or six digits:

```
relu 1.000000   reshape 0.999996   multiply 0.999991   typecast 0.999995
matmul 0.999860   linear 0.999711   permute NCHW<->NHWC 0.999996
conv2d 1x1 0.999996   conv2d 3x3 0.999996 / 0.999996   max_pool2d 0.999937
mean (dim -2) 0.999687
```

**The whole model does not run there.** Five attempts, none finished. Reasons, in the
order they bite:

1. A `to_device` deadlock at op 16 — reproducible across three builds, traced to a
   `to_layout` whose declared output is `#system_memory` being handed back as a
   device tensor. Diagnosed, not yet correctly fixed.
2. One worker core — all 53 convolutions serialise.
3. The 1.4 h default remote cap (raisable).
4. Weight volume is input-size independent: 25.6M parameters = 51.1 MB in bf16 plus
   const-eval programs. Shrinking 224 → 8 does nothing about it.

So: **emulation is for single ops and small compositions; whole models belong on
craq-sim** (full ResNet-50 in ~25 s there). If you batch emulator probes, run them in
one process so the model load is paid once — but put heavy decompositions **last**, because
one can leave the next model in the session reading memory nobody wrote, and ZeBu
DRAM is not zero-initialised, so that reads back as `inf` rather than zero.

---

## 11. Copy-paste: zero to one op

```bash
# --- once per machine -------------------------------------------------------
cd /proj_sw/user_dev/$USER
git clone --recurse-submodules git@yyz-gitlab.local.tenstorrent.com:tensix/tt-umd-simulators.git
cd tt-umd-simulators && cmake -B build -G Ninja; ninja -C build     # cmake errors: ignore
sed -i 's/^EMULATOR_TIMEOUT=5000/EMULATOR_TIMEOUT=${EMULATOR_TIMEOUT:-5000}/' \
    build/emu-quasar-1x3/quasar-1x3_run_dev.sh
ssh-copy-id soc-l-04

cd /proj_sw/user_dev/$USER
git clone --recurse-submodules https://github.com/tenstorrent/tt-forge-onnx.git
cd tt-forge-onnx
cmake -B env/build env && cmake --build env/build
source env/activate
cmake -G Ninja -B build -DCMAKE_CXX_COMPILER=clang++-17 -DCMAKE_C_COMPILER=clang-17
cmake --build build

# --- every session ---------------------------------------------------------
ssh soc-l-04 "ps -u \$USER -o pid,cmd | grep -E 'zrun|vovsh' | grep -v grep"   # must be empty

cd /proj_sw/user_dev/$USER/tt-forge-onnx
source env/activate && source ./scripts/quasar_sim_env.sh
cd "$TT_METAL_HOME" && python $OLDPWD/add_rs/probe_exec_one_op.py relu          # craq-sim first

cd /proj_sw/user_dev/$USER/tt-forge-onnx
source /proj_sw/user_dev/$USER/emu-run/env_quasar_emu.sh
export TT_METAL_HOME=$PWD/third_party/tt-mlir/third_party/tt-metal/src/tt-metal
export PYTHONPATH=$TT_METAL_HOME
source env/activate && cd "$TT_METAL_HOME"
TT_METAL_CACHE=/proj_sw/user_dev/$USER/emu-run/cache_emu TTMLIR_OP_TRACE=1 \
  timeout 5400 python -u /proj_sw/user_dev/$USER/tt-forge-onnx/add_rs/probe_exec_one_op.py \
  conv2d_3x3_sp2 2>&1 | tee /tmp/emu_op.log
grep -E "optrace|RESULT:" /tmp/emu_op.log | tail -20

ssh soc-l-04 "pkill -u \$USER -f 'verification/emu'; pkill -u \$USER -f zrun; pkill -u \$USER -f vovsh"
```

---

## 12. Five ways this goes wrong

1. **Forgot to copy the runtime into `build/install/lib`** → your change does nothing,
   silently. Compare sizes.
2. **Didn't `cd $TT_METAL_HOME`** → kernel compilation can't find relative includes.
3. **Orphaned remote job** → next run waits forever. `ps` on `soc-l-04`, not `bjobs`.
4. **Stale `NNG_SOCKET_ADDR`** after a re-reservation → waits forever. `ird list` on
   the host; only the debuda port changes.
5. **`pkill -f` matching your own command line** → kills your shell.

Companions: `ZEBU_AETHER_FIELD_GUIDE.md` (what ZeBu and aether are, and why each
layer exists), `add_rs/OP_STATUS_2026-09-10.md` (every measurement).
