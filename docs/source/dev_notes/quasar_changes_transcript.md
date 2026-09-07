# Explaining the Quasar Add bring-up — a call script

A speaking script for walking someone through the change set, paired with
`imgs/compiler_arch/quasar-changes.drawio.svg`. Roughly 12 minutes at a normal pace,
or 5 if you read only the **bold** lines and skip the code.

Assumes no Quasar background. Every claim here is measured; commands are copy-pasteable
and were run as written.

---

## 0 · The one-sentence version

> **We got the Add op running on Quasar. Three things were needed, and none of them was
> the op mapping — that had been correct the whole time.**

If you only say one more thing, say this:

> **The Quasar op existed, the runtime already dispatched to it, and it still did not
> run. That is why "add is unmapped" was the wrong diagnosis.**

---

## 1 · Frame it before any code  (2 min)

Score any op three ways, and only the third counts:

1. Does a Quasar implementation exist in tt-metal?
2. Does the tt-mlir runtime dispatch to it?
3. Does it actually run?

For Add, (1) and (2) were green well before this work. Show the mapping if asked — it is
six lines, in `third_party/tt-mlir/runtime/lib/ttnn/operations/eltwise/binary/binary.cpp:45`:

```cpp
#define RUN_ELTWISE_BINARY(NAME)                                          \
  runEltwiseBinaryOp(op, tensorPool, [](auto &&...args) {                 \
    return utils::isQuasar()                                              \
               ? ::ttnn::operations::experimental::quasar::binary::NAME(  \
                     std::forward<decltype(args)>(args)...)               \
               : ::ttnn::NAME(std::forward<decltype(args)>(args)...);     \
  })
```

One branch. Everything above it in the stack is shared between architectures.

**The useful context to give next:** the compiler needs no Quasar work at all, and that
is measurable rather than a claim.

```bash
cd third_party/tt-mlir
for a in wormhole_b0 quasar; do
  ./build/bin/ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=$a" \
    --dump-pass-pipeline /tmp/empty.mlir > /tmp/pipe_$a.txt 2>&1
done
diff /tmp/pipe_wormhole_b0.txt /tmp/pipe_quasar.txt
```

> **147 lines each, same passes, same order. Exactly three lines differ, and all three
> are the system descriptor or something derived from it.**

So an architecture reaches compilation *only* through the descriptor's numbers. That sets
up change 1.

*(Aside if someone tries this: `ttmlir-opt --help` segfaults on our build. Enumerate
passes with `--dump-pass-pipeline`.)*

---

## 2 · Change 1 — the descriptor lied about Quasar's data formats  (3 min)

**Row 1 of the diagram.** This is the only real code fix of the three.

> **Quasar was advertising Wormhole's thirteen data formats, including block-float
> bf8_b and bf4_b, which no Quasar device can run.**

The reason it went unnoticed is worth dwelling on:

> **Both descriptor paths — the mock one used for device-free compiles, and the live one
> read off a real device — carried the same hardcoded list. So they agreed with each
> other, and any consistency check between them passed.**

The live builder admitted it in a comment, at
`third_party/tt-mlir/runtime/lib/common/system_desc.cpp:178`:

```cpp
// The following is temporary place-holder value to be replaced by API value.
std::vector<::tt::target::DataType> supportedDataTypesVector = { /* 13 entries */ };
```

**The fix: stop restating it and ask tt-metal.** The API already existed —
`tt::is_data_format_supported(format, arch)`, public header
`tt-metalium/tt_backend_api_types.hpp:72`, dispatching to `is_supported_quasar`. Quasar
excludes Bfp2/Bfp4/Bfp8 and their `_b` variants — its narrow formats are MX,
microscaling — and has no unsigned 16- or 32-bit device format; its 32-bit formats are
Float32 and Int32.

Two edits, one commit (tt-mlir `849b674d64`):

* **Live builder** — enumerate all thirteen `target::DataType` values, map each to its
  `tt::DataFormat`, keep the ones that arch admits. No hardcoded list survives.
* **Mock builder** (`lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp:85`) — carry the matching
  five, so a device-free compile agrees with the device.

Show the diff if they want it — it reads as a deletion:

```diff
       DataTypeAttr::get(context, DataType::BFloat16),
-      DataTypeAttr::get(context, DataType::BFP_Float8),
-      DataTypeAttr::get(context, DataType::BFP_BFloat8),
-      DataTypeAttr::get(context, DataType::BFP_Float4),
-      DataTypeAttr::get(context, DataType::BFP_BFloat4),
-      DataTypeAttr::get(context, DataType::BFP_Float2),
-      DataTypeAttr::get(context, DataType::BFP_BFloat2),
-      DataTypeAttr::get(context, DataType::UInt32),
-      DataTypeAttr::get(context, DataType::UInt16),
       DataTypeAttr::get(context, DataType::UInt8),
```

**Verified four ways** — Quasar mock, Quasar live on craq-sim, Wormhole mock, Wormhole
live on silicon. Quasar reports 5, Wormhole still reports 13, and mock now equals live
for both.

**The bonus finding, if there is time.** The same commit fixes `num_cbs`, and this one
was wrong for *Wormhole*:

> **It was passing `NUM_CIRCULAR_BUFFERS`, a compile-time constant used for array
> sizing. That is 32 only under the device-side `ARCH_WORMHOLE` define and 64 for any
> host build — so a live Wormhole descriptor claimed 64 circular buffers when the real
> limit is 32. Quasar was correct by accident.**

`circular_buffer_constants.h` says outright not to use it for this and names the
replacement, so we use it: `hal::get_arch_num_circular_buffers()`. A live Wormhole
descriptor now reports 32.

---

## 3 · Change 2 — forge emitted f32, which takes a different kernel  (3 min)

**Row 2.** Not a code change: a way the compiler has to be driven.

> **An ONNX graph is float32, so forge emitted f32 tensors, and the run hung forever.**

The telemetry, which is worth reading out because it is unusually legible:

```
pending_tensix = 4
srcA = {valid=0x1 unpack=1 matrix=0}
srcB = {valid=0x1 unpack=1 matrix=0}
```

> **Both operands were delivered into the unpack bank, and the math unit never took
> them. Four Tensix pipes, stalled symmetrically.**

Note the watchdog stays silent, correctly: the cores keep retiring instructions in a
poll loop, so it is a *livelock*, not a deadlock, and `TTSIM_HANG_WATCHDOG_CLOCKS`
explicitly excludes that.

**The fix is one line of config:**

```python
cfg = CompilerConfig()
cfg.default_df_override = forge._C.DataFormat.Float16_b
compiled = forge.compile(onnx_model, inputs, compiler_cfg=cfg)
```

**The part people get wrong — say this explicitly:**

> **bf16 is not simply a narrower f32. `is_binary_sfpu_op` is true for *any* f32 op
> including add, so f32 routes Quasar's SFPU kernel while bf16 routes the FPU
> `binary_ng` kernel. They are different compute paths, so this is not a precision
> trade — it is a different piece of hardware doing the arithmetic.**

Confirm it landed in the IR rather than trusting the config:

```
"ttnn.add"(%0, %1) : (tensor<2x32x32xbf16, #ttnn_layout1>, ...)
```

f32 still livelocks. Those tests are kept, marked `f32`, and deselectable with
`-k "not f32"` — deliberately *not* `xfail`ed, because an xfail on a run that never
returns stalls the suite instead of reporting.

---

## 4 · Change 3 — kernel includes resolve against the cwd  (2 min)

**Row 3.** With bf16 the hang was replaced by something much better: a fast, explicit
error.

```
TT_THROW: Compiler include directory
  'ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels/compute'
  not found relative to current working directory '/…/tt-forge-onnx'
```

> **Quasar's `binary_ng` factory passes kernel include paths relative to the tt-metal
> root, and tt-metal resolves them against the process working directory — not
> `TT_METAL_HOME`.**

`tt_metal/impl/kernels/kernel.cpp:100-112`:

```cpp
fs::path resolve_compiler_include_dir(const fs::path& given) {
  if (given.is_absolute()) { return given; }
  auto resolved = fs::current_path() / given;      // <-- cwd, not TT_METAL_HOME
  if (!fs::is_directory(resolved)) { TT_THROW(...); }
  return resolved;
}
```

**Important framing — do not call this a bug:**

> **It is identical upstream. It is a tt-metal convention: their own test suite always
> runs from their repo root, so it never bites them. forge is a different repo, so it
> does.**

The fix is an autouse fixture in `forge/test/mlir/test_quasar_sim.py`, so the tests stay
runnable from the forge root:

```python
@pytest.fixture(autouse=True)
def _tt_metal_cwd():
    prev = os.getcwd()
    os.chdir(os.environ["TT_METAL_HOME"])
    try:
        yield
    finally:
        os.chdir(prev)
```

Outside pytest, `cd "$TT_METAL_HOME"` first.

---

## 5 · Why it took so long — the honest part  (1 min)

Worth saying, because it is the transferable lesson:

> **The two problems masked each other. The cwd problem is invisible in f32, because
> that path gets far enough to livelock. The livelock is invisible in bf16, because the
> run dies at kernel build before executing. Fixing either alone changes nothing
> observable.**

And a correction we made to our own notes:

> **`quasar.md` listed "cwd not `$TT_METAL_HOME`" as a ruled-out hypothesis. That was
> correct for the f32 livelock, which it genuinely does not explain — but it read as
> "cwd does not matter", and it is a hard blocker on the bf16 path. The note now says
> which.**

---

## 6 · The result, live if you can  (1 min)

```bash
cd /proj_sw/user_dev/ctr-lelanchelian/tt-forge-onnx
source env/activate
source ./scripts/quasar_sim_env.sh          # source it — piping runs it in a subshell
python -m pytest forge/test/mlir/test_quasar_sim.py -k bf16 -v
```

```
test_add_bf16 PASSED   test_mul_bf16 PASSED
test_sub_bf16 PASSED   test_div_bf16 PASSED
4 passed, 6 deselected in 11.03s
```

Add measures **PCC 0.999985**, about 1.4 s of execution. And the device-free coverage,
which is what a PR can actually gate on since there is no Quasar runner:

```bash
python -m pytest forge/test/mlir/test_add_op.py -q     # 16 passed in ~14 s
```

Five of those sixteen are Quasar: lowering to `ttnn.add`, the f32 default, the bf16
override reaching the IR, the descriptor contents, and Bfp8_b being rejected at config
time.

> **No tt-metal pin bump was needed. This is all on the existing pin.**

---

## 7 · Questions you should expect

**"Why not just bump the tt-metal pin?"**
We nearly concluded that. Forge's f32 configuration *does* run on a newer tt-metal, which
made the 442-commit pin gap look causal. But bf16 runs fine on our own pin, so the pin
gap is a real difference and not the blocker. Bumping alone would have converted a hang
into a silently wrong answer.

**"Is f32 broken, then?"**
Unresolved, and we are careful about this. Our minimal f32 test on a newer tt-metal
completed and gave PCC 0.695. Metal's own gap doc claims fp32 add works at PCC 1.0, and
their own test ran past thirty minutes at full CPU without returning a verdict, so we
cannot yet say whether their doc is stale or our harness was off. It is documented as
open, not filed as a bug.

**"How much of this is ours vs Metal's?"**
The whole `experimental/quasar/` op library and the Quasar HAL are upstream tt-metal. Our
tt-metal branch changes two files. What is ours is the *selection* — the `isQuasar()`
dispatch in tt-mlir's runtime — plus the descriptor fix above.

**"What is next?"**
Op dispatch is the bulk and it is mechanical: nine of a hundred and twenty-one runtime
op files carry a Quasar branch today, and eighteen more Quasar op families already exist
in tt-metal waiting to be wired. Genuine Metal asks are conv2d and the float compares.

---

## Companion material

| | |
|---|---|
| This script's diagram | `imgs/compiler_arch/quasar-changes.drawio.svg` |
| The full single-op path, pass by pass | `imgs/compiler_arch/forge-onnx_overview.drawio.svg` |
| How an op is mapped onto Quasar | `imgs/compiler_arch/quasar-add-mapping.drawio.svg` |
| The bring-up phases and what gates what | `imgs/compiler_arch/quasar-bringup-phases.drawio.svg` |
| Background and op status | [Quasar](quasar.md) |
| Step-by-step run instructions | [Running a single op on Quasar](quasar_run_single_op.md) |
| Interactive walkthrough of the compile | `python scripts/add_op_walkthrough.py --arch quasar` |

Every diagram is generated and re-derived from the real pipeline — `scripts/gen_*_diagram.py`
— and each checks that every source anchor it quotes still resolves, so a moved pass
fails the generator rather than leaving a stale picture on a slide.
