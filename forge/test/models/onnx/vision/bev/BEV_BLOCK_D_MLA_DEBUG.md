# Block D — MLA Hang Isolation & Sub-model Split Plan

## Problem

`test_opt_sweep` in `test_bev_blocks_benchmark.py` hangs during compilation when
`opt_level_2` is used for **Block B** and **Block D**.  Both blocks contain
`GridSample` ops.  The same config (`MLIRConfig.set_optimization_level(2)`)
enables the TTNNOptimizer Memory Layout Analysis (MLA) sharding pass, which
exhaustively searches L1 shard layouts and does not terminate on graphs that
contain `GridSample`.

Block D is the smallest affected block (20 nodes), making it the best target
for isolating the exact op type responsible.

---

## Block D Topology

**Block:** `block_D_cylinder_bev_transform` (20 nodes, 193 KB)

**Inputs:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `BLOCK_C_OUTPUT` (Clip_output_0) | `(1, 192, 80, 144)` | Camera-4 backbone feature from Block C |
| `input_lut_4` | `(1, 128, 64, 8, 2)` | BEV sampling LUT for camera 4 |

**Output:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `BLOCK_D_OUTPUT` (Conv_output_0) | `(1, 64, 128, 64)` | Camera-4 BEV feature to Block E |

**Data flow:**

```
input_lut_4 (1,128,64,8,2)          BLOCK_C_OUTPUT (1,192,80,144)
       │                                       │
  [0-7] 8× Gather                        [8] Conv (192→64, 1×1)
  → Gather_output_{0-7}                  [9] Clip
    each (1,128,64,2)                    → clip_out (1,64,80,144)
       │                                       │
  [10-17] 8× GridSample(clip_out, Gather_N_out)
            each → (1,64,128,64)
  [18] Concat → (1,512,128,64)
  [19] Conv (512→64, 1×1) → BLOCK_D_OUTPUT (1,64,128,64)
```

**Op breakdown:**

| Op | Count | Role |
|----|-------|------|
| `Conv` | 2 | Feature projection (192→64) and channel reduction (512→64) |
| `Clip` | 1 | ReLU6-style activation after first Conv |
| `Gather` | 8 | Slice `input_lut_4` along axis 3 → 8 (H,W,2) coordinate grids |
| `GridSample` | 8 | Sample feature map at grid coordinates → BEV-space features |
| `Concat` | 1 | Concatenate 8 sampled BEV features along channel dim |

---

## Why MLA Hangs on GridSample

`optimization-level=2` in the MLIR `ttir-to-ttnn-backend-pipeline` activates
`enable_memory_layout_analysis=true`.  The MLA pass uses a data-flow driven
sharding search (DFSharding policy) that evaluates up to `max_legal_layouts`
(default 8) shard candidates per operation.

`GridSample` is a gather-like op with indirect memory access:

- **Input feature:** `(1, 64, 80, 144)` — large spatial extent
- **Grid (coordinates):** `(1, 128, 64, 2)` — output spatial dims differ from input
- **Output:** `(1, 64, 128, 64)` — non-trivial shape change (80×144 → 128×64)

For this op the set of legal sharded layouts is large, hardware constraints on
the coordinate tensor are complex, and layout propagation across the spatial
reshape is non-trivial.  With 8 such ops the search does not converge.

Block B has 40× `GridSample` (same pattern, 4 cameras × 10 slices each) and
exhibits the same non-termination.

---

## 4-Way Sub-model Split

The full Block D model is split at three tensor boundaries to isolate each op
family into its own compilable sub-model:

```
┌──────────────────────────────────────────────────────────┐
│  Block D (20 nodes total)                                │
│                                                          │
│  D1: Conv + Clip ──────────────────────────────────────► clip_out
│  D2: 8× Gather ────────────────────────────────────────► gather_0..7
│  D3: 8× GridSample + Concat ───────────────────────────► concat_out
│  D4: Final Conv ───────────────────────────────────────► BLOCK_D_OUTPUT
└──────────────────────────────────────────────────────────┘
```

| Sub-model | File | Nodes | Op types | Input shapes | Output shape |
|-----------|------|-------|----------|--------------|--------------|
| **D1** `block_D_sub1_conv_clip` | 0.05 MB | 2 | Conv, Clip | `(1,192,80,144)` | `(1,64,80,144)` |
| **D2** `block_D_sub2_gather` | 0.01 MB | 8 | Gather ×8 | `(1,128,64,8,2)` | 8× `(1,128,64,2)` |
| **D3** `block_D_sub3_gridsample_concat` | 0.01 MB | 9 | GridSample ×8, Concat | clip `(1,64,80,144)` + 8× grid `(1,128,64,2)` | `(1,512,128,64)` |
| **D4** `block_D_sub4_reduce_conv` | 0.13 MB | 1 | Conv | `(1,512,128,64)` | `(1,64,128,64)` |

Sub-models are extracted from the existing
`BEV_model/split_models/block_D_cylinder_bev_transform.onnx` using
`onnx.utils.extract_model` and written to `BEV_model/block_d_debug_models/`.

### Tensor name boundaries

```python
# D1 output / D3 feature input
BLOCK_D_CONV_CLIP_OUTPUT = (
    "/model/_backbone/CameraCylinderEncoder/_camera_encoder"
    "/_transformation/_bev_transformation"
    "/_to_final_encoding_conv/_net/_net.2/Clip_output_0"
)  # shape (1, 64, 80, 144)

# D2 outputs / D3 grid inputs  (i=0 → "Gather_output_0", i>0 → "Gather_{i}_output_0")
BLOCK_D_GATHER_OUTPUTS = tuple(
    f"{_C_BEV}/Gather_{'output_0' if i == 0 else f'{i}_output_0'}"
    for i in range(8)
)  # each shape (1, 128, 64, 2)

# D3 output / D4 input
BLOCK_D_CONCAT_OUTPUT = f"{_C_BEV}/Concat_output_0"
# shape (1, 512, 128, 64)
```

---

## Files Changed

| File | Change |
|------|--------|
| `model_utils/bev_split_utils.py` | Added `BLOCK_D_CONV_CLIP_OUTPUT`, `BLOCK_D_GATHER_OUTPUTS`, `BLOCK_D_CONCAT_OUTPUT`, `BLOCK_D_SUB_DEFS`, `block_d_debug_dir()`, `block_d_subs_available()`, `split_block_d_subs()` |
| `test_bev_block_d_debug.py` | New test file — 8 parametrized cases (4 sub-models × 2 configs) |

---

## Running the Tests

Sub-models are extracted automatically by the `block_d_subs` fixture on first
run (requires `BEV_model/split_models/block_D_cylinder_bev_transform.onnx`).

**Step 1 — Sanity check: all sub-models compile with `opt_level_1` (no MLA)**

```bash
pytest forge/test/models/onnx/vision/bev/test_bev_block_d_debug.py -v -s \
    -k "opt_level_1"
```

Expected: all 4 pass.

**Step 2 — Hang isolation: run `opt_level_2` in sub-model order**

```bash
pytest forge/test/models/onnx/vision/bev/test_bev_block_d_debug.py -v -s \
    -k "opt_level_2"
```

Tests execute D1 → D2 → D3 → D4.  The first one that hangs identifies the
culprit op type.

**Step 3 — Run a single sub-model**

```bash
# Test only the GridSample+Concat sub-model
pytest forge/test/models/onnx/vision/bev/test_bev_block_d_debug.py -v -s \
    -k "sub3 and opt_level_2"

# Test only the Gather sub-model
pytest forge/test/models/onnx/vision/bev/test_bev_block_d_debug.py -v -s \
    -k "sub2 and opt_level_2"
```

---

## Expected Outcome

| Sub-model | opt_level_1 | opt_level_2 | Reason |
|-----------|-------------|-------------|--------|
| D1 Conv + Clip | Pass | Pass | Standard linear ops; small search space |
| D2 8× Gather | Pass | Pass (likely) | Index slicing; regular access pattern |
| D3 8× GridSample + Concat | Pass | **Hang** | Irregular gather-like access; large MLA search space |
| D4 Final Conv 1×1 | Pass | Pass | Trivial; one candidate layout |

If D3 hangs and D2 does not, the root cause is confirmed to be **`GridSample`**,
not `Gather`.  If D2 also hangs, both op types contribute.

---

## Fix Options

Once the culprit is confirmed:

1. **Downgrade to `opt_level_1` for blocks containing `GridSample`** — already
   done for `test_bev_benchmark.py` (see `BEV_MLIR_HANG_ANALYSIS.md`).

2. **Add MLA timeout** — patch `TTNNOptimizer` in `ttmlir` to abort the sharding
   search after a configurable wall-clock budget and fall back to the level-1
   layout.

3. **Reduce `max_legal_layouts`** — set `mlir_config.max_legal_layouts = 2` (or
   1) to limit the branching factor at the cost of layout quality.

4. **Exclude `GridSample` from MLA** — add an op-type exclusion list to the MLA
   pass so `GridSample` always uses the default non-sharded layout without search.
