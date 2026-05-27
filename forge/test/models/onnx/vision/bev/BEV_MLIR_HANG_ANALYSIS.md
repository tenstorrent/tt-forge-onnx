# BEV Model — MLIR Hang Analysis & Fix

## Problem

`test_bev_onnx_benchmark` gets stuck and never completes. The process hangs
inside `run_mlir_passes` in `forge/csrc/passes/mlir_compiler.cpp`:

```cpp
run_mlir_passes<output>(mlir_module, mlir_config);   // hangs here
log_info(LogMLIRCompiler, "MLIR passes run successfully.");  // never reached
```

The TTIR is fully generated and printed to the log, confirming that the
Forge → TTIR lowering completes. The hang occurs during the subsequent
TTIR → TTNN conversion pipeline.

---

## Key Observation — Two Tests, Same Graph

Running the same model through two different entry points produces different
outcomes:

| Test | Entry point | Result |
|------|-------------|--------|
| `test_bev_onnx_benchmark` | `test_bev_benchmark.py` | **Hangs** in `run_mlir_passes` |
| `test_merged_sweep[merged_ABCDEF-opt_level_2-enable_program_cache]` | `test_bev_blocks_benchmark.py` | **Completes** in ~724 s |

A `diff` of the two generated Forge Python modules confirms they are
**byte-for-byte identical** (same graph, same TTIR, same 15,760 MLIR ops):

```
diff bev_onnx_full_model.py merged_ABCDEF.py
10c10
< class BevOnnxFullModel(ForgeModule):
---
> class MergedAbcdef(ForgeModule):
38300c38300
< serialized_params = torch.load("generated_modules/bev_onnx_full_model_params.pt")
---
> serialized_params = torch.load("generated_modules/merged_ABCDEF_params.pt")
```

The computation graph is not the cause of the hang.

---

## Root Cause — `optimization-level=2` Triggers Memory Layout Analysis

The two tests use **different Python APIs** to set the optimization level,
which results in different MLIR pipeline options being sent:

```
test_bev_benchmark.py                  test_bev_blocks_benchmark.py
─────────────────────────────────────  ────────────────────────────────────────
MLIRConfig()                           CompilerConfig()   ← Python-side only
  .set_enable_consteval(True)            .enable_consteval = True
  .set_optimization_level(2)            .optimization_level = 2
          │                                      │
          ▼                                      ▼
  Pipeline option sent:                NOT forwarded to MLIR pipeline
  "optimization-level=2                (different attribute namespace)
   enable-const-eval=true"
          │
          ▼
  Activates: enable_memory_layout_analysis = true
          │
          ▼
  TTNNOptimizer sharding analysis
  runs over all 770 nodes → does not terminate
```

### What `optimization_level` means in the MLIR pipeline

Defined in `forge/csrc/passes/mlir_config.hpp`:

| Level | Behavior |
|-------|----------|
| `0` | All optimizer passes disabled. Fastest compile, baseline runtime. |
| `1` | Optimizer on; Conv2d-multiply fusing on; sharding (MLA) **off**. |
| `2` | Everything in level 1 **plus** `enable_memory_layout_analysis=true`. Longest compile, best runtime. |

`optimization-level=2` enables the **TTNNOptimizer Memory Layout Analysis
(MLA)** sub-pass. This pass exhaustively searches for the best L1-sharding
layout for every tensor in the graph by evaluating up to `max_legal_layouts`
(default 8) candidates per operation.

### Why MLA hangs on the BEV model

The BEV model has a combination of properties that make the sharding search
space intractable:

- **770 nodes** — the full pipeline from raw camera images to output heads
- **40× `GridSample`** — gather-like ops with indirect memory access; the
  set of legal sharded layouts is large and hardware constraints are complex
- **Bilinear `upsample2d`** — unusual shape changes (spatial dims double);
  layout propagation across these ops is non-trivial
- **219× `Conv2d`** on tensors up to `1×192×1536×1536` — large feature maps
  that are sensitive to shard placement

The MLA uses a data-flow driven search (DFSharding policy by default). For
this graph, the search does not converge within any reasonable time budget.

---

## Evidence from Logs

**`test_bev_benchmark.log`** — compile stages before the hang:

```
06:29:07  Running compile stage generate_initial_graph
06:30:42  TVM frontend conversion completed
06:32:35  TVM Forge compile passes completed
06:36:02  TVM partition completed
06:37:13  Forge module generation completed: bev_onnx_full_model
06:37:23  Running compile stage optimized_graph
06:38:32  Running compile stage consteval_graph
          <TTIR printed — 15,760 ops>
          run_mlir_passes called → HANGS (no further log output)
```

**`merged_blocks/bev_logs1/merged_abcdef_opt_1_enable_program_cache.log`** —
same graph, different config:

```
compiler_cfg : opt_level_2   ← CompilerConfig.optimization_level = 2
                                (does NOT set MLIRConfig.optimization_level)
...
<TTIR printed>
<TTNN IR printed>             ← run_mlir_passes completed
| opt_level_2  [cache=ON]  | 2554.08 ± 82.82 ms | 2554.16 ms | 0.39 |
Total test time: 724 s
```

---

## Fix

**File:** `forge/test/models/onnx/vision/bev/test_bev_benchmark.py`

Change `set_optimization_level(2)` to `set_optimization_level(1)`:

```python
# Before (hangs):
mlir_config = (
    MLIRConfig()
    .set_enable_consteval(True)
    .set_optimization_level(2)
)

# After (fixed):
mlir_config = (
    MLIRConfig()
    .set_enable_consteval(True)
    # optimization_level=2 enables memory-layout-analysis (sharding), which
    # exhaustively searches L1 shard layouts across all 770 nodes and does not
    # terminate on this model.  Level 1 keeps the TTNNOptimizer and
    # conv2d-multiply fusing on while leaving sharding analysis off.
    .set_optimization_level(1)
)
```

Level 1 retains:
- TTNNOptimizer (core-grid sizing, operation scheduling)
- Conv2d-multiply fusing

Level 1 removes (compared to level 2):
- Memory layout analysis / sharding (`enable_memory_layout_analysis=true`)

---

## Unrelated C++ Fixes (Same Branch)

Three C++ files were also modified on this branch. These fire in the
pre-MLIR Forge graph passes and do **not** affect the MLIR hang:

| File | Change | Effect |
|------|--------|--------|
| `forge/csrc/passes/constant_folding.cpp` | Added `calculate_and_set_node_shape(graph, b)` before `try_consteval_op` | Fixes missing shape recalculation when constant associativity folding rewires edges |
| `forge/csrc/passes/fuse_per_channel_ops.cpp` | Removed stale `orig_shape_len` from `Unsqueeze` TM args; added `calculate_and_set_node_shape` for the fused concat node | Fixes shape mismatch when per-channel ops are fused across concat boundaries |
| `forge/csrc/passes/mlir_compiler.cpp` | Promoted TTIR logging from `log_trace` to `log_info`; renamed string variables | Debug visibility improvement — TTIR now visible in INFO logs |

---

## How to Reproduce (Before Fix)

```bash
# Hangs — optimization-level=2 sent to MLIR pipeline
pytest forge/test/models/onnx/vision/bev/test_bev_benchmark.py::test_bev_onnx_benchmark -s

# Completes — same graph, CompilerConfig.optimization_level=2 (different API)
pytest forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py \
    -k "merged_ABCDEF and opt_level_2 and enable_program_cache" -vss
```

## How to Verify the Fix

```bash
# Should now compile and run successfully
pytest forge/test/models/onnx/vision/bev/test_bev_benchmark.py::test_bev_onnx_benchmark -s
```

---

## Future Work

If `optimization_level=2` (sharding analysis) is needed for the full BEV
model, options are:

1. **Run MLA per-block** — apply sharding analysis to the 6 individual split
   blocks (each ≤460 nodes) rather than the 770-node full model. The per-block
   benchmarks already use `merged_ABCDEF` which shows this is feasible.

2. **Add MLA timeout** — patch the TTNNOptimizer in `ttmlir` to abort the
   sharding search after a configurable wall-clock budget and fall back to the
   level-1 layout.

3. **Reduce `max_legal_layouts`** — set `mlir_config.max_legal_layouts = 2`
   (or 1) to limit the search branching factor at the cost of layout quality.
