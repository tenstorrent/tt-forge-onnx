# BEV Block A — Fix 12 Benchmark Analysis

**Date:** 2026-05-26  
**Log:** `BEV_MODEL_LOGS/LATEST/block_A_after_fix_11.log`  
**Metrics:** `BEV_MODEL_LOGS/LATEST/block_A_deformed_backbone_AFTER_FIX_11_perf_metrics.json`  
**MLIR IR:** `BEV_MODEL_IRS/LATEST/BLOCK_A_AFTER_FIX_11/ttnn_block_A_deformed_backbone.mlir`  
**Config:** `opt_level_2`, `bfloat16`, `hifi3+fp32_acc`, `trace_enabled`, `program_cache=ON`

---

## 1. Test Result

```
[validation] PASSED
```

PCC exceeded the 0.99 threshold. The fix is correctly applied.

**Inference performance:**

| Config | Inference (H2D+run+D2H) | Total/frame | FPS |
|--------|-------------------------|-------------|-----|
| opt_level_2_bfloat16_hifi3_fp32_acc_trace_enabled [cache=ON] | 350.89 ± 14.96 ms | 371.57 ms | **2.69** |

---

## 2. Sharding Metrics — Before vs After

| Metric | Before Fix 12 | After Fix 12 | Delta |
|--------|:-------------:|:------------:|:-----:|
| `sharded_and_spilled_ops` | 12 | **0** | −12 ✓ |
| `effectively_sharded_ops` | 212 | **216** | +4 |
| `sharded_ops` | 224 | **216** | −8 |
| `dram_spilled_ops` | 108 | **96** | −12 |
| `total_ops_with_output_tensor` | 504 | **484** | −20 |
| `effectively_sharded_percentage` | 53.5% | **55.7%** | +2.2pp |
| `sharded_percentage` | 56.6% | **55.7%** | −0.9pp |
| `total_shardable_ops` | 396 | **388** | −8 |

**Key observation:** `sharded_ops` fell from 224 to 216 while `effectively_sharded_ops` rose from 212 to 216. Previously 12 ops were sharded-but-spilled (counted in `sharded_ops` but not `effectively_sharded_ops`); now those 12 ops are no longer wastefully sharded, so all 216 sharded ops are also effectively sharded — no wasted L1 writes at all.

---

## 3. Per-Op-Type Sharding Breakdown

| Op Type | Total | Sharded | Spilled | DRAM/Interleaved |
|---------|------:|--------:|--------:|----------------:|
| `ttnn.conv2d` | 148 | 140 | 0 | 8 |
| `ttnn.slice_static` | 56 | 0 | 0 | 56 |
| `ttnn.concat` | 48 | 12 | 0 | 36 |
| `ttnn.reshape` | 36 | 4 | 0 | 32 |
| `ttnn.permute` | 28 | 0 | 0 | 28 |
| `ttnn.max_pool2d` | 20 | 12 | 0 | 8 |
| `ttnn.conv_transpose2d` | 16 | 16 | 0 | 0 |
| `ttnn.multiply` | 16 | 16 | 0 | 0 |
| `ttnn.add` | 16 | 16 | 0 | 0 |
| `ttnn.to_layout` | 4 | 0 | 0 | 4 |
| **Total** | **388** | **216** | **0** | **172** |

All `sharded_and_spilled` entries are **zero across every op type**. The fix is fully effective.

---

## 4. MaxPool2d Output Layout Verification (from MLIR)

The MLIR IR confirms that **no max_pool2d op outputs BLOCK_SHARDED** — the pre-fix root cause. All 20 max_pool2d ops now output either HEIGHT_SHARDED (via the MaxPool2dRuleBook fallback path) or L1-interleaved/DRAM (via the primary path):

| MLIR Alias | Memory Type | Grid | Spatial size | Channels | Use |
|-----------|------------|------|-----------|----------|-----|
| `#ttnn_layout108` | **L1 HEIGHT_SHARDED** | 58×1 | 192×192 → 96×96 | 96 | 4 ops |
| `#ttnn_layout118` | **L1 HEIGHT_SHARDED** | 36×1 | 96×96 → 48×48 | 128 | 4 ops |
| `#ttnn_layout127` | **L1 HEIGHT_SHARDED** | 18×1 | 48×48 → 24×24 | 192 | 4 ops |
| `#ttnn_layout138` | **L1 interleaved** | 8×8 | 24×24 → 12×12 | 384 | 4 ops |
| `#ttnn_layout147` | **L1 interleaved** | 8×8 | 12×12 → 6×6 | 448 | 4 ops |

The model runs 4 camera passes (4× repetition of identical structure), each with 5 max_pool2d stages at decreasing spatial resolution.

**MaxPool2dRuleBook path taken per stage:**

| Stage | Spatial | Output | Path |
|-------|---------|--------|------|
| Pool stage 1 | 192×192 → 96×96 | HEIGHT_SHARDED, 58×1 | fallback (DRAM hint not chosen by solver) |
| Pool stage 2 | 96×96 → 48×48 | HEIGHT_SHARDED, 36×1 | fallback |
| Pool stage 3 | 48×48 → 24×24 | HEIGHT_SHARDED, 18×1 | fallback |
| Pool stage 4 | 24×24 → 12×12 | L1 interleaved | primary DRAM/interleaved |
| Pool stage 5 | 12×12 → 6×6 | L1 interleaved | primary DRAM/interleaved |

The HEIGHT_SHARDED outputs (stages 1–3) feed directly into `SliceStaticOp` which the `SliceRuleBook` (Fix 11) now accepts as HEIGHT_SHARDED ROW_MAJOR input via `SliceRmShardedWidthTrimProgramFactory`. The L1-interleaved outputs (stages 4–5) take a different downstream path where interleaved is acceptable.

**All 20 max_pool2d ops: BLOCK_SHARDED count = 0** (confirmed by grepping the MLIR).

---

## 5. DRAM Conv2d and MaxPool2d Ops — Legitimately DRAM

The 8 DRAM conv2d and 8 DRAM max_pool2d are not regressions — they correspond to boundary ops where DRAM is the correct placement:

**DRAM conv2d (8 ops, 4 per pattern × 2 types):**
- `Conv2d_0 / Conv2d_224 / Conv2d_448 / Conv2d_672` — first conv in each camera pass (input from DRAM host tensor, no benefit to L1 sharding)
- `AvgPool2d_11 / AvgPool2d_235 / AvgPool2d_459 / AvgPool2d_683` — AvgPool2d decomposed as conv2d, final output goes to concat which expects interleaved

**DRAM max_pool2d (8 ops = 2 stages × 4 cameras):**
- `MaxPool2d_129/353/577/801` — pool stage 4 (24×24 → 12×12, L1 interleaved)
- `MaxPool2d_158/382/606/830` — pool stage 5 (12×12 → 6×6, L1 interleaved)

These correspond to `#ttnn_layout138` and `#ttnn_layout147` (L1 interleaved, not DRAM interleaved — but they are counted as non-sharded by `isShardedMemoryLayout()` which returns `false` for interleaved regardless of L1 vs DRAM buffer type).

---

## 6. `to_memory_config` Op Count (DRAM Spill Infrastructure)

```
Before Fix 12: 108 to_memory_config ops (spill transfers)
After Fix 12:   96 to_memory_config ops (−12)
```

The 12 eliminated `to_memory_config` ops were the BLOCK_SHARDED→DRAM spills that came immediately after each max_pool2d. These are now gone from the IR because the max_pool2d outputs are no longer BLOCK_SHARDED — the downstream slice op (or add op) can consume the output directly without a layout conversion.

The remaining 96 `to_memory_config` ops are legitimate DRAM transfers for data that genuinely needs to cross from L1 sharded chains to DRAM-backed concatenation points or host output.

---

## 7. Fix Correctness Confirmation

Three independent signals confirm the fix is correctly applied:

1. **Metrics:** `sharded_and_spilled_ops = 0` across all op types
2. **MLIR IR:** Zero `block_sharded` occurrences in any `max_pool2d` output type — confirmed by grep
3. **Validation:** `[validation] PASSED` — PCC > 0.99 threshold met with the correct numerical output

The MaxPool2dRuleBook operates as designed:
- Skips the NULL hint (prevents BLOCK_SHARDED inheritance)
- Uses DRAM/interleaved as primary (safe for any downstream consumer)
- Uses HEIGHT_SHARDED as fallback (selected by the L1 optimizer when DRAM doesn't yield a sharded result and the spatial size is large enough to benefit from height-sharding)
- Never emits BLOCK_SHARDED (which SliceRuleBook would reject)

---

## 8. Remaining Opportunities

The 96 `dram_spilled_ops` represent the next frontier. These are ops whose output is in DRAM due to downstream layout incompatibilities (primarily concat and reshape ops that don't propagate sharding). Key areas:

| Op | Count in DRAM | Potential |
|----|:---:|---|
| `ttnn.slice_static` | 56 | All output to DRAM (SliceRuleBook forces non-sharded output) |
| `ttnn.concat` | 36 | Interleaved-input concats can't produce sharded output |
| `ttnn.reshape` | 32 | Non-view reshapes use non-sharded output hints |
| `ttnn.permute` | 28 | Shared ReshapeRuleBook: non-view non-sharded |
| `ttnn.conv2d` | 8 | Boundary/AvgPool2d decomposition — structurally DRAM |
| `ttnn.max_pool2d` | 8 | L1-interleaved at small spatial sizes |
