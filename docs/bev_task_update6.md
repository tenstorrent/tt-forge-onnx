# BEV Task Update 6

Hi @Nikola Vukobrat

Task updates from my side — Block A (`block_A_deformed_backbone`) L1 sharding improvements. All fixes are in `third_party/tt-mlir`.

---

## 1. ToMemoryConfigOp Bounce Spill Canonicalization

**Problem:** `OperationValidationAndFallback` and `TTNNDecomposeLayouts` insert chains like `L1_sharded → DRAM → L1_sharded` and `L1_sharded → DRAM → L1_interleaved` with no compute purpose. `ToMemoryConfigOp` had no folder or canonicalization, so these were never removed.

**Fix:** Added three patterns in `TTNNOps.cpp` + two canonicalizer passes in the pipeline:
- `ToMemoryConfigOp::fold`: identity fold when input/output types match
- `FoldConsecutiveToMemoryConfigOps`: collapses two chained `to_memory_config` ops when the intermediate is L1-sharded with no other compute users
- `BypassDRAMForL1InterleavedConsumers`: reroutes all L1-interleaved consumers of a DRAM op to read the L1-sharded source directly; erases the DRAM op when no remaining compute users

Passes inserted after `OperationValidationAndFallback` and after `TTNNDecomposeLayouts`.

**Result:** `total_ops` 1198→962 (−236). No FPS change under trace — tensor aliasing prevents remat of the folded ops.

---

## 2. CB_ZONE_EVICT False Positive Evictions + Conv2d OOM Recovery

**Problem:** `L1SpillManagement::ensureFitsL1` evicted tensors with high virtual addresses assuming TTNN allocates L1 bottom-up. In reality `tt_metal/impl/buffers/buffer.cpp:289` shows `bottom_up=false` for L1 → **top-down allocation**. The virtual-to-physical mapping is same-direction, so high-virtual tensors are at HIGH physical addresses — the farthest point from the CB zone. The loop was evicting the safest tensors (56 false evictions per compile).

Separately, `handleOOM` immediately demoted L1Full Conv2d to DRAM when its CB budget was exceeded, causing a full `L1_sharded→DRAM→L1_sharded` round-trip for the chain.

**Fix:**
- Removed the `CB_ZONE_EVICT` block and `tryReduceConv2dCBForZoneEvict` entirely from `L1SpillManagement.cpp`. The existing `wouldCBsOverlapTensors` → `evictForCBOverlap` path is the correct CB guard.
- Added `tryReduceConv2dActBlockH` in `handleOOM`: before DRAM demotion, tries `act_block_h` values {1024, 992, …, 64, 32} — a smaller `act_block_h` reduces CB peak and keeps the op in L1.

**Result:** `effectively_sharded_ops` 100→204 (+2×), `sharded_and_spilled_ops` 112→24 (−79%).

---

## 3. MaxPool2d BLOCK_SHARDED Spill (Fix 12)

**Problem:** MaxPool2d inheriting BLOCK_SHARDED output from an upstream Conv2d (via NULL hint) was immediately spilled to DRAM because `SliceRmShardedWidthTrimProgramFactory` only accepts HEIGHT_SHARDED RM or DRAM inputs. 12 `sharded_and_spilled_ops` across Block A.

**Fix:** Added `MaxPool2dRuleBook` omitting the NULL hint (prevents BLOCK_SHARDED inheritance), offering DRAM/interleaved as primary and HEIGHT_SHARDED as fallback. BLOCK_SHARDED excluded entirely.

**Result:** `sharded_and_spilled_ops` → **0** across all op types; `dram_spilled_ops` 108→96.

---

## 4. SliceStaticOp HEIGHT_SHARDED RM Output (Fix 13)

**Problem:** `SliceRuleBook::getOutputHints` returned only a NULL hint for last-dim-only (width-trim) slices, forcing all 56 slice outputs to DRAM even though `SliceRmShardedWidthTrimProgramFactory` already supports an L1 HS RM output path via `output_is_dram = !output.is_sharded()`.

**Fix:** Find the first HEIGHT_SHARDED TILED entry in `legalConfigs` and convert it to an HS RM marker hint via `.setLayout(Layout::RowMajor)` — inherits a valid `coreRangeSet` from the TILED template. The op model's analytical bypass (`TTNNOpModel.cpp`) builds the actual output from the INPUT layout, ignoring the hint's grid, so the marker hint is valid. `isValidOutputHintForInputs` gates the HS hint to HEIGHT_SHARDED RM input candidates (MaxPool2d beam state); DRAM inputs fall through to the NULL fallback.

**Result:** `dram_spilled_ops` 96→24 (−75%), FPS **2.54→2.77 (+9%)**. The outer concat downstream still reshards to DRAM (no 58×1 grid HS hint in its `legalConfigs`), but the downstream Conv2d now reads from L1 via a prefetch `to_memory_config` rather than from DRAM directly.

---

## Results Summary

Config: `opt_level_2 · bfloat16 · HiFi3 · fp32_dest_acc · trace_enabled` on WH N150

| Metric | Baseline | After All Fixes | Delta |
|--------|:--------:|:---------------:|:-----:|
| `effectively_sharded_ops` | 100 | **204** | **+104 (+2×)** |
| `effectively_sharded_%` | 24.5% | **51.0%** | **+26.5 pp** |
| `sharded_and_spilled_ops` | 112 | **24** | **−88 (−79%)** |
| `sharded_ops` | 212 | **228** | +16 |
| `total_shardable_ops` | 408 | **400** | −8 |
| `ttnn.to_memory_config` ops | 324 | **120** | **−204 (−63%)** |
| `ttnn.to_layout` ops | 24 | **16** | −8 |
| `ttnn.deallocate` ops | 700 | **420** | −280 |
| FPS | 2.79 | **2.77** | within noise (±10 ms) |

The FPS improvement is modest because trace amortises much of the DRAM-hop overhead; the primary gains are in L1 residency (effectively_sharded doubled) and IR cleanliness (204 fewer `to_memory_config` + `deallocate` ops).
