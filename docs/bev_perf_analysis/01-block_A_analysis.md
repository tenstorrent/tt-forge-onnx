# Block A (Deformed Backbone) — Deep Dive Performance Analysis

**Role:** Camera feature extraction across 4 camera streams (deformed-cylinder projection)
**Measured time:** 326.45 ± 0.38 ms (3.05 FPS)
**Share of full model:** ~67.5% — the primary bottleneck
**Target:** 33 ms / 30 FPS (requires ~10x speedup)

Config (frozen): trace=True, opt_level=2, HiFi3, fp32_dest_acc=True, Float16_b, consteval=True

---

## Section 1: Op Graph Summary (from 01-ttir-passes.mlir)

### Op Type Counts

| Op Type | Count | Notes |
|---|---|---|
| `ttir.conv2d` | 148 | Dominant compute; backbone + neck + deformable offsets |
| `ttir.conv2d_weight` | 142 | Weight preparation nodes (paired with conv2d) |
| `ttir.relu6` | 116 | Activation (fused into many conv2d outputs by MLA) |
| `ttir.slice_static` | 56 | Channel/spatial slices |
| `ttir.concat` | 48 | Skip connections, FPN merges |
| `ttir.reshape` | 40 | Spatial flattening for conv2d NHWC-flattened layout |
| `ttir.permute` | 32 | NCHW↔NHWC conversions for conv2d convention |
| `ttir.max_pool2d` | 20 | Downsampling at each pyramid level |
| `ttir.multiply` | 16 | Scale factors in FPN decoder path |
| `ttir.conv_transpose2d` | 16 | FPN upsampling decoder (4 cameras × 4 levels = 16) |
| `ttir.add` | 16 | Residual / skip-connection adds |

Total compute ops: ~509. No attention/matmul ops are present — this is a pure convolutional backbone.

### Key Tensor Shapes (Critical Path)

The network processes the initial image at extremely large spatial resolutions:

| Stage | Spatial Shape (TTIR flattened) | Channels | HxW |
|---|---|---|---|
| Input preprocessing | `1x1x2359296x3` | 3 | 1536×1536 |
| After stride-1 color conv | `1x1x2359296x3` | 3 | 1536×1536 |
| After stride-2 downsampling | `1x1x589824x2` | 2 | 768×768 |
| After pixel-unshuffle 384×384 | `1x1x147456x24` | 24 | 384×384 |
| Stage 1 entry | `1x1x147456x64` | 64 | 384×384 |
| Stage 2 entry | `1x1x36864x64` | 64 | 192×192 |
| Stage 3 entry (many ops here) | `1x1x9216x96..192` | 96–192 | 96×96 |
| Stage 4 entry | `1x1x2304x128..192` | 128–192 | 48×48 |
| Stage 5 entry | `1x1x576x192..256` | 192–256 | 24×24 |
| Stage 6 entry | `1x1x144x384..448` | 384–448 | 12×12 |
| Stage 7 (deepest) | `1x1x36x448..640` | 448–640 | 6×6 |
| FPN decoder upsample output | `1x1x9216x192` | 192 | 96×96 |

**Notable shapes:** The first two convolutions run at 1536×1536 (2.36M spatial elements) in DRAM with only 3 channels. This is a massive tensor. The heavy compute cluster runs at 96×96 (stage 3), which accounts for many repeated ops at those dimensions.

### Critical Path

The critical path is a linear encoder → FPN decoder structure, executed sequentially 4 times (once per camera). Within each camera branch:

1. `permute + reshape` (1536×1536 input normalization)
2. Two preprocessing convs at 1536×1536 (1×1 and stride-2), running fully in DRAM
3. Pixel-unshuffle chain (reshape/permute/reshape) to produce 384×384×24
4. Encoder: 8 stages with progressive downsampling (384→192→96→48→24→12→6×6)
5. FPN decoder: 4 `conv_transpose2d` + grouped depthwise convs upsampling back to 96×96
6. Final 1×1 projection to `1x1x9216x192`

The block is executed separately per camera, meaning the 148 conv2ds are effectively run 4 times = ~592 total convolutions at model level (confirmed by the doubled/quadrupled weight parameters `%arg86`, `%arg87`, etc. used multiple times).

### Redundant Patterns

The FPN decoder side has 4 grouped-depthwise convolutions (`groups=6`, kernel 3×3) at 12×12, 24×24, 48×48, 96×96 spatial sizes. These produce only `1x1x144x192`, `1x1x576x192`, `1x1x2304x192`, `1x1x9216x192` outputs — small tensors where the overhead of sharding changes dominates.

---

## Section 2: MLA Sharding Assignments (from 09-mla.mlir)

### Summary of Conv2d Output Memory Assignments

| Memory Layout | Count | Notes |
|---|---|---|
| L1 height_sharded | 84 | Majority — MLA successfully placed most mid-stage convs in L1 |
| L1 block_sharded | 40 | FPN decoder path (small spatial: 6×6, 12×12, 24×24, 48×48) |
| DRAM interleaved | 12 | Early stages (1536×1536, 384×384 large-tensor ops) + some 96×96 |
| Other L1 | 12 | Interleaved L1 (transitional) |

### Grid Size Distribution for Conv2d Outputs

| Grid | Cores | Count | Stage/Use |
|---|---|---|---|
| `58x1` | 58 | 28× | 96×96 spatial (9216 rows, non-power-of-2 partition) |
| `64x1` | 64 | 20× | Larger stages (384×384, 192×192) — fully utilizing all 64 cores |
| `36x1` | 36 | 20× | 48×48 stage (2304 rows) |
| `18x1` | 18 | 16× | 24×24 stage (576 rows) |
| `1x1` | 1 | 8× | Early 1536×1536 (single-core DRAM-sliced) |
| `5x5` | 25 | 8× | Small spatial in FPN decoder |
| `5x6` | 30 | 8× | Small FPN |
| `2x6` | 12 | 8× | Deepest stage (6×6 spatial = 36 rows) |
| `6x8` | 48 | 4× | 48×48 transitional |
| `8x8` | 64 | 4× | Fully utilized |
| `6x6` | 36 | 4× | FPN decoder 24×24 |
| `8x6` | 48 | 4× | FPN decoder 48×48 |

**Key finding:** The dominant 96×96 convolutions (9,216 spatial elements) are sharded 58x1 — using only 58 of 64 cores. This is a ceiling-division artifact: `9216 / 32-tile-height = 288 tiles`, divided by `ceil(288/5)=5 tiles/core`, yields 58 cores when the remainder doesn't fill the 59th core. This means 6 cores are idle for the most-run conv stage.

**More critical:** The initial preprocessing (1536×1536 inputs) uses grid `1x1` (single-core DRAM-sliced = `l1_full`), indicating these 2.36M-element convolutions run completely sequential on a single core pipeline with DRAM streaming. These will be very slow. For the 1536×1536 stride-2 downsampler, `conv2d_slice_config = dram_width` is used — a different tiling strategy that still reads from DRAM.

### Specific DRAM Ops at 1536×1536

Two early convolutions run entirely in DRAM without sharding:
1. `conv2d` at 1536×1536, in_ch=3, out_ch=3, kernel=1×1 → output DRAM `1x1x2359296x3` (`l1_full` single-shard, `config_tensors_in_dram=true`). This is the color normalization step.
2. `conv2d` (depthwise) at 1536×1536, in_ch=2, out_ch=2, kernel=2×1, stride=2 → output DRAM `1x1x589824x2` (`dram_width` slicing). This is the pixel-unshuffle stride conv.

These have no `shard_layout` annotation, meaning MLA could not fit them in L1 and defaulted to single-core DRAM streaming.

### Block_sharded Region (FPN Decoder)

The FPN decoder at small spatial sizes uses block_sharded with grids like `5x6` (30 cores), `6x6` (36 cores), `8x6` (48 cores). These are the `conv_transpose2d` → `multiply` → `add` → grouped-DW-conv chains. The transition from height_sharded (encoder) to block_sharded (decoder) requires layout conversion ops at every FPN merge point.

---

## Section 3: Memory Traffic Analysis

### MEMDUMP Pipeline Summary

| Checkpoint | L1→DRAM Spills | DRAM→L1 Shards | DRAM→L1→DRAM Pairs |
|---|---|---|---|
| after-L1Spill | 180 | 72 | 32 |
| after-Canon1 | 180 | 72 | 32 |
| after-Canon2 | 160 | 72 | 32 |
| before-DecomposeLayouts | 160 | 72 | 32 |
| after-DecomposeLayouts | 176 | 84 | 32 |

**Canon2 cleanup:** Canonicalization pass 2 removes 20 spills (180→160), likely eliminating redundant conversions introduced during L1 spill.

**Layout decomposition adds cost:** After DecomposeLayouts, spills increase from 160 to 176 (+16) and shards from 72 to 84 (+12). These are format adjustments needed for conv2d input requirements that MLA did not see explicitly.

### to_memory_config Count Analysis (from 12-layout-decompose.mlir)

- **Total `to_memory_config` calls: 300**
- L1→DRAM spills: 176
- DRAM→L1 shards: 124
- L1→DRAM→L1 round-trip triplets: 12 (the output was pushed to DRAM only to be re-fetched as L1-sharded input for the next op)
- DRAM→L1→DRAM round-trip triplets: 44

The DRAM→L1→DRAM pattern (44 occurrences) means: data is in DRAM interleaved, loaded to L1-sharded for a conv2d computation, then immediately written back to DRAM because the consumer (e.g., concat, add, or next conv with DRAM input) requires DRAM. These are unavoidable when ops at different shard types are chained.

### Conv2d L1 Output Immediate Spills

**200 conv2d outputs that produce L1-sharded tensors are immediately followed by a `to_memory_config dram`** (a read-back spill). The DRAM output is what actually gets consumed by the next op — the L1 assignment was done by MLA but the next consumer requires DRAM input, forcing layout decompose to insert the spill.

This is the primary inefficiency: 200/300 = 67% of all `to_memory_config` calls are spilling freshly-computed L1 results back to DRAM.

### Root Cause of Round-trips

The pattern is:
```
conv2d → L1 height_sharded output
to_memory_config (L1 → DRAM)    ← spill because next conv wants DRAM input
[next conv reads from DRAM]      ← no L1 reuse at all
```

This recurs because TTNN's `config_tensors_in_dram=true` is set globally on all convs. When convs require their activation inputs to come from DRAM (the `config_tensors_in_dram` flag), MLA assigns DRAM for the conv input but then assigns L1 for the output — creating a mandatory L1→DRAM transition every time.

---

## Section 4: Root Causes of 326ms Latency

### Cause 1: 1536×1536 Preprocessing — Single-Core DRAM Streaming (~15-25ms estimated)

The initial two convolutions on `1x1x2359296x3` tensors at 1536×1536 use `conv2d_slice_config<l1_full, 0>` with grid `1x1` and `config_tensors_in_dram=true`. A 2.36M-element BF16 tensor = 4.72MB. DRAM bandwidth on Wormhole is ~288 GB/s aggregated, but at `1x1` core utilization, effective bandwidth is limited to a single core's DRAM fetch rate. These ops cannot be parallelized over the 8×8=64 core grid.

### Cause 2: 58-Core Height Sharding for 96×96 Ops (~100-150ms estimated for 9216-spatial convs)

28 convolutions at 9,216 spatial points use grid `58x1`, leaving 6 cores idle. More importantly, each is height-sharded in a single column. These are large convolutions (e.g., 96×96, in_ch=96/128/160, out_ch=96/128/192) that constitute the heart of the encoder. 28 convolutions × ~4-5ms each = significant fraction of runtime.

### Cause 3: 300 to_memory_config Calls — Pervasive L1↔DRAM Traffic

Every `to_memory_config` call moves a full tensor between L1 and DRAM. With 176 L1→DRAM spills and 124 DRAM→L1 shards, each call transferring tensors ranging from 36KB (6×6×192×2B) to 9.4MB (96×96×192×2B), the total data movement is enormous. At 32 L1↔DRAM round-trip pairs (44 DRAM→L1→DRAM + 12 L1→DRAM→L1), this constitutes dead-overhead moves.

### Cause 4: Sequential Camera Processing (4× Latency Multiplier)

Block A processes 4 cameras. The IR shows duplicate parameter usage (`%arg88`, `%arg91`, `%arg94` used 4× each for FPN conv_transpose2d), confirming 4 identical subgraph executions in sequence. With trace=True, these are captured inside a single trace but they still run sequentially — there is no inter-camera parallelism.

### Cause 5: FPN Decoder Shard-Type Mismatch at Skip Connections

The `ttir.add` ops (16 total) merge FPN decoder outputs with encoder skip connections. In the IR, the pattern is:
```
encoder output → DRAM interleaved (spilled from L1 height_sharded)
decoder output → L1 block_sharded
add → L1 block_sharded
```
One operand of each add comes from DRAM (the encoder skip), forcing a DRAM→L1 shard fetch before the add. These are the 32 DRAM→L1→DRAM pairs reported by MEMDUMP.

---

## Section 5: Optimization Opportunities

### Opt A: Enable L1 Height-Sharding for 1536×1536 Preprocessing Convs

**Problem:** The `1x1x2359296x3` → `1x1x2359296x3` (color norm) and `1x1x2359296x2` → `1x1x589824x2` (stride-2) convolutions run at grid `1x1` in DRAM-sliced mode.

**Fix:** Force `shard_layout=height_sharded` with `act_block_h_override` for these convs. At 384×384 sub-tiles (64 rows/core), the 1536×1536 input can be split into 64 height shards across all 64 cores, reducing compute latency by ~64×. The 2.36M input tensor (4.72MB BF16) would need to be streamed per-shard from DRAM, but 64 parallel cores make this effective.

**Estimated impact:** 5–15ms reduction per camera, 20–60ms total.

### Opt B: Eliminate `config_tensors_in_dram=true` for Mid-Spatial Convolutions

**Problem:** `config_tensors_in_dram=true` is set on all 148 conv2d ops. This causes the compiler to require DRAM-interleaved inputs to each conv, forcing the 200 immediate L1→DRAM spills.

**Fix:** Disable `config_tensors_in_dram` for convolutions in the 96×96, 48×48, and 24×24 ranges. At these sizes the activation input (9216×96 = 1.77MB, 2304×192 = 884KB, 576×192 = 221KB) fits in L1 when distributed. Setting `config_tensors_in_dram=false` would allow a conv to accept an L1-sharded input directly from the previous conv's L1 output, eliminating the L1→DRAM→L1 round-trip.

**Estimated impact:** Potentially eliminating ~150 of the 176 L1→DRAM spills. Each spill avoided saves ~0.5–2ms. Total estimate: 30–80ms reduction.

### Opt C: Align 96×96 Grid to 64 Cores (58x1 → 64x1)

**Problem:** 9,216 spatial rows are distributed over 58 cores (non-power-of-2 alignment), wasting 6 cores (9.4%).

**Fix:** Pad the 9,216-row height dimension to 9,248 (64 × 32 × 4.53 → next multiple that fits 64 cores cleanly) or use `act_block_h_override=144` (9216/64=144 rows/core with 0 padding). The MLA scheduler may need a hint to prefer `64x1` over `58x1`.

**Estimated impact:** ~9% throughput improvement on all 96×96 convolutions. Since these are 28/148 = 19% of all convolutions, and at the 96×96 stage which is a major time sink, estimated 5–15ms.

### Opt D: Fuse FPN Decoder Add with Conv_transpose2d Output

**Problem:** Each FPN skip-connection `add` has one DRAM input (encoder skip) and one L1-block_sharded input (decoder output). This requires a DRAM→L1 shard fetch for the encoder skip before the add.

**Fix:** Force encoder skip-connection outputs to remain in L1-sharded form (matching the FPN decoder's block_sharded layout) rather than spilling to DRAM. This requires either (a) keeping the encoder output in L1 longer until the FPN add is reached, or (b) using the same block_sharded layout in the encoder at the skip-connection levels.

**Estimated impact:** 32 pairs of DRAM→L1 loads eliminated. Estimated 5–15ms.

### Opt E: Inter-Camera Parallelism (Structural Change — High Impact)

**Problem:** 4 camera branches run sequentially. Each branch is identical except for input data.

**Fix:** If the 4 camera branches were expressed as a batch of 4 (changing the batch dimension from 1 to 4), the entire block would run in ~1/4 the time. However, this is a model-level change that may conflict with the trace capture structure and the way BEV computes deformed offsets per camera. This would require re-expressing block A as a batched convolution.

**Estimated impact:** ~3–4× speedup (240ms saved), but requires model restructuring outside the compiler scope.

### Opt F: Act_block_h_override Tuning for Small Spatial Stages

**Problem:** Convs at 6×6 (36 elements) and 12×12 (144 elements) are assigned `2x6` or `5x5` grids. With only 36–144 spatial rows, the entire activation can fit in a single core's CBs, making multi-core sharding counterproductive (shard overhead exceeds compute).

**Fix:** For spatial ≤ 144 rows, force `shard_layout = block_sharded` with a small grid (4×4 or fewer), or use single-core with `act_block_h_override` sized to the full activation. Currently the `conv_transpose2d` at 6×6 uses `block_sharded` with `5x6=30 cores`, which gives only `36/30 ≈ 1.2` rows/core — essentially no spatial parallelism.

**Estimated impact:** 5–10ms on FPN decoder path.
