# Block C (Cylinder Backbone) and Block B (Deformed BEV Transform) — Performance Analysis

Config (frozen): trace=True, opt_level=2, HiFi3, fp32_dest_acc=True, Float16_b, consteval=True

---

# Part 1: Block C (Cylinder Backbone) Analysis

**Role:** Cylinder-view camera backbone (different projection from Block A)
**Measured time:** 106.91 ± 0.21 ms (9.31 FPS)
**Share of full model:** ~22% — secondary bottleneck
**Target:** 33 ms / 30 FPS (requires ~3.2× speedup)

---

## Section 1: Op Graph Summary (from 01-ttir-passes.mlir)

### Op Type Counts

| Op Type | Count | Notes |
|---|---|---|
| `ttir.conv2d` | 39 | Smaller network than Block A (roughly 1/4 the count) |
| `ttir.conv2d_weight` | 39 | Weight preparation nodes |
| `ttir.relu6` | 32 | Activation (fused into conv2d by MLA) |
| `ttir.slice_static` | 15 | Channel/spatial slices |
| `ttir.concat` | 13 | FPN merges |
| `ttir.reshape` | 10 | Spatial flattening for NHWC-flattened layout |
| `ttir.multiply` | 8 | FPN decode scale factors |
| `ttir.permute` | 8 | NCHW↔NHWC conversions |
| `ttir.max_pool2d` | 6 | Downsampling at pyramid levels |
| `ttir.add` | 6 | Residual/skip-connection adds |
| `ttir.conv_transpose2d` | 4 | FPN decoder upsamplers |

Total: ~138 compute ops. Same architectural pattern as Block A (encoder + FPN decoder) but approximately 1/4 the scale.

### Key Tensor Shapes

Block C has a wider input (1280×2304 instead of 1536×1536), making the initial preprocessing convolutions actually larger than Block A's:

| Stage | Flattened Shape | HxW |
|---|---|---|
| Input preprocessing | `1x1x2949120x3` | 1280×2304 (2.95M elements) |
| After stride-2 downsampling | `1x1x737280x2` | 640×1152 |
| After pixel-unshuffle to 320×576 | `1x1x184320x24` | 320×576 |
| Stage 1 | `1x1x184320x64` | 320×576 |
| Stage 2 (high compute density) | `1x1x46080x96..192` | 160×288 |
| Stage 3 | `1x1x11520x128..192` | 80×144 |
| Stage 4 | `1x1x2880x192` | 40×72 |
| Stage 5 (deepest) | `1x1x720x192..640` | 20×36 |
| FPN decoder output | `1x1x11520x192` | 80×144 |

**Critical difference from Block A:** Block C's high-compute stage is at 160×288 (46,080 spatial = 5× Block A's 96×96). This means fewer operations but each is on a much larger tensor.

### Critical Path

Same structure as Block A: encoder stages → FPN decoder. One camera stream only (not 4×). The convolutions at 160×288 are the compute-intensive cluster:
- `tensor<1x1x46080x96>` with 3×3 kernel: 46,080 × 96 × 9 = 39.8M MACs per conv
- `tensor<1x1x46080x192>` with 1×1 kernel: 46,080 × 192 = 8.8M MACs per conv

Several convolutions at this stage: 2× 96-channel 3×3, 1× 1×1 96→160, 1× 160→128 3×3, expansion convs to 192 — approximately 8-10 convolutions at 160×288.

---

## Section 2: MLA Sharding Assignments (from 09-mla.mlir)

### Summary of Conv2d Output Memory Assignments

| Memory Layout | Count | Notes |
|---|---|---|
| L1 height_sharded | 20 | Mid-stage convolutions |
| L1 block_sharded | 15 | FPN decoder, small spatial |
| DRAM interleaved | 4 | Initial preprocessing (1280×2304) |

Block C has a higher fraction of L1-assigned ops (35/39 = 90%) vs Block A (136/148 = 92%), both broadly similar.

### Grid Size Distribution for Conv2d Outputs

| Grid | Cores | Count | Stage/Use |
|---|---|---|---|
| `60x1` | 60 | 8× | 160×288 stage (46,080 rows → 60 cores) |
| `63x1` | 63 | 6× | Large 160×288 stage (borderline 63/64 cores) |
| `8x6` | 48 | 6× | Block_sharded FPN decoder |
| `6x6` | 36 | 4× | FPN decoder, 40×72 stage |
| `1x1` | 1 | 3× | Initial 1280×2304 preprocessing (single-core) |
| `64x1` | 64 | 3× | Fully utilized 64-core runs |
| `45x1` | 45 | 3× | Intermediate stage (11,520 rows) |
| `8x5` | 40 | 3× | FPN decoder |
| `8x7` | 56 | 1× | FPN decoder |
| `6x7` | 42 | 1× | FPN decoder |
| `8x8` | 64 | 1× | Fully utilized |

**Key finding:** The dominant 160×288 stage convolutions use grids `60x1` (60 cores, 4 idle) and `63x1` (63 cores, 1 idle). At 46,080 spatial rows: `46080 / 32 = 1440 tiles`. Dividing 1440 by core count: `1440/60 = 24 tiles/core` (clean), `1440/63 ≈ 22.86` (fractional, hence 63 cores not 64). This is similar to Block A's 58-core issue.

**Large DRAM ops:** The initial 1280×2304 convolutions use grid `1x1`, same as Block A's 1536×1536 problem. The Block C input `1x1x2949120x3` is actually larger (2.95M vs 2.36M elements). Since Block C has only 1 camera stream, these run once (vs 4× in Block A).

### Shard Layout Summary

Block C's conv2d MLA attributes: `shard_layout = height_sharded` (stages 1-5) and `shard_layout = block_sharded` (FPN decoder). The `config_tensors_in_dram=true` is set globally, same as Block A.

---

## Section 3: Memory Traffic Analysis

### MEMDUMP Pipeline Summary

| Checkpoint | L1→DRAM Spills | DRAM→L1 Shards | DRAM→L1→DRAM Pairs |
|---|---|---|---|
| after-L1Spill | 49 | 21 | 10 |
| after-Canon1 | 49 | 21 | 10 |
| after-Canon2 | 45 | 21 | 10 |
| before-DecomposeLayouts | 45 | 21 | 10 |
| after-DecomposeLayouts | 49 | 25 | 10 |

**Canon2 removes 4 redundant spills** (49→45). DecomposeLayouts adds back 4 spills (+4) and 4 new shards (+4).

### to_memory_config Count Analysis (from 12-layout-decompose.mlir)

- **Total `to_memory_config` calls: 89**
- L1→DRAM spills: 49
- DRAM→L1 shards: 40
- L1→DRAM consecutive pairs: 25 (DRAM re-inputs after L1 output)
- DRAM→L1 consecutive pairs: 25

Block C has 89 total vs Block A's 300 — consistent with Block C being roughly 1/3 the op count. The ratio of spills to total ops is similar: Block C 49/39 conv2d = 1.26 spills/conv; Block A 176/148 = 1.19 spills/conv. The spill pattern is structurally identical — `config_tensors_in_dram=true` forcing L1→DRAM→L1 for every inter-conv boundary.

### Round-Trip Analysis

Block C has 25 consecutive L1→DRAM pairs and 25 DRAM→L1 pairs. The 10 DRAM→L1→DRAM pairs (MEMDUMP) are the FPN skip-connection adds (6 adds × ~1.67 round-trips each, with some fused).

---

## Section 4: Root Causes of 107ms Latency

### Cause 1: 1280×2304 Preprocessing — Single-Core DRAM Streaming

The initial convolutions on `1x1x2949120x3` tensors at grid `1x1` are the equivalent of Block A's 1536×1536 problem, but with an even larger tensor (2.95M vs 2.36M elements = 5.9MB BF16). Three convolutions (color norm 1×1, stride-2 downsampler 2×1) run at single-core DRAM bandwidth.

**Estimated impact:** ~5–10ms (1 camera, not 4).

### Cause 2: 60x1/63x1 Height Sharding at 160×288 Stage

The most compute-intensive stage (46,080 spatial points) is sharded as 60 or 63 cores in a single column, wasting 1–4 cores. More importantly, 8 convolutions at this size with channels ranging from 96 to 192 form the core execution budget.

Arithmetic: A 160×288, 192×192 3×3 conv at HiFi3 with fp32 accumulation: 46,080 × 192 × 192 × 9 = ~16B MACs. At Wormhole's ~4 TFLOPS (fp32 acc), this is ~4ms per conv. With 6+ such convolutions, this contributes 25–30ms.

### Cause 3: 89 to_memory_config Calls — L1↔DRAM Traffic

At 160×288×192 channels = 35.4MB per tensor, each L1→DRAM spill at this stage moves ~18MB of data (BF16 = 2B, height-sharded 60-core output = 1/60th stored on each core, full tensor in DRAM). The DRAM bandwidth cost per full tensor write: 35.4MB / 288 GB/s ≈ 0.12ms. With 49 spills, this contributes ~6ms just in data movement.

### Cause 4: FPN Decoder Shard-Type Mismatch at Skip Connections

Same pattern as Block A: 6 `add` ops with one DRAM input (encoder skip) and one L1-block_sharded input (decoder output). The 10 DRAM→L1→DRAM round-trips from MEMDUMP correspond to these adds.

---

## Section 5: Optimization Opportunities for Block C

### Opt A: Height-Shard the 1280×2304 Preprocessing Convs

Same fix as Block A. The `1x1x2949120x3` preprocessing convs should use `height_sharded` with `act_block_h_override` to split across 64 cores. At 46,080 rows/core, the full tensor is fetched in parallel from DRAM.

**Estimated impact:** 3–8ms reduction.

### Opt B: Eliminate `config_tensors_in_dram=true` for 160×288 Convolutions

The `config_tensors_in_dram=true` flag forces each conv's output to be spilled to DRAM before the next conv reads it. At 160×288×192 channels, the tensor is 35.4MB — too large for L1 directly, but with height-sharding, each core holds only 35.4MB/60 = 590KB which is within L1 capacity (1.46MB unreserved). Disabling `config_tensors_in_dram` for these would allow back-to-back L1→L1 conv chains.

**Estimated impact:** Eliminate ~30 of 49 spills. Estimated 15–25ms savings (dominates Block C budget).

### Opt C: Align 160×288 Grid to 64 Cores

At 46,080 spatial rows, the 60-core partition (46,080/60=768 rows/core) is natural but wastes 4 cores. Padding to 46,112 (next 64-aligned point) or using `act_block_h_override=720` (64×720=46,080 exactly — checks: 46080/64=720, 720/32=22.5 — not tile-aligned). Better: use `act_block_h_override=736` (64×736=47,104, pads 46,080). Alternatively, MLA could use a 45×1 or 45×1 grid that divides cleanly.

**Estimated impact:** 4/64 = 6.3% compute improvement on 160×288 convolutions → ~2–3ms.

### Opt D: FPN Skip Connection L1 Retention

Same as Block A Opt D: keep encoder skip-connection outputs in L1-sharded form to avoid the DRAM→L1 load for the skip adds. Block C has 6 such adds (vs 16 in Block A) — smaller impact but still worthwhile.

**Estimated impact:** 2–4ms.

### Opt E: Pipeline 160×288 Conv Chain (If op scheduling allows)

Currently each conv in the 160×288 chain is independent in L1 but runs sequentially. If the TTNN runtime allows, pipelining (overlapping the DRAM weight fetch for conv N+1 with the compute of conv N) could hide ~50% of weight-load latency. This requires prefetch logic in the kernel scheduler.

**Estimated impact:** 5–15ms if implemented.

---

# Part 2: Block B (Deformed BEV Transform) Analysis

**Role:** Deformable grid sampling from the camera backbone features into BEV feature space
**Status:** FAILED — PCC = 0.9880456726465873, required 0.99
**Runtime:** ~70s test duration (includes compilation), actual inference ~unknown
**Note:** Block B does not have a clean ms measurement because the test fails before reporting inference time.

---

## Section 6: Block B IR Structure (from 09-mla.mlir)

### Op Composition

| Op Type | Count | Notes |
|---|---|---|
| `ttnn.grid_sample` | 32 | Core of the block — deformable BEV sampling |
| `ttnn.conv2d` | 8 | Post-sampling convolutions |
| `ttnn.to_layout` | 81 | Very high — format conversions around grid_sample |
| `ttnn.to_memory_config` | 36 (MLA) / 104 (after layout decompose) | Explodes after layout decompose |
| `ttnn.reshape` | 76 | Coordinate/feature reshaping |
| `ttnn.concat` | 4 | Merging sampled features |
| `ttnn.permute` | 8 | NCHW↔NHWC transitions |
| `ttnn.slice_static` | 32 | Coordinate extraction |

### Grid Sample Characteristics

All 32 `ttnn.grid_sample` ops have the same signature:
- **Feature input (source):** `tensor<1x96x96x64xbf16>` — DRAM interleaved, `memref<9216x64xbf16>`
- **Coordinates (grid/LUT):** `tensor<1x128x64x2xbf16>` — DRAM interleaved, `memref<8192x2xbf16>`
- **Output:** `tensor<1x128x64x64xbf16>` — L1 height_sharded, grid `64x1`, `memref<128x64xbf16>`

The feature map `1x96x96x64` = 9,216 × 64 BF16 = 1.18MB. The coordinate LUT `1x128x64x2` = 8,192 × 2 BF16 = 32KB per sample. Output `1x128x64x64` = 8,192 × 64 BF16 = 1MB.

All 32 grid_sample ops share the same feature map source (single `%16` or `%88` from a prepare step), but each gets a different coordinate LUT (different `%arg4..%arg7` for 4 LUT sets × 8 head dimensions).

### Trace Capture Analysis (from 11-trace-hoist.mlir)

The trace in Block B is structured as:
- `@trace_0_forward` function takes: 4 camera feature inputs (`1x192x96x96`), 4 LUT inputs (`1x128x64x8x2` — note 5D tensor), and ~16 weight constants
- The grid_sample ops **ARE inside** `@trace_0_forward` — they are in the traced region

**Critical detail:** The LUT inputs have shape `1x128x64x8x2` (5D). Before `grid_sample`, they are reshaped: `1x128x64x8x2` → `1x128x64x2` (implicitly per-slice). The LUT is passed as `input_lut_0..3` with `ttcore.argument_type = input` — meaning it is a runtime-variable input to the trace, not a captured constant.

This is the trace capture issue. `TTNNTraceHoistTransform` includes the grid_sample ops in the trace region, but the coordinate LUT inputs (`input_lut_0..3`) are runtime tensors that change each frame. When trace is executed with `capture_or_execute_trace`, the LUT data must be written to pre-allocated device buffers via `ttnn.write_tensor` calls, and the trace reads from those pre-allocated addresses.

### 5D Tensor Handling

The LUT inputs enter as `tensor<1x128x64x8x2xbf16>` (5D). In the trace function signature:
```
%arg4: tensor<1x128x64x8x2xbf16, ... memref<65536x2xbf16, dram>, interleaved>
```
This maps the 5D tensor to a 2D memref `[65536, 2]`. The 5D tensor is processed via `ttnn.reshape` to `1x128x512x2` and then `ttnn.slice_static` to extract each `1x128x64x2` coordinate slice for each of the 8 grid_sample calls per LUT.

The 5D→2D memref mapping (`65536 = 128×64×8×1 = 65536`) works correctly at the DRAM level. There is no indication of a 5D fallback path here — the shapes resolve cleanly to 2D memrefs.

### to_memory_config Explosion

MLA: 36 `to_memory_config` calls
Layout decompose: 104 `to_memory_config` calls — **+68 new insertions**

This is the largest relative increase of any block (3× growth vs Block A's 1.1× and Block C's 1.1×). The 68 new insertions are almost certainly caused by the grid_sample output (L1 height_sharded `64x1`) flowing into conv2d which requires DRAM interleaved input. With 32 grid_sample outputs and 32 corresponding L1→DRAM spills, that accounts for most of the 68 new ops (32 spills + 36 reshapes/concat pre-conditions).

Block B layout-decompose also generates 72 L1→DRAM spills from the MEMDUMP (after-DecomposeLayouts: spills=72, up from 4 at after-L1Spill). This 18× increase in spills after layout decompose is the largest of any block and suggests the layout decompose pass is inserting many grid_sample→conv transitions that require L1→DRAM moves.

---

## Section 7: Block B PCC Failure Analysis

### PCC = 0.988 vs Required 0.99

The error message shows `Tensor mismatch. PCC = 0.9880456726465873, but required = 0.99`.

The output tensor mismatch (printed in the log) shows a pattern like:
```
[-0.3750, -0.3750, -0.3750, ..., 1.4609, ...]
```
Values show BF16-precision rounding artifacts (discrete jumps like -0.375, 1.4609, 0.3008), which are consistent with nearest-neighbor grid_sample quantization error, not a systematic computation error.

### Root Cause: Nearest-Neighbor Grid Sampling in Traced Execution

All 32 `ttnn.grid_sample` ops use `mode = "nearest"` and `padding_mode = "zeros"`. The `align_corners = true` attribute affects coordinate mapping. In trace mode, the LUT coordinates are captured once (at capture time) and replayed at execution time.

The issue is likely one of two things:

1. **Trace capture uses a different LUT than inference:** If the LUT inputs (`input_lut_0..3`) at trace-capture time differ from the first inference pass, the captured grid_sample lookups will produce different (incorrect) outputs. The trace buffer captures the LUT data via `write_tensor` before each execution, so this should be correct — but if `write_tensor` is called with wrong data in one pass, PCC diverges.

2. **Nearest-neighbor rounding difference between CPU reference and TT implementation:** The TT `grid_sample` with `mode=nearest, align_corners=true` may round coordinates differently than PyTorch at boundary pixels. At `align_corners=true`, coordinates in `[-1, 1]` map to `[0, H-1]`. A coordinate of exactly `0.0` maps to pixel `(H-1)/2`. At half-pixel boundaries, nearest-neighbor rounding (round-half-away-zero vs round-half-to-even) can differ.

3. **5D LUT Slicing Error:** The reshape from `1x128x64x8x2` → 8 slices of `1x128x64x2` uses `slice_static`. If the slice indices are off by one or the stride is incorrect, all 32 grid_sample calls would use wrong coordinate LUTs, producing systematic PCC degradation.

### Trace Hoisting Correctness Check

The `input_lut_0..3` tensors have `ttcore.argument_type = input` (runtime variable), not `constant`. The trace correctly lists them as inputs to `@trace_0_forward`. At each execution, `write_tensor` is called on the pre-allocated LUT buffers before `capture_or_execute_trace`. This part of the trace infrastructure appears correct.

The PCC failure is more likely a **grid_sample numerical difference** between PyTorch reference and the TTNN `ttnn.grid_sample(mode=nearest, align_corners=true)` implementation at boundary coordinates, or an off-by-one in the LUT slicing.

### Recommended Investigation Steps

1. **Disable trace** for Block B only and rerun — if PCC passes without trace, the issue is in `write_tensor` ordering or trace buffer aliasing.
2. **Replace `mode=nearest` with `mode=bilinear`** — if PCC improves, the nearest-neighbor rounding in TTNN differs from PyTorch. Note: this changes model behavior.
3. **Print the LUT slice being passed to each grid_sample** — compare `slice_static(lut, begins=[0,0,0,0,0], ends=[1,128,64,1,2])` output between CPU and device to check if slicing is correct.
4. **Compare individual grid_sample output** (before concat/conv) on a single grid_sample call — isolate whether the PCC failure comes from grid_sample or the post-sampling conv2d path.

### Block B Performance Context

Despite the PCC failure, the compilation completes in ~70s (dominated by JIT kernel compilation: 77 JIT builds at ~14ms each = ~1.1s, plus model compile). The inference time for Block B is not reported due to test failure, but based on op count (32 grid_sample + 8 conv2d) and tensor sizes, it is estimated at 15–30ms.

Block B represents a different compute pattern: scatter/gather memory access (grid_sample) rather than regular convolution. The 32 grid_sample ops each read 1.18MB of feature data and 32KB of coordinates, producing 1MB output. Total data movement: 32 × (1.18 + 0.032 + 1.0)MB = 71MB. At effective bandwidth, this is substantial.

---

## Summary Comparison: Blocks A, B, C

| Metric | Block A | Block C | Block B |
|---|---|---|---|
| Latency | 326ms | 107ms | ~70ms (est) |
| Conv2d count | 148 | 39 | 8 |
| Primary op | conv2d | conv2d | grid_sample |
| to_memory_config (final) | 300 | 89 | 104 |
| L1→DRAM spills (MEMDUMP) | 176 | 49 | 72 |
| DRAM→L1→DRAM pairs | 32 | 10 | 0 |
| L1 spill increase in layout decompose | +16 | +4 | +68 |
| Core utilization (dominant stage) | 58/64 (91%) | 60–63/64 (94–98%) | 64/64 (100%) |
| PCC status | PASS | PASS | FAIL (0.988) |

Block C is structurally healthier than Block A: better core utilization (60–63 vs 58 cores), 4× fewer ops, and scales proportionally with latency. Block B's layout decompose explosion (+68 `to_memory_config`) is its unique characteristic and deserves a targeted fix.
