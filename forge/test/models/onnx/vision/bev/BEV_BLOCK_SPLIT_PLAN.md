# BEV Model Block-Splitting Plan

## Goal

Increase FPS of the BEV model by splitting `simple_bev_prep.onnx` into logical blocks,
benchmarking each block independently, and applying targeted compiler optimizations to
the blocks that are bottlenecks.

---

## Model Overview

**File:** `BEV_model/model/simple_bev_prep.onnx`  
**Size:** 346 MB | **Nodes:** 771 | **Opset:** 18 | **Initializers:** 1139

### Inputs (10 tensors)

| Name | Shape | Purpose |
|------|-------|---------|
| `input_0` | `(1, 3, 1536, 1536)` | Camera 0 RGB image |
| `input_1` | `(1, 3, 1536, 1536)` | Camera 1 RGB image |
| `input_2` | `(1, 3, 1536, 1536)` | Camera 2 RGB image |
| `input_3` | `(1, 3, 1536, 1536)` | Camera 3 RGB image |
| `input_lut_0` | `(1, 128, 64, 8, 2)` | Spatial sampling coords for camera 0 |
| `input_lut_1` | `(1, 128, 64, 8, 2)` | Spatial sampling coords for camera 1 |
| `input_lut_2` | `(1, 128, 64, 8, 2)` | Spatial sampling coords for camera 2 |
| `input_lut_3` | `(1, 128, 64, 8, 2)` | Spatial sampling coords for camera 3 |
| `input_4` | `(1, 3, 1280, 2304)` | Camera 4 RGB image (different aspect) |
| `input_lut_4` | `(1, 128, 64, 8, 2)` | Spatial sampling coords for camera 4 |

### Outputs (3 tensors)

| Name | Shape | Purpose |
|------|-------|---------|
| `occupancy_mid_range_height_map` | `(1, 1, 256, 128)` | Occupancy prediction |
| `aux_semantic_logits_mid` | `(1, 12, 256, 128)` | 12-class semantic logits |
| `aux_visibility_logits_mid` | `(1, 1, 256, 128)` | Visibility/occlusion map |

### Op Distribution

| Op | Count | % |
|----|-------|---|
| Conv | 219 | 28.4% |
| Clip | 180 | 23.3% |
| Slice | 77 | 10.0% |
| Concat | 73 | 9.5% |
| Gather | 40 | 5.2% |
| GridSample | 40 | 5.2% |
| MaxPool | 29 | 3.8% |
| Add | 25 | 3.2% |
| Mul | 24 | 3.1% |
| Others | 64 | 8.3% |

---

## Architecture and Natural Split Points

The ONNX node names encode the module hierarchy. Six natural blocks emerge:

```
INPUT (10 tensors)
  4× images (1,3,1536,1536) + 4 LUTs (1,128,64,8,2)  ─┐  cameras 0-3
  1× image  (1,3,1280,2304) + 1 LUT  (1,128,64,8,2)  ─┘  camera 4

┌─────────────────────────────────────────────────────────────────┐
│ Block A  CameraDeformedCylinder Backbone          460 nodes     │
│   inputs:  input_0, input_1, input_2, input_3                   │
│   outputs: 4× (1, 192, 96, 96)  — feature map per camera       │
│   ONNX module: CameraDeformedCylinderEncoder/_camera_backbone   │
└─────────────────────────────────────────────────────────────────┘
                          │ 4× (1,192,96,96)
┌─────────────────────────────────────────────────────────────────┐
│ Block B  CameraDeformedCylinder BEV Transform      80 nodes     │
│   inputs:  Block A outputs + input_lut_0..3                     │
│   outputs: 4× (1, 64, 128, 64)  — each camera in BEV space     │
│   ONNX module: CameraDeformedCylinderEncoder/_transformation    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Block C  CameraCylinder Backbone                  129 nodes     │
│   inputs:  input_4                                              │
│   output:  (1, 192, 80, 144)  — feature map for camera 4       │
│   ONNX module: CameraCylinderEncoder/_camera_backbone           │
└─────────────────────────────────────────────────────────────────┘
                          │ (1,192,80,144)
┌─────────────────────────────────────────────────────────────────┐
│ Block D  CameraCylinder BEV Transform              20 nodes     │
│   inputs:  Block C output + input_lut_4                         │
│   output:  (1, 64, 128, 64)  — camera 4 in BEV space           │
│   ONNX module: CameraCylinderEncoder/_transformation            │
└─────────────────────────────────────────────────────────────────┘

              5× (1,64,128,64) from Block B + Block D
┌─────────────────────────────────────────────────────────────────┐
│ Block E  BEV Aggregator Backbone                   57 nodes     │
│   inputs:  5× (1, 64, 128, 64)  [concat then process]          │
│   output:  (1, 64, 128, 64)  — vision_bev_encoder_output       │
│   ONNX module: CameraBevMidRangeEncoder/_camera_bev_aggregator  │
└─────────────────────────────────────────────────────────────────┘
                          │ (1,64,128,64)
┌─────────────────────────────────────────────────────────────────┐
│ Block F  Output Heads                              24 nodes     │
│   input:   (1, 64, 128, 64)                                     │
│   outputs: occupancy (1,1,256,128)                              │
│            semantic  (1,12,256,128)                             │
│            visibility(1,1,256,128)                              │
│   ONNX module: occupancy_mid_range/_head                        │
└─────────────────────────────────────────────────────────────────┘
```

> **Note on interleaving:** Blocks A–D have overlapping node indices in the raw ONNX graph
> because the compiler visits both camera streams interleaved. This does not affect splitting:
> `onnx.utils.extract_model` operates on tensor names, not node indices.

---

## Boundary Tensors

These are the intermediate tensor names used as split points for `onnx.utils.extract_model`.

### Block A → Block B (4 tensors, each `(1, 192, 96, 96)`)

```
/model/_backbone/CameraDeformedCylinderEncoder/_camera_encoder/_camera_backbone/_encoder/
  _trifocal_backbone/_neck/_top_down.3/_conv_block/_convblock_3/_net/_net.2/Clip_output_0
  _trifocal_backbone/_neck/_top_down.3/_conv_block/_convblock_3/_net/_net.2_1/Clip_output_0
  _trifocal_backbone/_neck/_top_down.3/_conv_block/_convblock_3/_net/_net.2_2/Clip_output_0
  _trifocal_backbone/_neck/_top_down.3/_conv_block/_convblock_3/_net/_net.2_3/Clip_output_0
```

### Block C → Block D (1 tensor, `(1, 192, 80, 144)`)

```
/model/_backbone/CameraCylinderEncoder/_camera_encoder/_camera_backbone/_encoder/
  _trifocal_backbone/_neck/_bottom_up.0/_conv_block/_convblock_3/_net/_net.2/Clip_output_0
```

### Block B/D → Block E (5 tensors, each `(1, 64, 128, 64)`)

```
/model/_backbone/CameraDeformedCylinderEncoder/_camera_encoder/_transformation/
  _bev_transformation/_bev_transformation/_reduce_conv/Conv_output_0
  _bev_transformation/_bev_transformation/_reduce_conv_1/Conv_output_0
  _bev_transformation/_bev_transformation/_reduce_conv_2/Conv_output_0
  _bev_transformation/_bev_transformation/_reduce_conv_3/Conv_output_0
/model/_backbone/CameraCylinderEncoder/_camera_encoder/_transformation/
  _bev_transformation/_bev_transformation/_reduce_conv/Conv_output_0
```

### Block E → Block F (1 tensor, `(1, 64, 128, 64)`)

```
vision_bev_encoder_output
```

---

## Files to Create

```
forge/test/models/onnx/vision/bev/
├── model_utils/
│   ├── bev_utils.py                  (existing — unchanged)
│   └── bev_split_utils.py            (NEW — boundary tensor names, extract_model helpers)
├── create_bev_splits.py              (NEW — one-shot: split model + dump intermediate tensors)
├── test_bev_blocks_benchmark.py      (NEW — per-block benchmark + opt sweep)
└── BEV_BLOCK_SPLIT_PLAN.md           (this file)

BEV_model/                            (generated outputs)
├── split_models/
│   ├── block_A_deformed_backbone.onnx
│   ├── block_B_deformed_bev_transform.onnx
│   ├── block_C_cylinder_backbone.onnx
│   ├── block_D_cylinder_bev_transform.onnx
│   ├── block_E_bev_aggregator.onnx
│   └── block_F_output_heads.onnx
└── intermediate_samples/
    └── <seq_id>/
        ├── feat_deformed_cam0.npy        # (1,192,96,96)
        ├── feat_deformed_cam1.npy        # (1,192,96,96)
        ├── feat_deformed_cam2.npy        # (1,192,96,96)
        ├── feat_deformed_cam3.npy        # (1,192,96,96)
        ├── feat_cylinder_cam4.npy        # (1,192,80,144)
        ├── bev_deformed_cam0.npy         # (1,64,128,64)
        ├── bev_deformed_cam1.npy         # (1,64,128,64)
        ├── bev_deformed_cam2.npy         # (1,64,128,64)
        ├── bev_deformed_cam3.npy         # (1,64,128,64)
        ├── bev_cylinder_cam4.npy         # (1,64,128,64)
        └── vision_bev_encoder_output.npy # (1,64,128,64)
```

---

## Implementation Plan

### Phase 1 — Model Splitting and Intermediate Tensor Extraction

**`bev_split_utils.py`**
- Defines all 6 block boundary tensor name constants
- `split_model(full_model_path, output_dir)` — calls `onnx.utils.extract_model` for each block and saves to `split_models/`
- `check_all_splits(split_dir)` — runs `onnx.checker.check_model` on each saved subgraph

**`create_bev_splits.py`** (run once, not a pytest test)
- Loads full model
- Adds all boundary tensors as extra ONNX graph outputs
- Runs ONNX Runtime (CPU) on the augmented model for each available sequence
- Saves boundary tensors as `.npy` files under `BEV_model/intermediate_samples/<seq_id>/`
- Calls `split_model()` to write the 6 ONNX block files
- Prints a summary table of block sizes and boundary tensor shapes

### Phase 2 — Per-Block Benchmark Tests

**`test_bev_blocks_benchmark.py`**

One pytest test per block, each following the same pattern as `test_bev_benchmark.py`:

```
test_block_A  — CameraDeformedCylinder Backbone     (460 nodes)
test_block_B  — CameraDeformedCylinder BEV Transform ( 80 nodes)
test_block_C  — CameraCylinder Backbone             (129 nodes)
test_block_D  — CameraCylinder BEV Transform         ( 20 nodes)
test_block_E  — BEV Aggregator Backbone              ( 57 nodes)
test_block_F  — Output Heads                         ( 24 nodes)
test_pipeline — All 6 blocks in sequence (measures total pipeline FPS)
```

Each test:
1. Loads the split ONNX file
2. Loads inputs: original model inputs for A/C, saved intermediate tensors for B/D/E/F
3. Compiles with `forge.compile` + same `CompilerConfig` as the full-model benchmark
4. Runs `N_WARMUP=3, N_TIMED=10`
5. Reports FPS, mean inference time, std

### Phase 3 — Optimization Sweep per Block

For each block, test these `CompilerConfig` variants and print a side-by-side table:

| Label | Settings |
|-------|----------|
| `baseline` | `enable_optimization_passes=True` (current default) |
| `+consteval` | `MLIRConfig.set_enable_consteval(True)` |
| `+opt_level_1` | `MLIRConfig.set_optimization_level(1)` |
| `+fp16b` | `default_df_override = DataFormat.Float16_b` |
| `+consteval+fp16b` | consteval + fp16b combined |

Output format (one table per block):

```
  Block A: CameraDeformedCylinder Backbone
---------------------------------------------------------------------------
| Configuration                          | Inference Time   | FPS        |
---------------------------------------------------------------------------
| baseline                               | 1234.56 ± 12.34  |   0.81     |
| +consteval                             | 1100.00 ± 10.00  |   0.91     |
| +opt_level_1                           | 1150.00 ± 11.00  |   0.87     |
| +fp16b                                 |  900.00 ±  9.00  |   1.11     |
| +consteval+fp16b                       |  850.00 ±  8.00  |   1.18     |
---------------------------------------------------------------------------
```

If an optimization **hurts** FPS (regression > 5%) or **fails to compile**, that is noted
and the block is flagged for investigation.

---

## Expected Outcomes

| Block | Nodes | Expected bottleneck? | Why |
|-------|-------|---------------------|-----|
| A | 460 | **Yes — likely highest latency** | 219 Conv ops on large 1536² inputs |
| B | 80 | Moderate | GridSample + Gather on feature maps |
| C | 129 | Moderate | Conv on 1280×2304 input (camera 4) |
| D | 20 | Low | Small BEV transform |
| E | 57 | Low–moderate | Conv on compact 128×64 BEV grid |
| F | 24 | Low | Final head Conv on 256×128 grid |

After running the benchmarks we will have:
1. **FPS per block** — pinpoints which block consumes the most wall time
2. **Optimization signal per block** — which flags help, which hurt, which are neutral
3. **Pipeline FPS** — baseline for later pipelining / async execution work
4. A focused optimization target (almost certainly Block A and/or Block C)

---

## Execution Order

```bash
# Step 1: generate split models and intermediate tensors (run once)
python forge/test/models/onnx/vision/bev/create_bev_splits.py

# Step 2: run per-block benchmarks (baseline)
pytest forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py -v -k "block"

# Step 3: run optimization sweep per block
pytest forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py -v -k "opt_sweep"

# Step 4: run full pipeline benchmark for comparison
pytest forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py -v -k "pipeline"
```
