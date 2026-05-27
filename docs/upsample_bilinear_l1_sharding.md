# Bilinear Upsample: L1 Sharding Analysis

## Current State

Block E (`block_E_bev_aggregator`) has three bilinear upsample ops at opt_level_2. The Memory Layout Analysis (MLA) pass proposes a sharded L1 layout for these ops, which causes a segfault inside `generate_halo_kernel_config_tensors`. The current fix rejects all sharded L1 configs for bilinear upsample, forcing MLA to fall back to DRAM interleaved. This is functionally correct but leaves performance on the table — the metal kernel internally autoreshards from DRAM to L1 HEIGHT_SHARDED anyway.

---

## What tt-metal Actually Supports

**File:** `ttnn/cpp/ttnn/operations/pool/upsample/upsample.cpp`

The bilinear path in `ttnn::upsample()`:

```
if (!input_tensor.is_sharded()):
    → compute_bilinear_autoshard_memory_config()   // build HEIGHT_SHARDED spec
    → to_memory_config(input, sharded_config)       // convert DRAM → L1
    → apply_bilinear_halo_preprocessing()

if (input_tensor.is_sharded() && HEIGHT_SHARDED):
    → apply_bilinear_halo_preprocessing()           // halo exchange, then kernel
```

**Device operation validation** (`upsample_device_operation.cpp` lines 61–73):
```cpp
TT_FATAL(input.memory_config().is_sharded(), "Bilinear upsample requires sharded input tensor");
TT_FATAL(
    input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
    "Bilinear upsample requires HEIGHT_SHARDED input tensor");
```

So:
- **DRAM interleaved** → works (autoresharded internally to HEIGHT_SHARDED)
- **HEIGHT_SHARDED L1** → works if shard spec is valid
- **BLOCK_SHARDED / WIDTH_SHARDED** → metal `TT_FATAL` in device validation

---

## Root Cause of the Segfault

MLA proposes HEIGHT_SHARDED with an arbitrary shard spec. The OpModel passes this directly to `QUERY_OP_CONSTRAINTS(::ttnn::upsample, ...)`. Since the input is already sharded, the autoresharding step is skipped and `apply_bilinear_halo_preprocessing` runs with the wrong shard spec.

Call chain:
```
OpModel::getOpConstraints(HEIGHT_SHARDED with arbitrary shard)
→ ttnn::upsample()  [input.is_sharded() → skips autoshard]
→ apply_bilinear_halo_preprocessing()
→ ttnn::halo()
→ UntilizeWithHaloProgramFactory::create()
→ generate_halo_kernel_config_tensors(tensor_metadata, shard_boundaries)
→ SEGFAULT: tensor_metadata[global_idx] out of bounds
```

The crash is an out-of-bounds access. `tensor_metadata` is sized based on total NHW sticks. `shard_boundaries` are computed assuming a specific shard height. When MLA's shard height doesn't match the autoshard formula, the boundary indices exceed `tensor_metadata.size()`.

---

## The Autoshard Formula

**File:** `upsample.cpp` lines 65–81

```cpp
static tt::tt_metal::MemoryConfig compute_bilinear_autoshard_memory_config(
    const ttnn::Tensor& input_tensor) {

    const uint32_t total_input_sticks = N * H * W;   // logical_shape [0]*[1]*[2]
    const uint32_t max_num_cores = grid.x * grid.y;  // device compute grid
    const uint32_t num_shards = std::min(max_num_cores, total_input_sticks);
    const uint32_t shard_height = round_up(total_input_sticks, num_shards) / num_shards;

    ShardSpec shard_spec(
        num_cores_to_corerangeset(num_shards, grid, true),
        {shard_height, C},                            // [height, width]
        ShardOrientation::ROW_MAJOR);

    return MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, shard_spec);
}
```

For the BEV block E upsample inputs (WH N150, 8×8 grid = 64 cores):

| Op | Input shape | N\*H\*W | num_shards | shard_height | shard_spec |
|---|---|---|---|---|---|
| upsample2d #1 | `1×16×8×256` | 128 | 64 | 2 | `[2, 256]` |
| upsample2d #2 | `1×32×16×128` | 512 | 64 | 8 | `[8, 128]` |
| upsample2d #3 | `1×64×32×64` | 2048 | 64 | 32 | `[32, 64]` |

The shard height is strictly determined by input shape and device grid. MLA's proposed shard spec almost certainly differs from this, which is why the crash occurs.

---

## Required Fix

### In `TTNNOpModel.cpp` — `getOpConstraints` and `getOpRuntime`

Replace the blanket rejection of all sharded L1 with a two-pronged approach:

**1. Reject non-HEIGHT_SHARDED sharded configs** (these metal `TT_FATAL`)

**2. For HEIGHT_SHARDED: recompute the valid shard spec before querying**

Instead of passing the MLA-proposed TensorSpec directly to `QUERY_OP_CONSTRAINTS`, rebuild the input TensorSpec using the autoshard formula. This ensures the query uses a shard spec that matches what the halo preprocessing expects.

```cpp
if (mode == "bilinear") {
    if (inputLayout.hasL1BufferType() && inputLayout.getMemLayout()) {
        auto memLayout = inputLayout.getMemLayout().getValue();

        // Reject BLOCK_SHARDED / WIDTH_SHARDED — metal TT_FATALs for these
        if (isShardedMemoryLayout(memLayout) &&
            memLayout != TensorMemoryLayout::HeightSharded) {
            return llvm::createStringError(
                "Bilinear upsample only supports HEIGHT_SHARDED L1");
        }

        if (memLayout == TensorMemoryLayout::HeightSharded) {
            // Recompute the valid autoshard spec for this input shape.
            // MLA's shard spec is arbitrary; using it directly causes
            // generate_halo_kernel_config_tensors to go out of bounds.
            //
            // autoshard formula (mirrors compute_bilinear_autoshard_memory_config):
            //   total_nhw = inputShape[0] * inputShape[1] * inputShape[2]
            //   num_shards = min(grid.x * grid.y, total_nhw)
            //   shard_height = round_up(total_nhw, num_shards) / num_shards
            //   shard_spec = {shard_height, inputShape[3]}, HEIGHT_SHARDED
            //
            // → build a new TensorSpec with this shard spec
            // → call QUERY_OP_CONSTRAINTS with the rebuilt spec
        }
    }
}
```

### Implementation steps

| Step | File | Work |
|---|---|---|
| Change guard to allow HEIGHT_SHARDED | `TTNNOpModel.cpp` (getOpConstraints + getOpRuntime) | ~5 lines |
| Add helper `computeBilinearAutoshardSpec(device, inputShape, channels)` | `TTNNOpModel.cpp` | ~20 lines |
| Rebuild inputSpec from corrected metal MemoryConfig | `TTNNOpModel.cpp` | ~10 lines — use `detail::convertToTensorSpec` or construct `TensorSpec` directly from metal types |
| Compute matching output shard spec for HEIGHT_SHARDED output | `TTNNOpModel.cpp` | ~10 lines — output total_nhw = N * out_H * out_W, same formula |
| Test at opt_level_2 | `test_bev_blocks_benchmark.py block_E` | Verify no segfault, L1 sharded path is selected |

### Key implementation detail

`detail::convertToTensorSpec` takes a `TTNNLayoutAttr` and produces a metal `TensorSpec`. The challenge is that the corrected shard spec lives in the metal type system, not the MLIR type system. Two options:

**Option A** — Construct `TensorSpec` directly from metal types (bypasses MLIR conversion):
```cpp
uint64_t total_nhw = inputShape[0] * inputShape[1] * inputShape[2];
uint64_t channels = inputShape[3];
auto grid = device->compute_with_storage_grid_size();
uint64_t num_shards = std::min((uint64_t)(grid.x * grid.y), total_nhw);
uint64_t shard_h = (total_nhw + num_shards - 1) / num_shards;
auto core_range = ::tt::tt_metal::num_cores_to_corerangeset(num_shards, grid, true);
auto shard_spec = ::tt::tt_metal::ShardSpec(
    core_range, {shard_h, channels}, ::tt::tt_metal::ShardOrientation::ROW_MAJOR);
auto mem_config = ::tt::tt_metal::MemoryConfig(
    ::tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
    ::tt::tt_metal::BufferType::L1,
    shard_spec);
// Build TensorSpec from original inputSpec but with new mem_config
```

**Option B** — Pass DRAM interleaved to the query and let tt-metal autoshard internally (current working path). This queries the correct constraints but reports DRAM cost, not L1 cost, so MLA won't prefer the L1 path.

Option A gives MLA accurate L1 cost for HEIGHT_SHARDED bilinear upsample. Option B is the current state.

---

## Performance Impact

With DRAM interleaved (current):
- MLA places input in DRAM
- `ttnn::upsample` autoreshards DRAM → HEIGHT_SHARDED L1 at runtime (extra H2D copy)
- Output is HEIGHT_SHARDED L1

With HEIGHT_SHARDED L1 (after fix):
- MLA places input in HEIGHT_SHARDED L1 directly (no runtime reshard)
- `ttnn::upsample` runs halo and bilinear kernel entirely in L1
- Eliminates the DRAM round-trip on the input tensor

For the three upsample ops in block E, this saves 3× `to_memory_config` (DRAM→L1) calls at runtime. On the BEV aggregator model the upsample inputs range from 32KB (1×16×8×256 BF16) to 512KB (1×64×32×64 BF16) — the DRAM round-trip cost is measurable.
