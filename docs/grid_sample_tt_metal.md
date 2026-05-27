# `grid_sample` in tt-metal (TTNN)

## Overview

`ttnn::grid_sample` is the hardware-level op that performs spatial resampling on Tenstorrent
devices. It takes a feature map and a grid of sampling coordinates, and produces a new tensor
where each output pixel is sampled from the input at the corresponding (fractional) coordinate.

---

## Tensor conventions

| Argument | Shape | Memory layout | Data type |
|---|---|---|---|
| `input` | `(N, H_in, W_in, C)` — **NHWC** | ROW_MAJOR | BF16 |
| `grid` | `(N, H_out, W_out, 2)` | ROW_MAJOR | BF16 (direct path) / float32 (precomputed path) |
| `output` | `(N, H_out, W_out, C)` | INTERLEAVED DRAM (after fixup) | BF16 |

**Critical constraint — C must be divisible by 32 (TILE_WIDTH).**  The metal kernel packs channels
into hardware tiles. If C is not a multiple of 32 the kernel will produce garbage output or fault.

---

## C++ API signature

```cpp
// Direct path (kernel computes coordinates at runtime):
ttnn::Tensor ttnn::grid_sample(
    const ttnn::Tensor &input,          // NHWC, ROW_MAJOR, BF16
    const ttnn::Tensor &grid,           // (N, H_out, W_out, 2), ROW_MAJOR, BF16
    const std::string &mode,            // "bilinear" | "nearest"
    const std::string &paddingMode,     // "zeros" | "border" | "reflection"
    bool alignCorners,
    bool usePrecomputedGrid,            // false for direct path
    bool batchOutputChannels,           // false (not used)
    std::optional<ttnn::MemoryConfig> memoryConfig
);

// Precomputed path: host preprocessing + device kernel:
ttnn::Tensor ttnn::prepare_grid_sample_grid(
    const ttnn::Tensor &hostGridF32,    // (N, H_out, W_out, 2), host, float32
    const std::vector<uint32_t> &inputShapeNHWC,  // {N, H_in, W_in, C}
    const std::string &mode,
    const std::string &paddingMode,
    bool alignCorners,
    ttnn::DataType outputDtype          // BF16 for device use
);
// Returns:
//   bilinear → (N, H_out, W_out, 6)  — 4 bilinear weights + top-left pixel coords
//   nearest  → (N, H_out, W_out, 2)  — integer (y, x) pixel coords
```

---

## Interpolation modes

### Bilinear

The kernel reads the four neighbouring integer pixels around the fractional sampling location and
blends them with bilinear weights.

**Coordinate formula hardcoded by the kernel:**

```
x_pixel = (x_norm + 1) * W_in / 2 - 0.5      # align_corners = False
```

The kernel **always** uses this `align_corners=False` formula internally regardless of what is
passed in `alignCorners`. For `align_corners=True` the correct formula is:

```
x_pixel = (x_norm + 1) * (W_in - 1) / 2      # align_corners = True
```

Using the kernel directly with `alignCorners=True` and large input dimensions produces coordinate
errors on the order of `(W_in - 1)/2 - W_in/2 + 0.5 ≈ 0` for large `W_in`, which is small but
still accumulates in BF16 for large grids. The recommended fix is to use the precomputed grid
path (see below) which handles the formula correctly on the host in float32.

### Nearest

The kernel snaps to the nearest integer pixel. The nearest-mode path **always** requires a
precomputed grid — the kernel does not compute nearest-neighbor coordinates at runtime. Passing
`usePrecomputedGrid=false` with `mode="nearest"` is unsupported and will produce incorrect results.

---

## Nearest-mode output memory layout

The nearest-mode kernel produces output in **HEIGHT_SHARDED L1** format, where each shard has
shape `(1, C)`. This is incompatible with subsequent layout conversion ops (e.g. `permute` which
requires tile-aligned shards of at least `(32, 32)`).

**Fix:** Always convert nearest-mode output to INTERLEAVED DRAM immediately after the kernel call:

```cpp
if (output.memory_config().is_sharded()) {
    ::ttnn::MemoryConfig dramInterleaved{
        ::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM};
    output = ::ttnn::to_memory_config(output, dramInterleaved);
}
```

---

## `prepare_grid_sample_grid` — host-side preprocessing

This utility function precomputes pixel coordinates and interpolation weights on the host CPU
before dispatching to the device kernel. It is the recommended path for:

- `mode = "nearest"` (always required)
- `mode = "bilinear"` with `align_corners = True` (kernel formula is wrong otherwise)

### What it computes

**Bilinear output `(N, H_out, W_out, 6)`:**
```
[w_tl, w_tr, w_bl, w_br, y_top_left, x_top_left]
```
where `w_*` are the four bilinear blending weights and `(y, x)` is the top-left integer pixel.

**Nearest output `(N, H_out, W_out, 2)`:**
```
[y_nearest, x_nearest]
```

### Why float32 is required for the grid

`prepare_grid_sample_grid` requires the input grid to be **float32** on the **host**. If the grid
has been converted to BF16 before calling this function, BF16's limited mantissa precision
introduces coordinate quantization error. For large spatial dimensions this error is large enough
to select the wrong pixel:

```
Example: H_in = 96, scale_factor = (H_in - 1) / 2 = 47.5
BF16 step at 47.5 ≈ 0.25  (next representable value)
Coordinate error ≈ 0.38   → wrong nearest-neighbor for integers near 0.5 boundaries
```

The precomputed path resolves this by operating in float32 on the host, where coordinates are
exact, and only converting the final precomputed weights to BF16 for device use.

---

## Execution flow: precomputed vs. direct path

```
Direct path (bilinear, align_corners=False):

  input [NHWC, ROW_MAJOR, BF16] ──┐
                                   ├──► ttnn::grid_sample(use_precomputed_grid=false)
  grid  [ROW_MAJOR, BF16]    ──────┘                       ↓
                                              output [NHWC, ROW_MAJOR, BF16]


Precomputed path (nearest OR bilinear+align_corners=True):

  grid [device] ──► from_device ──► grid [host, float32]
                                          │
                                          ▼
                              prepare_grid_sample_grid(...)
                                          │
                                          ▼
                              precomputedGrid [host, BF16]
                                          │
                                          ▼
                                      to_device ──► precomputedGridDevice [DRAM, BF16]
                                                            │
  input [NHWC, ROW_MAJOR, BF16] ─────────────────┐         │
                                                  ├────► ttnn::grid_sample(use_precomputed_grid=true)
  precomputedGridDevice ──────────────────────────┘         ↓
                                              output [possibly HEIGHT_SHARDED L1]
                                                            │
                                                     to_memory_config (if sharded)
                                                            │
                                                            ▼
                                              output [INTERLEAVED DRAM, BF16]
```

---

## Decision table — which path to use

| `mode` | `align_corners` | Path | Reason |
|---|---|---|---|
| `"bilinear"` | `False` | Direct | Kernel formula is correct |
| `"bilinear"` | `True` | **Precomputed** | Kernel hardcodes align_corners=False formula |
| `"nearest"` | `False` | **Precomputed** | Nearest mode requires precomputed grid |
| `"nearest"` | `True` | **Precomputed** | Both reasons above |
