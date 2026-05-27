# Block D BEV — GridSample Trace Capture Fix

## Overview

`test_opt_sweep[enable_program_cache-opt_level_2_bfloat16_hifi3_fp32_acc_trace_enabled-block_D]`
failed at runtime with:

```
RuntimeError: TT_FATAL @ fd_mesh_command_queue.cpp:622: !trace_id_.has_value()
info: Reads are not supported during trace capture.
```

The Block D BEV model uses 8 `GridSample` ops (mode=nearest, align_corners=True) with
input shape `(1,64,80,144)` and grid shape `(1,128,64,2)`. When compiled with
`trace_enabled=True`, these ops fail during the trace capture phase.

All fixes are in `third_party/tt-mlir` only.

---

## Root Cause — `from_device` Inside Trace Capture Boundary

### TTNN Trace Architecture

The tt-mlir trace infrastructure (`TTNNTraceHoistTransform`) generates a **capture
program** that runs the trace function **twice**:

```
capture_program:
  1. WriteTensorOp          — copy host inputs to device trace-input slots
  2. CallOp traceFunc        — warmup run (no trace active); populates program cache
  3. BeginTraceCaptureOp     — activates TTNN trace capture on CQ-0
  4. CallOp traceFunc        — trace-capture run (trace active); records device kernels
  5. EndTraceCaptureOp       — completes trace capture
  6. ExecuteTraceOp          — replay once to populate output slots
```

After `BeginTraceCaptureOp` (step 3), TTNN trace capture is active. Any device read
(`from_device` / `enqueue_read`) triggers `TT_FATAL @ fd_mesh_command_queue.cpp:622:
!trace_id_.has_value()`.

### The Failing Code Path

For `mode="nearest"` or `alignCorners=True`, `GridSampleOp::run` (in
`grid_sample.cpp`) enters the `needsPrecomputedGrid` path:

```cpp
bool needsPrecomputedGrid = (mode == "nearest") || alignCorners;
if (needsPrecomputedGrid) {
    ::ttnn::Tensor hostGrid = ::ttnn::from_device(grid);   // ← FAILS at step 4
    // ... CPU precomputation ...
    ::ttnn::Tensor precomputedGridDevice = ::ttnn::to_device(precomputedGrid, ...);
    ::ttnn::grid_sample(..., use_precomputed_grid=true, ...);
}
```

The `from_device(grid)` at step 4 fails because trace capture forbids device reads.

The warmup call (step 2) runs the same code with trace inactive — `from_device`
succeeds there. The fix exploits this timing difference.

### Why Block D and Not Other Configs?

Block D uses `mode="nearest"` with `alignCorners=True` — both conditions trigger
`needsPrecomputedGrid=true`. The trace-enabled config is the only one that runs
`BeginTraceCaptureOp`, so this failure only appears in the `trace_enabled` variant.

---

## Fix — ProgramContext Precomputed Grid Cache

**Files:**
- `runtime/lib/ttnn/operations/pool/grid_sample.cpp`
- `runtime/include/tt/runtime/detail/ttnn/types/types.h`

The fix uses `ProgramContext::getOrCreateImplicitPrecomputedGrid` — the same pattern
already used by `getOrCreateImplicitGlobalSemaphore` for GlobalSemaphore resources.

```
warmup run (step 2, no trace):
  → factory lambda invoked (cache miss)
  → from_device(grid) succeeds
  → CPU precompute via prepare_grid_sample_grid(...)
  → to_device(precomputedGrid)
  → stored in root ProgramContext::implicitOpPrecomputedGrids

trace-capture run (step 4, trace active):
  → factory lambda NOT invoked (cache hit via parentContext forwarding)
  → use cached precomputedGridDevice directly
  → call grid_sample(..., use_precomputed_grid=true)   (recorded in trace)
```

### Cache Key Stability

The cache key is `reinterpret_cast<uintptr_t>(op)` — the flatbuffer `GridSampleOp*`
pointer.

**Why not `grid.buffer()->address()`?**
In Block D the `grid` argument to `GridSampleOp` is not a model constant — it is the
**output of a `GatherOp`** (intermediate tensor):

```
%9 = GatherOp(input_lut_4, index_i)   // allocates at address X on warmup
%10 = GridSampleOp(data, %9)
```

The capture program deallocates the warmup result (`ttnn.deallocate(%warmup_out)`)
before the trace-capture run begins. On the trace-capture run, `GatherOp` re-allocates
its output — potentially at a different address Y. So `grid.buffer()->address()`
gives X on warmup and Y on trace-capture → cache miss.

**Why `reinterpret_cast<uintptr_t>(op)` works:**
`const ::tt::target::ttnn::GridSampleOp *op` is a pointer into the static flatbuffer
binary. It is unique per op instance, invariant for the lifetime of the program, and
identical on every call regardless of intermediate tensor addresses. Each of the 8
`GridSampleOp` instances in Block D has a distinct pointer value → distinct cache entry.

### parentContext Forwarding

`FuncCallOp::run` creates a sub-context with `parentContext = &context`. When
`getOrCreateImplicitPrecomputedGrid` is called from the sub-context during either
the warmup or trace-capture invocation, it forwards to the root context. Both calls
therefore share the same cache entry.

### Why ProgramContext Instead of Static Storage

An earlier approach used a `static std::unordered_map<uintptr_t, ::ttnn::Tensor>`
and a `std::atexit` handler to clear it before process exit. This failed:

- The static map holds `::ttnn::Tensor` objects that call
  `GraphTracker::instance().is_enabled()` in their destructor.
- Library unloading order at process exit causes `GraphTracker` (in `libtt_metal.so`)
  to be destroyed BEFORE the static map in `libTTMLIRRuntime.so`.
- `Tensor::~Tensor()` → `deallocate_impl()` → `GraphTracker::instance().is_enabled()`
  on an already-destroyed singleton → SIGSEGV.
- `std::atexit` handlers registered from `libTTMLIRRuntime.so` run during that
  library's own `__cxa_finalize`, which is AFTER `GraphTracker`'s library finalizes —
  so the atexit approach cannot clear the map in time.

`ProgramContext` is bound to the device lifetime: it is created and destroyed while
the device (and `GraphTracker`) are still alive. Storing tensors in
`implicitOpPrecomputedGrids` (a `ProgramContext` member) ensures they are destroyed
safely during context teardown, not at static-destructor time.

### Implementation

**`types.h` — `ProgramContext` additions:**

```cpp
// Public method:
::ttnn::Tensor getOrCreateImplicitPrecomputedGrid(
    uintptr_t opKey,
    const std::function<::ttnn::Tensor()> &factory) {
  if (parentContext) {
    return parentContext->getOrCreateImplicitPrecomputedGrid(opKey, factory);
  }
  auto it = implicitOpPrecomputedGrids.find(opKey);
  if (it == implicitOpPrecomputedGrids.end()) {
    it = implicitOpPrecomputedGrids.emplace(opKey, factory()).first;
  }
  return it->second;
}

// Private field:
std::unordered_map<uintptr_t, ::ttnn::Tensor> implicitOpPrecomputedGrids;
```

**`grid_sample.cpp` — needsPrecomputedGrid path:**

```cpp
uintptr_t gridCacheKey = reinterpret_cast<uintptr_t>(op);

::ttnn::Tensor precomputedGridDevice =
    context.getOrCreateImplicitPrecomputedGrid(
        gridCacheKey, [&]() -> ::ttnn::Tensor {
          ::ttnn::Tensor hostGrid = ::ttnn::from_device(grid);
          if (hostGrid.layout() != ::ttnn::Layout::ROW_MAJOR)
            hostGrid = ::ttnn::to_layout(hostGrid, ::ttnn::Layout::ROW_MAJOR);
          ::ttnn::Tensor hostGridF32 =
              (hostGrid.dtype() == ::ttnn::DataType::FLOAT32)
                  ? hostGrid
                  : ::ttnn::typecast(hostGrid, ::ttnn::DataType::FLOAT32);
          ::ttnn::Tensor precomputedGrid = ::ttnn::prepare_grid_sample_grid(
              hostGridF32, inputShapeNHWC, mode, paddingMode, alignCorners,
              ::ttnn::DataType::BFLOAT16);
          return ::ttnn::to_device(precomputedGrid, &device, dramInterleaved);
        });

::ttnn::grid_sample(input, precomputedGridDevice, ..., use_precomputed_grid=true, ...);
```

---

## Non-Trace Execution Path — No Change

When `set_enable_trace(False)` (the default), the factory lambda is called on the
first invocation and the result is cached in `implicitOpPrecomputedGrids` for
subsequent calls within the same program execution. `from_device` is always called
during that first invocation. There is no behavioral change for non-trace models.

---

## File Summary

| File | Change |
|------|--------|
| `runtime/lib/ttnn/operations/pool/grid_sample.cpp` | Use `context.getOrCreateImplicitPrecomputedGrid` instead of static cache; removed all static storage, mutex, and trace-active detection |
| `runtime/include/tt/runtime/detail/ttnn/types/types.h` | Added `getOrCreateImplicitPrecomputedGrid` method and `implicitOpPrecomputedGrids` map to `ProgramContext` |

---

## Test Results

| Test | Before | After |
|------|--------|-------|
| `opt_level_2_bfloat16_hifi3_fp32_acc_trace_enabled-block_D` | **FAIL (TT_FATAL trace read)** | **PASS (clean exit)** |
| `opt_level_2_bfloat16_hifi3_fp32_acc-block_D` (no trace) | PASS | PASS (no regression) |
