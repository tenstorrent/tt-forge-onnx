# Block C — ConvTranspose2d weights_dtype Compile Crash

## 1. Affected Test Cases

- `test_opt_sweep[enable_program_cache-opt_level_1-block_C]` — FAIL
- `test_opt_sweep[disable_program_cache-opt_level_1-block_C]` — FAIL
- `test_opt_sweep[enable_program_cache-opt_level_2-block_C]` — FAIL
- `test_conv_transpose2d_block_c[op1_20x36-opt_level_1]` — FAIL
- `test_conv_transpose2d_block_c[op2_40x72-opt_level_1]` — FAIL
- `test_conv_transpose2d_block_c[op3_80x144-opt_level_1]` — FAIL

Block: `block_C_cylinder_backbone`. Has four ConvTranspose2d ops: in_ch=192, out_ch=192, kernel=2x2, stride=2x2, no bias, f32.

## 2. Failure

```
TT_FATAL @ conv2d_op_program_factory_common.cpp:91:
  get_cb_info expects conv_config.weights_dtype to be already set
```

## 3. Failure Reason

`prepare_conv_transpose2d_weights.cpp` computes `weight_dtype` from:

```cpp
DataType weight_dtype = conv_config.weights_dtype.value_or(weight_tensor.dtype());
```

But **never writes `weight_dtype` back** into `conv_config.weights_dtype`. So `conv_config.weights_dtype` remains `nullopt`.

Later, the DRAM slice determination path calls:

```
determine_slice_config -> get_L1_usage -> calculate_L1_usage -> get_cb_info
```

And `get_cb_info` asserts:

```cpp
TT_FATAL(conv_config.weights_dtype.has_value(),
         "get_cb_info expects conv_config.weights_dtype to be already set");
```

The `conv_transpose2d_L1` (line 123–125) and `conv_transpose2d_DRAM` (lines 1029–1031) paths already set `weights_dtype` early. `prepare_conv_transpose2d_weights` was the only code path that did not.

The bug only surfaces for larger ConvTranspose2d ops (op1–op3) because those exercises the DRAM slice determination path. The smallest op (op0, 10x18 output) fits in L1 entirely and takes a different code path that never reaches `get_cb_info` via `determine_slice_config`.

The `TTNNOpModel.cpp` also has the same gap: `getOpConstraints` and `getPrepareConv2dWeightsOpOutputTensorSpec` for ConvTranspose2d were calling `QUERY_OP_CONSTRAINTS` without setting `weights_dtype` in the config, causing the same fatal error during MLA layout evaluation at opt_level_2.

## 4. Fix Implementation Details

**Fix 1 — `prepare_conv_transpose2d_weights.cpp`**

After computing `weight_dtype`, write it back if not already set. This is the minimal, correct fix at the source of the problem: `weight_dtype` is computed but the result is only used locally. Writing it back to `conv_config` ensures every downstream call to `get_cb_info` finds the field populated.

```cpp
DataType weight_dtype = conv_config.weights_dtype.value_or(weight_tensor.dtype());
// Ensure weights_dtype is set on conv_config before it is used in slice determination
// (get_cb_info requires conv_config.weights_dtype to be set).
if (!conv_config.weights_dtype.has_value()) {
    conv_config.weights_dtype = weight_dtype;
}
```

**Fix 2 — `TTNNOpModel.cpp::getOpConstraints` for ConvTranspose2dOp**

Before calling `QUERY_OP_CONSTRAINTS`, ensure `conv2dConfigConverted->weights_dtype` is set from the weight tensor spec. Without this, MLA crashes during the layout evaluation of ConvTranspose2d ops at opt_level_2 (where the op model is queried for all candidate layouts):

```cpp
// The tt-metal conv_transpose2d_DRAM path calls get_cb_info which requires
// weights_dtype to be set in conv_config.
if (conv2dConfigConverted.has_value()) {
    if (!conv2dConfigConverted->weights_dtype.has_value()) {
        conv2dConfigConverted->weights_dtype = weightSpec.data_type();
    }
} else {
    ::ttnn::Conv2dConfig defaultConfig;
    defaultConfig.weights_dtype = weightSpec.data_type();
    conv2dConfigConverted = defaultConfig;
}
```

(Identical guard applied in both `getOpConstraints` and `getOpRuntime` for ConvTranspose2dOp.)

**Fix 3 — `TTNNOpModel.cpp::getPrepareConv2dWeightsOpOutputTensorSpec`**

When `transpose=true` and `outputDtype` is available, set `weights_dtype` before the query closure captures `conv2dConfigConverted`:

```cpp
if (transpose && outputDtype.has_value()) {
    if (!conv2dConfigConverted.has_value())
        conv2dConfigConverted = ::ttnn::Conv2dConfig{};
    if (!conv2dConfigConverted->weights_dtype.has_value())
        conv2dConfigConverted->weights_dtype = *outputDtype;
}
```

## 5. Files Changed with Diffs

**`ttnn/cpp/ttnn/operations/conv/conv_transpose2d/prepare_conv_transpose2d_weights.cpp`** (tt-metal)
```diff
     DataType weight_dtype = conv_config.weights_dtype.value_or(weight_tensor.dtype());
+    // Ensure weights_dtype is set on conv_config before it is used in slice determination
+    // (get_cb_info requires conv_config.weights_dtype to be set).
+    if (!conv_config.weights_dtype.has_value()) {
+        conv_config.weights_dtype = weight_dtype;
+    }
```

**`lib/OpModel/TTNN/TTNNOpModel.cpp`** — `getPrepareConv2dWeightsOpOutputTensorSpec` (tt-mlir)
```diff
+  // For ConvTranspose2d, get_cb_info requires conv_config.weights_dtype to already be set.
+  if (transpose && outputDtype.has_value()) {
+    if (!conv2dConfigConverted.has_value()) {
+      conv2dConfigConverted = ::ttnn::Conv2dConfig{};
+    }
+    if (!conv2dConfigConverted->weights_dtype.has_value()) {
+      conv2dConfigConverted->weights_dtype = *outputDtype;
+    }
+  }
```

**`lib/OpModel/TTNN/TTNNOpModel.cpp`** — `getOpConstraints` for ConvTranspose2dOp (tt-mlir)
```diff
+  // The tt-metal conv_transpose2d_DRAM path calls get_cb_info which requires
+  // weights_dtype to be set in conv_config.
+  if (conv2dConfigConverted.has_value()) {
+    if (!conv2dConfigConverted->weights_dtype.has_value()) {
+      conv2dConfigConverted->weights_dtype = weightSpec.data_type();
+    }
+  } else {
+    ::ttnn::Conv2dConfig defaultConfig;
+    defaultConfig.weights_dtype = weightSpec.data_type();
+    conv2dConfigConverted = defaultConfig;
+  }
```
(Identical guard also added in `getOpRuntime` for ConvTranspose2dOp.)

## 6. After Fix — How It Works

The call chain for larger ConvTranspose2d ops (e.g., op1 with output 20x36) now flows correctly:

1. `prepare_conv_transpose2d_weights()` is called with `conv_config.weights_dtype = nullopt`.
2. `weight_dtype = conv_config.weights_dtype.value_or(weight_tensor.dtype())` computes `BFloat16` (from the weight tensor's actual dtype).
3. The new guard writes it back: `conv_config.weights_dtype = BFloat16`.
4. `determine_slice_config()` is called.
5. `get_L1_usage()` → `calculate_L1_usage()` → `get_cb_info()` finds `conv_config.weights_dtype.has_value() == true`.
6. `get_cb_info` proceeds without assertion failure, returning the correct CB size for the slice configuration.
7. The DRAM slice path correctly partitions the weight tensors and compiles successfully.

All four ConvTranspose2d ops (op0–op3) compile and run cleanly at opt_level_0/1/2.

## 7. Test Results

| Test | Before | After |
|------|--------|-------|
| `test_conv_transpose2d_block_c[op0_10x18-opt_level_1/2]` | PASS | PASS |
| `test_conv_transpose2d_block_c[op1_20x36-opt_level_1/2]` | **FAIL** | **PASS** |
| `test_conv_transpose2d_block_c[op2_40x72-opt_level_1/2]` | **FAIL** | **PASS** |
| `test_conv_transpose2d_block_c[op3_80x144-opt_level_1/2]` | **FAIL** | **PASS** |
| `test_opt_sweep[opt_level_1-block_C]` | **FAIL** | **PASS** |
