# ONNX Pow Operation — Complete Reference and Transpiler Integration Guide

Based on the [official ONNX Pow documentation](https://onnx.ai/onnx/operators/onnx__Pow.html), this document provides a comprehensive summary of all opset versions and a step-by-step guide for integrating the Pow operator into the Forge transpiler pipeline.

## Overview

The **Pow** operator performs element-wise exponentiation between two tensors. Unlike simpler unary operations (Relu, Sqrt), Pow is a **binary operator** — it takes both a base tensor **X** and an exponent tensor **Y** as inputs, producing `Z = X ** Y` element-wise.

**Key Behavior**:
- Performs element-wise power: `Z = X ^ Y`
- Supports multidirectional (NumPy-style) broadcasting (OPSET 7+)
- Limited broadcast support in earlier versions (OPSET 1-6)
- X and Z share type T; Y may be a different type T1 (OPSET 12+)
- Y is frequently a constant scalar (e.g., squaring: Y=2, cube-root: Y=0.333)

**Critical difference from Add/Mul**: Pow uses **heterogeneous type constraints** from v12 onward — the base (X) and exponent (Y) can have different types. X and the output Z share the same type T, while Y is type T1 which is broader (includes integer types not supported in T).

**Examples**:
- X: `[2.0, 3.0, 4.0]`, Y: `2.0` (scalar) → Z: `[4.0, 9.0, 16.0]`
- X: `[[1.0, 2.0], [3.0, 4.0]]`, Y: `[2.0, 3.0]` → Z: `[[1.0, 8.0], [9.0, 64.0]]` (broadcasting)
- X: `[4.0, 9.0, 16.0]`, Y: `0.5` → Z: `[2.0, 3.0, 4.0]` (square root via Pow)

**Important**:
- When Y is a constant scalar (the overwhelmingly common case in neural networks), this collapses to a unary-style operation that can be handled with a fixed `exponent` attribute
- When Y is a runtime tensor, full binary broadcasting semantics apply
- Broadcasting between X and Y follows the same rules as Add, Mul, etc. (OPSET 7+)

---

## Version-by-Version Breakdown

### **Pow v1** (since version 1)

**Key Characteristics:**
- **Broadcasting**: Limited broadcast support (requires `broadcast=1` attribute)
- **Broadcast Behavior**: Right-hand-side (Y) is broadcasted to match left-hand-side (X)
- **Axis Attribute**: Optional `axis` attribute to specify broadcast dimensions
- **Homogeneous types**: X and Y must have the same type T
- **Inputs**:
  - `X` (T): Input tensor, base of the exponent
  - `Y` (T): Input tensor broadcastable to X shape, the exponent component
- **Outputs**:
  - `Z` (T): Output tensor (same size as X)
- **Type Constraints**:
  - **LIMITED**: Only float types supported
  - `tensor(double)`, `tensor(float)`, `tensor(float16)`
- **Shape Inference**: Yes (shape inference: True)
- **Function**: No
- **Support Level**: COMMON

**Attributes:**

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `broadcast` | INT | No | `0` | Pass 1 to enable broadcasting |
| `axis` | INT | No | - | If set, defines the broadcast dimensions |

**Broadcasting Rules (v1):**
- Right-hand-side (Y) can be:
  - Scalar tensor (empty shape `()`)
  - 1-element tensor (shape with all 1s)
  - Contiguous suffix of X's shape (suffix matching by default)
  - Contiguous subset starting at `axis` if specified
- 1-dim expansion does not work
- Examples (with `broadcast=1`):
  - `shape(X) = (2, 3, 4, 5)`, `shape(Y) = ()` → scalar
  - `shape(X) = (2, 3, 4, 5)`, `shape(Y) = (5,)` → suffix match
  - `shape(X) = (2, 3, 4, 5)`, `shape(Y) = (4, 5)` → suffix match
  - `shape(X) = (2, 3, 4, 5)`, `shape(Y) = (3, 4)`, `axis=1` → axis-specified

**Supported Types (v1):**
- Floating point only: `float`, `double`, `float16`

---

### **Pow v7** (since version 7)

**Key Characteristics:**
- **Broadcasting**: Multidirectional (NumPy-style) broadcasting — **ALWAYS ENABLED**
- **No Attributes**: `broadcast` and `axis` attributes removed
- **Homogeneous types**: X and Y still share the same type T (no T1 yet)
- **Inputs**:
  - `X` (T): First operand, base of the exponent
  - `Y` (T): Second operand, power of the exponent
- **Outputs**:
  - `Z` (T): Output tensor
- **Type Constraints**:
  - Same as v1 — only float types
  - `tensor(double)`, `tensor(float)`, `tensor(float16)`
- **Shape Inference**: Yes
- **Function**: No
- **Support Level**: COMMON

**Attributes:** None (all removed)

**Changes from v1:**
- Full NumPy-style broadcasting always enabled — no `broadcast=1` needed
- Removed `broadcast` and `axis` attributes
- Automatic dimension alignment from rightmost dimension
- No type expansion yet (still float-only, homogeneous)

**Broadcasting Rules (v7+):**
- Full NumPy-style multidirectional broadcasting
- Automatic dimension alignment from right to left
- Dimensions of size 1 are automatically expanded
- Missing dimensions on the left are treated as size 1

**Supported Types (v7):**
- Floating point only: `float`, `double`, `float16`

---

### **Pow v12** (since version 12)

**Key Characteristics:**
- **Broadcasting**: Multidirectional (NumPy-style) — **ALWAYS ENABLED** (same as v7)
- **MAJOR CHANGE**: Introduces **heterogeneous type constraints** — X (type T) and Y (type T1) may now differ
- **Inputs**:
  - `X` (T): First operand, base of the exponent
  - `Y` (T1): Second operand, power of the exponent — **new type T1**
- **Outputs**:
  - `Z` (T): Output tensor — same type as X
- **Type Constraints**:
  - **T** (X and Z): `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`
  - **T1** (Y only): `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int8)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`
- **Shape Inference**: Yes
- **Function**: No
- **Support Level**: COMMON

**Changes from v7:**
- **BREAKING**: Introduced heterogeneous type constraints (T vs T1)
- Added integer support for X/Z: `int32`, `int64`
- T1 (exponent Y) now supports all integer widths: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`
- Output Z always has type T (same as base X), regardless of Y type

**Supported Types (v12):**
- T (X and Z): `float`, `double`, `float16`, `int32`, `int64`
- T1 (Y only): `float`, `double`, `float16`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`

---

### **Pow v13** (since version 13)

**Key Characteristics:**
- Same as v12 with type expansion
- **T adds `bfloat16`** for X and Z
- **T1 unchanged** from v12
- **Inputs/Outputs**: Same as v12
- **Type Constraints**:
  - **T**: `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)` (added `bfloat16`)
  - **T1**: Same as v12 — no `bfloat16` in T1
- **Shape Inference**: Yes

**Changes from v12:**
- Added `tensor(bfloat16)` to T (base X and output Z)

---

### **Pow v15** (since version 15)

**Key Characteristics:**
- Same as v13 with type expansion
- **T1 adds `bfloat16`** — exponent Y can now also be bfloat16
- **Type Constraints**:
  - **T** (X and Z): `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)` — **unchanged from v13**
  - **T1** (Y only): `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int8)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint8)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)` (added `bfloat16`)
- **Shape Inference**: Yes

**Changes from v13:**
- Added `tensor(bfloat16)` to T1 (exponent Y)

---

## Summary of Changes Across Versions

### Type Support Evolution

| Version | T (X and Z) | T1 (Y only) | Key Change |
|---------|-------------|-------------|------------|
| **v1** | `float`, `double`, `float16` | same as T (homogeneous) | Initial version, float only |
| **v7** | same as v1 | same as T (homogeneous) | Multidirectional broadcasting |
| **v12** | + `int32`, `int64` | float + all int widths | Heterogeneous types introduced |
| **v13** | + `bfloat16` | same as v12 | bfloat16 base/output |
| **v15** | same as v13 | + `bfloat16` | bfloat16 exponent |

### Broadcasting Evolution

| Version | Broadcasting | Attributes | Notes |
|---------|--------------|------------|-------|
| **v1** | Limited (requires `broadcast=1`) | `broadcast`, `axis` | Right-hand-side broadcasted to left |
| **v7+** | Multidirectional (always on) | None | Full NumPy-style broadcasting |

### Key Architectural Difference vs. Add/Mul

Unlike Add/Mul (where both inputs and output share one type T), Pow from v12 onward uses **two separate type variables**:

```
X: type T  (base)
Y: type T1 (exponent — broader type set)
Z: type T  (output — always same type as X, not Y)
```

This matters for the converter: the output dtype must be taken from X's dtype, not Y's dtype.

---

## Key Behavioral Notes

1. **Constant Exponent vs. Tensor Exponent**:
   - In practice, the exponent Y is almost always a constant initializer (e.g., squaring in normalization layers, 0.333 for cube-roots)
   - When Y is a constant, the converter extracts its scalar value and stores it as an attribute — collapsing to a unary-style operation
   - When Y is a runtime tensor, full binary broadcasting semantics apply

2. **Output Type**:
   - Z always has type T (same as X), never T1 (unlike Y which can be a different type)
   - The converter must always read output dtype from X's dtype

3. **Broadcasting**:
   - v7+: Same full NumPy broadcasting as Add, Mul, Sub, Div
   - Shape inference: `output_shape = broadcast_shape(X.shape, Y.shape)`

4. **Legacy Attributes (v1 only)**:
   - `broadcast=1` must be set to enable any broadcasting in v1
   - `axis` specifies start of alignment for non-suffix broadcasting

5. **Special values of Y**:
   - `Y = 0.5` → square root (equivalent to `torch.sqrt`, but Pow is more general)
   - `Y = 2.0` → squaring (common in normalization: variance = mean of squares)
   - `Y = -1.0` → reciprocal (equivalent to `torch.reciprocal`)
   - `Y = 0.333...` → cube root

---

## Transpiler Integration Guide

This section explains every code component needed to fully support the ONNX Pow operator in the forge-onnx transpiler pipeline. The integration has **four layers**: TIR operation node, shape mixin, ONNX converter, and engine registration.

### Current Implementation Status

The Pow operator is **partially implemented** in the transpiler:

| Component | Status | Notes |
|-----------|--------|-------|
| `PowNode` in `operations/activation.py` | Exists | Handles constant-exponent case only |
| `PowShape` in `operations/shape_mixins.py` | Exists | Uses `ElementwiseUnaryShape` — correct for constant Y |
| `PowConverter` in `converters/` | **Missing** | No dedicated converter — Pow falls through as unsupported |
| `"Pow"` in `engine.py` op registry | **Missing** | Not registered in `_op_converters` |
| `"Pow"` in `op_shape_meta.py` | Exists | Registered as `SHAPE_ONLY` |

The current `PowNode` was designed assuming Y is always a constant scalar (extracted by the converter). The following sections describe how to implement the missing pieces and upgrade `PowNode` to handle both the constant-Y and tensor-Y cases correctly.

---

### Layer 1 — TIR Operation Node (`operations/activation.py`)

The TIR node is the framework-agnostic representation of the operation that `eval()` executes using PyTorch.

**Current implementation** (handles constant exponent only):

```python
class PowNode(ElementwiseUnaryShape, TIRNode):
    """
    Element-wise power operation: Z = X ^ exponent.

    Maps to torch.pow(X, exponent) where exponent is a constant scalar
    attribute set at conversion time.  Maps to forge.op.Pow.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        exponent: float,
    ) -> "PowNode":
        return PowNode(
            name=name,
            op_type="Pow",
            inputs=inputs,
            outputs=outputs,
            attrs={"exponent": float(exponent)},
            forge_op_name="Pow",
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        exponent = self.attrs["exponent"]
        x = input_tensors[self.input_names[0]]
        return {self.output_names[0]: torch.pow(x, exponent)}
```

**Why `ElementwiseUnaryShape` is correct here**: When Y is a constant scalar, the output shape always equals X's shape exactly — there is no broadcasting to compute. `ElementwiseUnaryShape.infer_output_shapes()` returns `{output: input_shape}` which is correct.

**Upgraded implementation** (handles both constant and tensor exponent):

```python
class PowNode(BinaryBroadcastShape, TIRNode):
    """
    Element-wise power operation: Z = X ^ Y.

    Supports two modes:
    - Constant exponent: Y is a scalar stored in attrs["exponent"].
      Created by PowConverter when the ONNX Y input is an initializer.
      eval() calls torch.pow(X, exponent_scalar).
    - Tensor exponent: Y is a second runtime input tensor.
      Created by PowConverter when the ONNX Y input is an activation.
      eval() calls torch.pow(X, Y_tensor).

    Shape inference: output = broadcast_shape(X.shape, Y.shape).
    For constant scalar Y, output shape = X.shape (no broadcasting needed).
    Maps to forge.op.Pow.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        exponent: Optional[float] = None,
    ) -> "PowNode":
        """
        Static factory method to create a PowNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo.
                    When exponent is a constant, contains only one entry (X).
                    When exponent is a tensor, contains two entries (X, Y).
            outputs: OrderedDict mapping output names to TensorInfo
            exponent: Constant scalar exponent value; None when Y is a tensor input.

        Returns:
            PowNode instance
        """
        attrs = {}
        if exponent is not None:
            attrs["exponent"] = float(exponent)
        return PowNode(
            name=name,
            op_type="Pow",
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            forge_op_name="Pow",
        )

    def convert_attrs_to_forge_attrs(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert exponent attribute for Forge code generation."""
        forge = attrs.copy()
        return forge

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate Pow operation using PyTorch.

        Handles both constant-exponent and tensor-exponent modes:
        - If attrs["exponent"] is set, uses it directly as a scalar.
        - Otherwise reads the second input tensor as the exponent.

        Args:
            input_tensors: Dictionary mapping input names to PyTorch tensors.

        Returns:
            Dictionary mapping output name to result tensor.

        Raises:
            ValueError: If neither a constant exponent nor a second input tensor is available.
        """
        x = input_tensors[self.input_names[0]]

        if "exponent" in self.attrs:
            # Constant exponent mode — Y was an initializer, collapsed to attr
            return {self.output_names[0]: torch.pow(x, self.attrs["exponent"])}

        if len(self.input_names) < 2:
            raise ValueError(
                f"PowNode '{self.name}': no constant exponent attribute and "
                "no second input tensor. Cannot evaluate."
            )
        # Tensor exponent mode — Y is a runtime activation
        y = input_tensors[self.input_names[1]]
        return {self.output_names[0]: torch.pow(x, y)}
```

**Why `BinaryBroadcastShape` for the upgraded version**: When Y is a tensor, the output shape is `broadcast_shape(X.shape, Y.shape)`. `BinaryBroadcastShape.infer_output_shapes()` correctly handles both the constant-Y case (Y scalar → no broadcast effect) and the tensor-Y case.

---

### Layer 2 — Shape Mixin (`operations/shape_mixins.py`)

The shape mixin provides compile-time output shape inference used by `resolve_output_shapes()` in `shape_finder.py`.

**Current state**: `PowNode` inherits `ElementwiseUnaryShape` — correct only for constant scalar Y.

**For the current single-input design** (constant exponent only), no change is needed:
```python
# ElementwiseUnaryShape correctly returns X.shape when Y is a constant scalar
class PowNode(ElementwiseUnaryShape, TIRNode): ...
```

**For the upgraded two-input design** (tensor exponent support), switch to `BinaryBroadcastShape`:
```python
# BinaryBroadcastShape correctly computes broadcast_shape(X.shape, Y.shape)
class PowNode(BinaryBroadcastShape, TIRNode): ...
```

**Shape behavior summary**:

| Y value | Input shapes | Output shape | Shape class |
|---------|-------------|--------------|-------------|
| Constant scalar (e.g., 2.0) | X: `[1, 128, 768]` | `[1, 128, 768]` | Either works |
| Tensor scalar shape `()` | X: `[1, 128, 768]`, Y: `()` | `[1, 128, 768]` | `BinaryBroadcastShape` |
| Tensor same shape | X: `[2, 3]`, Y: `[2, 3]` | `[2, 3]` | `BinaryBroadcastShape` |
| Tensor broadcastable | X: `[2, 3, 4]`, Y: `[3, 4]` | `[2, 3, 4]` | `BinaryBroadcastShape` |

**Registration in `op_shape_meta.py`** (already correct, no changes needed):
```python
"Pow": SHAPE_ONLY,   # output shape depends only on input shapes, not values
```

`SHAPE_ONLY` is correct because `Z = X^Y` always produces an output with the same shape as the broadcasted inputs — no value-dependent shape changes.

---

### Layer 3 — ONNX Converter (`frontends/onnx/converters/`)

The converter is the bridge between the ONNX graph representation and the TIR node. It needs to handle:
1. Extracting the exponent Y (either as a constant scalar or as a tensor input)
2. Handling the heterogeneous type constraint (output dtype = X dtype, not Y dtype)
3. Computing the output shape for opset 7+ (multidirectional broadcasting)

**Create a new file** `forge/forge/transpiler/frontends/onnx/converters/pow.py`:

```python
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Pow operation converter.

Handles all opset versions of the ONNX Pow operator:
  v1      : Limited broadcasting via broadcast/axis attributes, homogeneous types
  v7      : Multidirectional broadcasting, homogeneous types (float only)
  v12     : Heterogeneous types introduced (X:T, Y:T1, Z:T)
  v13     : bfloat16 added to T
  v15     : bfloat16 added to T1

Key design decision — two execution modes:
  Constant-exponent mode : Y is an ONNX initializer; extract its scalar value
                           and store as attrs["exponent"]. PowNode.eval() uses
                           torch.pow(X, scalar). Single input in TIR node.
  Tensor-exponent mode   : Y is a runtime activation tensor. PowNode.eval()
                           uses torch.pow(X, Y_tensor). Two inputs in TIR node.

Output type is always T (same as X), never T1 (Y type), per ONNX spec v12+.
"""
from typing import Any, Dict, List, Optional
from collections import OrderedDict

import numpy
from loguru import logger
from onnx import NodeProto, TensorProto

from forge.transpiler.core.types import TensorInfo
from forge.transpiler.operations.activation import PowNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts
from forge.transpiler.frontends.onnx.utils.constant_value_extractor import (
    resolve_constant_tensor_value,
)
from forge.transpiler.utils.binary_ops import compute_broadcasted_shape


class PowConverter(OnnxOpConverter):
    """
    Converter for ONNX Pow (element-wise power) operation.

    Supports all opset versions:
    - v1    : broadcast/axis attributes, float types only, homogeneous (T=T1)
    - v7+   : multidirectional broadcasting always on, no attributes
    - v12+  : heterogeneous types (X:T, Y:T1, Z:T); integer types for X/Z
    - v13+  : bfloat16 support for X/Z (type T)
    - v15+  : bfloat16 support for Y (type T1)

    Conversion strategy:
    1. Try to resolve Y as a constant scalar (initializer or computed constant).
       If successful, create a single-input PowNode with attrs["exponent"] = scalar.
    2. If Y is a runtime tensor, create a two-input PowNode with no exponent attr.

    Output dtype: always taken from X, not Y (heterogeneous type constraint).
    """

    @classmethod
    def convert(
        cls,
        node_proto: NodeProto,
        input_tensors: "OrderedDict[str, TensorInfo]",
        output_tensors: "OrderedDict[str, TensorInfo]",
        attrs: Dict[str, Any],
        node_index: int,
        graph_proto=None,
        opset: int = 1,
        tir_graph=None,
    ) -> List:
        """
        Convert ONNX Pow node to a TIR PowNode.

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dict of input TensorInfo (X and possibly Y)
            output_tensors: Dict of output TensorInfo (Z)
            attrs: Extracted attributes (broadcast, axis for v1; empty for v7+)
            node_index: Index of this node in the graph
            graph_proto: ONNX graph protocol buffer (for constant lookup)
            opset: ONNX opset version
            tir_graph: Partially-built TIRGraph (for constant/param lookup)

        Returns:
            List containing a single PowNode instance.

        Raises:
            ValueError: If inputs are invalid or shapes are incompatible.
        """
        node_name = node_proto.name or f"Pow_{node_index}"

        if len(node_proto.input) < 2:
            raise ValueError(
                f"Pow node '{node_name}': Expected 2 inputs (X, Y), "
                f"got {len(node_proto.input)}"
            )

        x_name = node_proto.input[0]
        y_name = node_proto.input[1]

        if x_name not in input_tensors:
            raise ValueError(
                f"Pow node '{node_name}': Input X '{x_name}' not found in input_tensors"
            )

        tensor_x = input_tensors[x_name]

        # Output dtype is always T (same as X), per ONNX heterogeneous type spec.
        # This is correct for all versions: v1-v7 (homogeneous T=T1) and v12+
        # (heterogeneous, but Z still has type T matching X).
        output_dtype = tensor_x.onnx_dtype
        output_name = list(output_tensors.keys())[0] if output_tensors else node_proto.output[0]

        # ── Strategy 1: Try to resolve Y as a compile-time constant scalar ───
        #
        # This is the common case in neural networks:
        #   - Normalization: X^2 (Y=2 initializer)
        #   - GELU approximation: X^3 (Y=3 initializer)
        #   - Softmax temperature scaling: X^(1/T) (Y=scalar initializer)
        #
        exponent_scalar: Optional[float] = None

        if tir_graph is not None and graph_proto is not None:
            success, values, _err = resolve_constant_tensor_value(y_name, tir_graph, graph_proto)
            if success and values is not None and len(values) == 1:
                exponent_scalar = float(values[0])
                logger.trace(
                    f"Pow node '{node_name}': Y resolved as constant scalar {exponent_scalar}"
                )

        if exponent_scalar is None and tir_graph is not None:
            # Also check tir_graph.constants and computed_constants directly
            for store in (tir_graph.constants, tir_graph.computed_constants):
                y_tensor = store.get(y_name)
                if y_tensor is not None and y_tensor.numel() == 1:
                    exponent_scalar = float(y_tensor.item())
                    logger.trace(
                        f"Pow node '{node_name}': Y found in constants as scalar {exponent_scalar}"
                    )
                    break

        # ── Compute output shape ──────────────────────────────────────────────
        if exponent_scalar is not None:
            # Constant scalar Y: output shape = X shape (no broadcasting)
            output_shape = tensor_x.shape
        elif y_name in input_tensors:
            tensor_y = input_tensors[y_name]
            if opset >= 7:
                output_shape = compute_broadcasted_shape(tensor_x.shape, tensor_y.shape)
                if output_shape is None:
                    raise ValueError(
                        f"Pow node '{node_name}': Shapes X={tensor_x.shape} and "
                        f"Y={tensor_y.shape} are not broadcastable"
                    )
            else:
                # v1: output shape matches X (Y is broadcasted to X)
                output_shape = tensor_x.shape
        else:
            output_shape = tensor_x.shape  # best-effort fallback

        # Update output TensorInfo with concrete shape and dtype
        output_tensors[output_name] = TensorInfo(
            name=output_name,
            shape=output_shape,
            onnx_dtype=output_dtype,
        )

        # ── Build TIR node ────────────────────────────────────────────────────
        if exponent_scalar is not None:
            # Constant-exponent mode: single input (X only), exponent as attribute
            x_info = input_tensors[x_name]
            input_dict = OrderedDict({x_name: x_info})
            output_dict = OrderedDict({output_name: output_tensors[output_name]})
            return [
                PowNode.create(
                    name=node_name,
                    inputs=input_dict,
                    outputs=output_dict,
                    exponent=exponent_scalar,
                )
            ]
        else:
            # Tensor-exponent mode: two inputs (X and Y), no exponent attribute
            if y_name not in input_tensors:
                raise ValueError(
                    f"Pow node '{node_name}': Y input '{y_name}' is neither a "
                    "constant nor a known activation tensor. Cannot convert."
                )
            input_dict, output_dict = build_input_output_dicts(
                node_proto, input_tensors, output_tensors
            )
            return [
                PowNode.create(
                    name=node_name,
                    inputs=input_dict,
                    outputs=output_dict,
                    exponent=None,
                )
            ]
```

---

### Layer 4 — Engine Registration (`frontends/onnx/engine.py`)

The engine maps ONNX op type strings to converter classes. Two changes are needed.

**Step 4a — Import the converter** (add to the existing import block):

```python
from forge.transpiler.frontends.onnx.converters.pow import PowConverter
```

**Step 4b — Register in `_build_op_converters()`** (add to the `_op_converters` dict):

```python
# ── Power operations ───────────────────────────────────────────────────
"Pow": PowConverter.get_converter(opset),
```

The complete context within `_build_op_converters()`:

```python
def _build_op_converters(self, opset: int) -> Dict[str, Callable]:
    return {
        # ... existing entries ...

        # ── Arithmetic / element-wise ──────────────────────────────────
        "Add":  BinaryOpConverter.get_converter(opset),
        "Sub":  BinaryOpConverter.get_converter(opset),
        "Mul":  BinaryOpConverter.get_converter(opset),
        "Div":  BinaryOpConverter.get_converter(opset),
        "Pow":  PowConverter.get_converter(opset),       # ← ADD THIS

        # ... rest of entries ...
    }
```

---

### Layer 5 — Test Case (`test/transpiler/ops/test_pow.py`)

Following the pattern used by all other op tests in `forge/test/transpiler/ops/`:

```python
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ONNX Pow operation.

Tests:
  - Basic element-wise power (constant scalar exponent)
  - Broadcasting cases (tensor exponent)
  - Opset version coverage (v7, v13, v15)
  - Integer base types (v12+)
  - Heterogeneous input types (v12+)
  - Edge cases (Y=0, Y=1, Y=0.5, negative base)
"""
import pytest
import numpy as np
import onnx

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from test.transpiler.test_utils import create_onnx_model, compare_tir_with_onnx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_pow_model(
    opset: int,
    x_shape,
    y_shape,
    x_dtype=onnx.TensorProto.FLOAT,
    y_dtype=onnx.TensorProto.FLOAT,
    y_as_initializer: bool = True,
    y_value=None,
    node_name: str = "pow_node",
):
    """
    Create a single-node ONNX Pow model.

    Args:
        opset: ONNX opset version
        x_shape: Shape of input X (base)
        y_shape: Shape of input Y (exponent)
        x_dtype: ONNX dtype for X (and output Z)
        y_dtype: ONNX dtype for Y
        y_as_initializer: If True, Y is an ONNX initializer (constant)
        y_value: Value for Y when y_as_initializer=True (numpy array or scalar)
        node_name: Name for the Pow node
    """
    initializers = {}
    if y_as_initializer and y_value is not None:
        y_arr = np.array(y_value, dtype=np.float32).reshape(y_shape) if y_shape else np.array(y_value, dtype=np.float32)
        initializers["exponent"] = y_arr

    input_names = ["X", "exponent" if y_as_initializer else "Y"]
    input_shapes = [x_shape, y_shape]
    input_dtypes = [x_dtype, y_dtype]

    # For initializers, ONNX excludes them from graph inputs
    if y_as_initializer:
        input_shapes = [x_shape]
        input_dtypes = [x_dtype]
        input_names_for_create = ["X"]
    else:
        input_names_for_create = ["X", "Y"]

    return create_onnx_model(
        op_type="Pow",
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=[x_shape],  # output shape = X shape (for scalar Y)
        output_dtypes=[x_dtype],
        opset_version=opset,
        node_name=node_name,
        input_names=["X", "exponent"] if y_as_initializer else ["X", "Y"],
        output_names=["Z"],
        initializers=initializers,
    )


# ---------------------------------------------------------------------------
# Basic Tests — Constant scalar exponent (most common case)
# ---------------------------------------------------------------------------


@pytest.mark.transpiler
class TestPowConstantExponent:
    """Test Pow with constant scalar exponent (initializer)."""

    @pytest.mark.parametrize("exponent", [2.0, 3.0, 0.5, -1.0])
    def test_pow_scalar_exponent_1d(self, exponent):
        """Test Pow with 1D input and various constant exponents."""
        model = _create_pow_model(
            opset=13,
            x_shape=(4,),
            y_shape=(),
            y_as_initializer=True,
            y_value=exponent,
        )
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        input_data = {"X": x_data}
        comparison = compare_tir_with_onnx(tir_graph, model, input_data)
        assert len(comparison["errors"]) == 0, f"Errors: {comparison['errors']}"
        assert comparison["matches"]["Z"], "Outputs should match"

    def test_pow_square_2d(self):
        """Test squaring a 2D tensor (Y=2, the most common Pow usage)."""
        model = _create_pow_model(opset=13, x_shape=(2, 3), y_shape=(), y_value=2.0)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"X": x_data})
        assert len(comparison["errors"]) == 0
        np.testing.assert_allclose(
            comparison["tir_outputs"]["Z"],
            np.array([[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]], dtype=np.float32),
            rtol=1e-5,
        )

    def test_pow_sqrt_via_pow(self):
        """Test square root via Pow with exponent=0.5."""
        model = _create_pow_model(opset=13, x_shape=(4,), y_shape=(), y_value=0.5)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x_data = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"X": x_data})
        assert len(comparison["errors"]) == 0
        np.testing.assert_allclose(
            comparison["tir_outputs"]["Z"],
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            rtol=1e-5,
        )

    def test_pow_3d_batch(self):
        """Test Pow with 3D batch tensor (typical model use case)."""
        model = _create_pow_model(opset=13, x_shape=(2, 3, 4), y_shape=(), y_value=2.0)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x_data = np.ones((2, 3, 4), dtype=np.float32) * 3.0
        comparison = compare_tir_with_onnx(tir_graph, model, {"X": x_data})
        assert len(comparison["errors"]) == 0
        np.testing.assert_allclose(
            comparison["tir_outputs"]["Z"],
            np.ones((2, 3, 4), dtype=np.float32) * 9.0,
            rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# Broadcasting Tests — Tensor exponent (runtime Y)
# ---------------------------------------------------------------------------


@pytest.mark.transpiler
class TestPowBroadcasting:
    """Test Pow with tensor exponent and broadcasting."""

    def test_pow_tensor_exponent_same_shape(self):
        """Test Pow with both inputs as tensors of the same shape."""
        model = _create_pow_model(
            opset=13,
            x_shape=(2, 3),
            y_shape=(2, 3),
            y_as_initializer=False,
        )
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x_data = np.array([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        y_data = np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"X": x_data, "Y": y_data})
        assert len(comparison["errors"]) == 0
        assert comparison["matches"]["Z"]

    def test_pow_broadcasting_1d_exponent(self):
        """Test Pow with 2D base and 1D exponent (suffix broadcasting)."""
        model = _create_pow_model(
            opset=13,
            x_shape=(2, 3),
            y_shape=(3,),
            y_as_initializer=False,
        )
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x_data = np.array([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        y_data = np.array([2.0, 3.0, 0.5], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"X": x_data, "Y": y_data})
        assert len(comparison["errors"]) == 0
        assert comparison["matches"]["Z"]


# ---------------------------------------------------------------------------
# Opset Version Tests
# ---------------------------------------------------------------------------


@pytest.mark.transpiler
class TestPowOpsetVersions:
    """Test Pow across all supported opset versions."""

    @pytest.mark.parametrize("opset", [7, 12, 13, 15])
    def test_pow_square_all_opsets(self, opset):
        """Test squaring across all major opset versions."""
        model = _create_pow_model(opset=opset, x_shape=(2, 3), y_shape=(), y_value=2.0)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"X": x_data})
        assert len(comparison["errors"]) == 0, f"Opset {opset}: {comparison['errors']}"
        assert comparison["matches"]["Z"], f"Opset {opset}: outputs should match"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


@pytest.mark.transpiler
class TestPowEdgeCases:
    """Test Pow edge cases."""

    def test_pow_exponent_zero(self):
        """Any non-zero base to power 0 should give 1."""
        model = _create_pow_model(opset=13, x_shape=(3,), y_shape=(), y_value=0.0)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x_data = np.array([2.0, 5.0, 100.0], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"X": x_data})
        assert len(comparison["errors"]) == 0
        np.testing.assert_allclose(
            comparison["tir_outputs"]["Z"],
            np.ones(3, dtype=np.float32),
            rtol=1e-5,
        )

    def test_pow_exponent_one(self):
        """Any base to power 1 should equal the base."""
        model = _create_pow_model(opset=13, x_shape=(3,), y_shape=(), y_value=1.0)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x_data = np.array([2.0, 5.0, 100.0], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"X": x_data})
        assert len(comparison["errors"]) == 0
        np.testing.assert_allclose(
            comparison["tir_outputs"]["Z"], x_data, rtol=1e-5
        )

    def test_pow_negative_exponent(self):
        """Test Pow with Y=-1 (equivalent to reciprocal)."""
        model = _create_pow_model(opset=13, x_shape=(3,), y_shape=(), y_value=-1.0)
        transpiler = ONNXToForgeTranspiler(validate_model=True)
        tir_graph = transpiler.transpile(model)

        x_data = np.array([2.0, 4.0, 8.0], dtype=np.float32)
        comparison = compare_tir_with_onnx(tir_graph, model, {"X": x_data})
        assert len(comparison["errors"]) == 0
        np.testing.assert_allclose(
            comparison["tir_outputs"]["Z"],
            np.array([0.5, 0.25, 0.125], dtype=np.float32),
            rtol=1e-5,
        )
```

---

## Integration Checklist

Use this checklist when adding Pow support to the transpiler:

| Step | File | Change | Status |
|------|------|--------|--------|
| 1 | `operations/activation.py` | Upgrade `PowNode` to support tensor Y input | Required |
| 2 | `operations/shape_mixins.py` | Change `PowNode` base class to `BinaryBroadcastShape` | Required (with tensor Y) |
| 3 | `frontends/onnx/converters/pow.py` | Create `PowConverter` class | Required |
| 4 | `frontends/onnx/engine.py` | Import `PowConverter` and register `"Pow"` | Required |
| 5 | `frontends/onnx/operations/op_shape_meta.py` | `"Pow": SHAPE_ONLY` | Already done |
| 6 | `test/transpiler/ops/test_pow.py` | Create test file | Required |

---

## Comparison with Add and Other Binary Ops

| Aspect | Add | Mul | Pow |
|--------|-----|-----|-----|
| **Operation** | `A + B` | `A * B` | `X ^ Y` |
| **Inputs** | 2 (A, B) — same type | 2 (A, B) — same type | 2 (X:T, Y:T1) — may differ |
| **Output type** | Same as inputs | Same as inputs | Same as X (type T), not Y |
| **Homogeneous?** | Yes (v1+) | Yes (v1+) | v1-v7 yes; v12+ no |
| **Broadcasting v1** | Limited (`broadcast=1`) | Limited (`broadcast=1`) | Limited (`broadcast=1`) |
| **Broadcasting v7+** | Multidirectional | Multidirectional | Multidirectional |
| **Shape inference** | `broadcast(A, B)` | `broadcast(A, B)` | `broadcast(X, Y)` |
| **Constant Y?** | Rarely | Rarely | Very commonly (scalar) |
| **TIR node type** | `BinaryBroadcastShape` | `BinaryBroadcastShape` | `ElementwiseUnaryShape` (const Y) or `BinaryBroadcastShape` (tensor Y) |
| **Forge op name** | `forge.op.Add` | `forge.op.Multiply` | `forge.op.Pow` |

---

## Differences from Other Operators

**Pow vs. Sqrt**:
- `Sqrt(X)` is equivalent to `Pow(X, 0.5)` — but they map to different TIR nodes
- `SqrtNode` is always unary (no second input); `PowNode` supports both modes
- For constant `Y=0.5`, the converter could optimize to `SqrtNode`, but this is not done in the current design (Pow always produces `PowNode`)

**Pow vs. Mul**:
- `Pow(X, 2)` is equivalent to `Mul(X, X)` for squaring — but Pow is cleaner for non-integer exponents
- Pow handles fractional exponents (`Y=0.333`) which Mul cannot

---

## References

- [ONNX Pow Operator Documentation](https://onnx.ai/onnx/operators/onnx__Pow.html)
- [ONNX Broadcasting Documentation](https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md)
- [PyTorch torch.pow](https://pytorch.org/docs/stable/generated/torch.pow.html)
- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
