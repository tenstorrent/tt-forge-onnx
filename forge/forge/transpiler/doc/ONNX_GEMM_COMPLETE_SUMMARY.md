# ONNX Gemm Operator - Complete Summary

## Overview

The **Gemm** (General Matrix Multiplication) operator performs general matrix multiplication following the BLAS Level 3 specification. It computes `Y = alpha * A' * B' + beta * C`, where `A'` and `B'` are optionally transposed versions of inputs `A` and `B`, and `C` is an optional bias tensor that can be broadcasted.

The operator is fundamental to linear algebra operations in neural networks, commonly used in fully connected layers, attention mechanisms, and various matrix transformations.

The operator has evolved across ONNX versions, transitioning from a required `broadcast` attribute to automatic unidirectional broadcasting, and making the `C` input optional for more flexibility.

## Version History

| Version | Since | Shape Inference | Function | Key Changes |
|---------|-------|----------------|----------|-------------|
| 1 | 1 | ❌ | ❌ | Initial version, `broadcast` attribute, C required, float types only |
| 6 | 6 | ✅ | ❌ | Removed `broadcast` attribute, automatic unidirectional broadcasting, shape inference enabled |
| 7 | 7 | ✅ | ❌ | Same as v6, improved documentation |
| 9 | 9 | ✅ | ❌ | **Major change:** C input becomes optional (between 2-3 inputs) |
| 11 | 11 | ✅ | ❌ | Extended type support (int types: int32, int64, uint32, uint64) |
| 13 | 13 | ✅ | ❌ | Extended type support (bfloat16) |

---

## Gemm - Version 1

**Since Version:** 1  
**Shape Inference:** ❌ False  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

General Matrix multiplication: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

Compute `Y = alpha * A * B + beta * C`, where input tensor A has dimension (M X K), input tensor B has dimension (K X N), input tensor C and output tensor Y have dimension (M X N). If attribute `broadcast` is non-zero, input tensor C will be broadcasted to match the dimension requirement. A will be transposed before doing the computation if attribute `transA` is non-zero, same for B and `transB`.

**Example:** Basic matrix multiplication with transposed A.

```
Input A: [3, 4]  (M=3, K=4) with transA=1 → A' becomes [4, 3]
Input B: [4, 5]  (K=4, N=5)
Input C: [3, 5]  (M=3, N=5)
Attributes: alpha=1.0, beta=1.0, transA=1, transB=0, broadcast=1
Output Y: [3, 5]  (M=3, N=5)
```

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `alpha` | FLOAT | ❌ | `1.0` | Scalar multiplier for the product of input tensors A * B, the default value is 1.0. |
| `beta` | FLOAT | ❌ | `1.0` | Scalar multiplier for input tensor C, the default value is 1.0. |
| `broadcast` | INT | ❌ | `0` | Whether C should be broadcasted |
| `transA` | INT | ❌ | `0` | Whether A should be transposed |
| `transB` | INT | ❌ | `0` | Whether B should be transposed |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `A` | T | Input tensor A |
| `B` | T | Input tensor B |
| `C` | T | Input tensor C, can be inplace. |

**Input Count:** 3 inputs (all required)

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output tensor. |

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input and output types to float tensors.

### Notes

- **Shape Inference:** Not supported in v1
- **Formula:** `Y = alpha * A' * B' + beta * C`
  - `A' = transpose(A)` if `transA != 0`, else `A`
  - `B' = transpose(B)` if `transB != 0`, else `B`
- **Shape Requirements:**
  - `A`: (M, K) if `transA=0`, or (K, M) if `transA != 0`
  - `B`: (K, N) if `transB=0`, or (N, K) if `transB != 0`
  - `C`: (M, N) or broadcastable to (M, N) if `broadcast != 0`
  - `Y`: (M, N)
- **Broadcast Attribute:** When `broadcast=1`, tensor C can have smaller dimensions and will be broadcasted to match (M, N)
- **Inplace Operation:** Input C can be modified in-place (output Y can alias input C)

### Example: Basic Matrix Multiplication (v1)

```python
# ONNX Model (v1)
# Input A: [3, 4]  (M=3, K=4)
# Input B: [4, 5]  (K=4, N=5)
# Input C: [3, 5]  (M=3, N=5)
# Attributes: alpha=1.0, beta=1.0, transA=0, transB=0, broadcast=0
# Output Y: [3, 5]  (M=3, N=5)
# Formula: Y = 1.0 * A * B + 1.0 * C

# PyTorch Equivalent
import torch
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.randn(3, 5)
Y = torch.matmul(A, B) + C  # alpha=1.0, beta=1.0
```

### Example: With Transposed A (v1)

```python
# ONNX Model (v1)
# Input A: [4, 3]  (K=4, M=3) with transA=1 → A' becomes [3, 4]
# Input B: [4, 5]  (K=4, N=5)
# Input C: [3, 5]  (M=3, N=5)
# Attributes: alpha=2.0, beta=0.5, transA=1, transB=0, broadcast=0
# Output Y: [3, 5]  (M=3, N=5)
# Formula: Y = 2.0 * A^T * B + 0.5 * C

# PyTorch Equivalent
A = torch.randn(4, 3)
B = torch.randn(4, 5)
C = torch.randn(3, 5)
Y = 2.0 * torch.matmul(A.t(), B) + 0.5 * C
```

### Example: With Broadcasting (v1)

```python
# ONNX Model (v1)
# Input A: [3, 4]  (M=3, K=4)
# Input B: [4, 5]  (K=4, N=5)
# Input C: [5]  (scalar-like, will be broadcasted to [3, 5])
# Attributes: alpha=1.0, beta=1.0, transA=0, transB=0, broadcast=1
# Output Y: [3, 5]  (M=3, N=5)
# Formula: Y = A * B + broadcast(C)

# PyTorch Equivalent
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.randn(5)  # Will be broadcasted
Y = torch.matmul(A, B) + C.unsqueeze(0)  # Broadcast C to [1, 5] then to [3, 5]
```

---

## Gemm - Version 6

**Since Version:** 6  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

General Matrix multiplication: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute `Y = alpha * A' * B' + beta * C`, where input tensor A has shape (M, K) or (K, M), input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N), and output tensor Y has shape (M, N). A will be transposed before doing the computation if attribute `transA` is non-zero, same for B and `transB`. This operator supports unidirectional broadcasting (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check Broadcasting in ONNX.

**Example:** Matrix multiplication with automatic broadcasting.

```
Input A: [3, 4]  (M=3, K=4)
Input B: [4, 5]  (K=4, N=5)
Input C: [1, 5]  (will be broadcasted to [3, 5])
Attributes: alpha=1.0, beta=1.0, transA=0, transB=0
Output Y: [3, 5]  (M=3, N=5)
```

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `alpha` | FLOAT | ❌ | `1.0` | Scalar multiplier for the product of input tensors A * B. |
| `beta` | FLOAT | ❌ | `1.0` | Scalar multiplier for input tensor C. |
| `transA` | INT | ❌ | `0` | Whether A should be transposed |
| `transB` | INT | ❌ | `0` | Whether B should be transposed |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `A` | T | Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero. |
| `B` | T | Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero. |
| `C` | T | Input tensor C. The shape of C should be unidirectional broadcastable to (M, N). |

**Input Count:** 3 inputs (all required)

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output tensor of shape (M, N). |

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input and output types to float tensors.

### Notes

- **Shape Inference:** Supported, allowing automatic shape propagation
- **Removed Attribute:** `broadcast` attribute removed - broadcasting is now automatic and unidirectional
- **Unidirectional Broadcasting:** Tensor C should be unidirectional broadcastable to (M, N). This means C can have fewer dimensions and will be automatically broadcasted along the appropriate axes.
- **Formula:** Same as v1: `Y = alpha * A' * B' + beta * C`
- **Shape Requirements:** Same as v1

### Example: Automatic Broadcasting (v6)

```python
# ONNX Model (v6)
# Input A: [3, 4]  (M=3, K=4)
# Input B: [4, 5]  (K=4, N=5)
# Input C: [1, 5]  (will be automatically broadcasted to [3, 5])
# Attributes: alpha=1.0, beta=1.0, transA=0, transB=0
# Output Y: [3, 5]  (M=3, N=5)
# Note: No broadcast attribute needed - automatic unidirectional broadcasting

# PyTorch Equivalent
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.randn(1, 5)  # Will be automatically broadcasted
Y = torch.matmul(A, B) + C  # PyTorch automatically broadcasts
```

### Example: Scalar-like C (v6)

```python
# ONNX Model (v6)
# Input A: [3, 4]  (M=3, K=4)
# Input B: [4, 5]  (K=4, N=5)
# Input C: [5]  (1D tensor, will be broadcasted to [3, 5])
# Attributes: alpha=1.0, beta=1.0, transA=0, transB=0
# Output Y: [3, 5]  (M=3, N=5)

# PyTorch Equivalent
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.randn(5)  # 1D tensor
Y = torch.matmul(A, B) + C  # PyTorch broadcasts C to [3, 5]
```

---

## Gemm - Version 7

**Since Version:** 7  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

General Matrix multiplication: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute `Y = alpha * A' * B' + beta * C`, where input tensor A has shape (M, K) or (K, M), input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N), and output tensor Y has shape (M, N). A will be transposed before doing the computation if attribute `transA` is non-zero, same for B and `transB`. This operator supports unidirectional broadcasting (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check Broadcasting in ONNX.

**Example:** Same as v6, with improved documentation.

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `alpha` | FLOAT | ❌ | `1.0` | Scalar multiplier for the product of input tensors A * B. |
| `beta` | FLOAT | ❌ | `1.0` | Scalar multiplier for input tensor C. |
| `transA` | INT | ❌ | `0` | Whether A should be transposed |
| `transB` | INT | ❌ | `0` | Whether B should be transposed |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `A` | T | Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero. |
| `B` | T | Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero. |
| `C` | T | Input tensor C. The shape of C should be unidirectional broadcastable to (M, N). |

**Input Count:** 3 inputs (all required)

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output tensor of shape (M, N). |

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input and output types to float tensors.

### Notes

- **Same as v6:** No functional changes from v6, only improved documentation
- **Shape Inference:** Supported
- **Unidirectional Broadcasting:** Same as v6

### Example: Same as v6

All examples from v6 apply to v7 as well.

---

## Gemm - Version 9

**Since Version:** 9  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

General Matrix multiplication: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute `Y = alpha * A' * B' + beta * C`, where input tensor A has shape (M, K) or (K, M), input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N), and output tensor Y has shape (M, N). A will be transposed before doing the computation if attribute `transA` is non-zero, same for B and `transB`. This operator supports unidirectional broadcasting (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check Broadcasting in ONNX. This operator has optional inputs/outputs. See ONNX IR for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

**Example:** Matrix multiplication without C input.

```
Input A: [3, 4]  (M=3, K=4)
Input B: [4, 5]  (K=4, N=5)
Input C: (omitted or empty string)
Attributes: alpha=1.0, beta=1.0, transA=0, transB=0
Output Y: [3, 5]  (M=3, N=5)
Formula: Y = alpha * A * B (C treated as scalar 0)
```

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `alpha` | FLOAT | ❌ | `1.0` | Scalar multiplier for the product of input tensors A * B. |
| `beta` | FLOAT | ❌ | `1.0` | Scalar multiplier for input tensor C. |
| `transA` | INT | ❌ | `0` | Whether A should be transposed |
| `transB` | INT | ❌ | `0` | Whether B should be transposed |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `A` | T | Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero. |
| `B` | T | Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero. |
| `C` | T (optional) | Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N). |

**Input Count:** Between 2 and 3 inputs

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output tensor of shape (M, N). |

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`

**Total:** 3 types

**Description:** Constrain input and output types to float tensors.

### Notes

- **Major Change:** Input C is now optional
- **Optional Input Handling:** If C is not provided (omitted or empty string), the computation is done as if C is a scalar 0, meaning `Y = alpha * A' * B'`
- **Shape Inference:** Supported
- **Unidirectional Broadcasting:** Same as v6/v7

### Example: Without C Input (v9)

```python
# ONNX Model (v9)
# Input A: [3, 4]  (M=3, K=4)
# Input B: [4, 5]  (K=4, N=5)
# Input C: (omitted)
# Attributes: alpha=1.0, beta=1.0, transA=0, transB=0
# Output Y: [3, 5]  (M=3, N=5)
# Formula: Y = alpha * A * B (C treated as 0)

# PyTorch Equivalent
A = torch.randn(3, 4)
B = torch.randn(4, 5)
Y = torch.matmul(A, B)  # No C input, equivalent to C=0
```

### Example: With C Input (v9)

```python
# ONNX Model (v9)
# Input A: [3, 4]  (M=3, K=4)
# Input B: [4, 5]  (K=4, N=5)
# Input C: [3, 5]  (M=3, N=5)
# Attributes: alpha=1.0, beta=1.0, transA=0, transB=0
# Output Y: [3, 5]  (M=3, N=5)
# Formula: Y = alpha * A * B + beta * C

# PyTorch Equivalent
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.randn(3, 5)
Y = torch.matmul(A, B) + C
```

---

## Gemm - Version 11

**Since Version:** 11  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

General Matrix multiplication: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute `Y = alpha * A' * B' + beta * C`, where input tensor A has shape (M, K) or (K, M), input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N), and output tensor Y has shape (M, N). A will be transposed before doing the computation if attribute `transA` is non-zero, same for B and `transB`. This operator supports unidirectional broadcasting (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check Broadcasting in ONNX. This operator has optional inputs/outputs. See ONNX IR for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

**Example:** Matrix multiplication with integer types.

```
Input A: [3, 4]  (int32, M=3, K=4)
Input B: [4, 5]  (int32, K=4, N=5)
Input C: [3, 5]  (int32, M=3, N=5)
Attributes: alpha=1.0, beta=1.0, transA=0, transB=0
Output Y: [3, 5]  (int32, M=3, N=5)
```

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `alpha` | FLOAT | ❌ | `1.0` | Scalar multiplier for the product of input tensors A * B. |
| `beta` | FLOAT | ❌ | `1.0` | Scalar multiplier for input tensor C. |
| `transA` | INT | ❌ | `0` | Whether A should be transposed |
| `transB` | INT | ❌ | `0` | Whether B should be transposed |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `A` | T | Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero. |
| `B` | T | Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero. |
| `C` | T (optional) | Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N). |

**Input Count:** Between 2 and 3 inputs

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output tensor of shape (M, N). |

### Type Constraints

**T** in:
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`
- `tensor(int32)`
- `tensor(int64)`
- `tensor(uint32)`
- `tensor(uint64)`

**Total:** 7 types

**Description:** Constrain input and output types to float/int tensors.

### Notes

- **Extended Type Support:** Now supports integer types (int32, int64, uint32, uint64) in addition to float types
- **Optional Input:** Same as v9 - C is optional
- **Shape Inference:** Supported
- **Unidirectional Broadcasting:** Same as previous versions

### Example: Integer Matrix Multiplication (v11)

```python
# ONNX Model (v11)
# Input A: [3, 4]  (int32, M=3, K=4)
# Input B: [4, 5]  (int32, K=4, N=5)
# Input C: [3, 5]  (int32, M=3, N=5)
# Attributes: alpha=1.0, beta=1.0, transA=0, transB=0
# Output Y: [3, 5]  (int32, M=3, N=5)

# PyTorch Equivalent
A = torch.randint(0, 10, (3, 4), dtype=torch.int32)
B = torch.randint(0, 10, (4, 5), dtype=torch.int32)
C = torch.randint(0, 10, (3, 5), dtype=torch.int32)
Y = torch.matmul(A, B) + C  # Note: alpha and beta are still float, but inputs can be int
```

### Example: Float Matrix Multiplication (v11)

```python
# ONNX Model (v11)
# Input A: [3, 4]  (float, M=3, K=4)
# Input B: [4, 5]  (float, K=4, N=5)
# Input C: (omitted)
# Attributes: alpha=2.0, beta=0.5, transA=0, transB=0
# Output Y: [3, 5]  (float, M=3, N=5)
# Formula: Y = 2.0 * A * B (C treated as 0)

# PyTorch Equivalent
A = torch.randn(3, 4)
B = torch.randn(4, 5)
Y = 2.0 * torch.matmul(A, B)  # alpha=2.0, C=0
```

---

## Gemm - Version 13

**Since Version:** 13  
**Shape Inference:** ✅ True  
**Function:** ❌ False  
**Support Level:** COMMON

### Summary

General Matrix multiplication: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute `Y = alpha * A' * B' + beta * C`, where input tensor A has shape (M, K) or (K, M), input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N), and output tensor Y has shape (M, N). A will be transposed before doing the computation if attribute `transA` is non-zero, same for B and `transB`. This operator supports unidirectional broadcasting (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check Broadcasting in ONNX. This operator has optional inputs/outputs. See ONNX IR for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

**Example:** Matrix multiplication with bfloat16 precision.

```
Input A: [3, 4]  (bfloat16, M=3, K=4)
Input B: [4, 5]  (bfloat16, K=4, N=5)
Input C: [3, 5]  (bfloat16, M=3, N=5)
Attributes: alpha=1.0, beta=1.0, transA=0, transB=0
Output Y: [3, 5]  (bfloat16, M=3, N=5)
```

### Attributes

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `alpha` | FLOAT | ❌ | `1.0` | Scalar multiplier for the product of input tensors A * B. |
| `beta` | FLOAT | ❌ | `1.0` | Scalar multiplier for input tensor C. |
| `transA` | INT | ❌ | `0` | Whether A should be transposed |
| `transB` | INT | ❌ | `0` | Whether B should be transposed |

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `A` | T | Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero. |
| `B` | T | Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero. |
| `C` | T (optional) | Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N). |

**Input Count:** Between 2 and 3 inputs

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `Y` | T | Output tensor of shape (M, N). |

### Type Constraints

**T** in:
- `tensor(bfloat16)`
- `tensor(double)`
- `tensor(float)`
- `tensor(float16)`
- `tensor(int32)`
- `tensor(int64)`
- `tensor(uint32)`
- `tensor(uint64)`

**Total:** 8 types

**Description:** Constrain input and output types to float/int tensors.

### Notes

- **Extended Type Support:** Added `bfloat16` support for improved memory efficiency and performance
- **Optional Input:** Same as v9/v11 - C is optional
- **Shape Inference:** Supported
- **Unidirectional Broadcasting:** Same as previous versions

### Example: BFloat16 Matrix Multiplication (v13)

```python
# ONNX Model (v13)
# Input A: [3, 4]  (bfloat16, M=3, K=4)
# Input B: [4, 5]  (bfloat16, K=4, N=5)
# Input C: [3, 5]  (bfloat16, M=3, N=5)
# Attributes: alpha=1.0, beta=1.0, transA=0, transB=0
# Output Y: [3, 5]  (bfloat16, M=3, N=5)

# PyTorch Equivalent
A = torch.randn(3, 4, dtype=torch.bfloat16)
B = torch.randn(4, 5, dtype=torch.bfloat16)
C = torch.randn(3, 5, dtype=torch.bfloat16)
Y = torch.matmul(A, B) + C
```

### Example: With Transposed B (v13)

```python
# ONNX Model (v13)
# Input A: [3, 4]  (float, M=3, K=4)
# Input B: [5, 4]  (N=5, K=4) with transB=1 → B' becomes [4, 5]
# Input C: (omitted)
# Attributes: alpha=1.0, beta=1.0, transA=0, transB=1
# Output Y: [3, 5]  (M=3, N=5)
# Formula: Y = A * B^T

# PyTorch Equivalent
A = torch.randn(3, 4)
B = torch.randn(5, 4)
Y = torch.matmul(A, B.t())  # transB=1 means transpose B
```

---

## Version Comparison Examples

### Comparison: v1 vs v6 - Broadcasting

**v1 Approach:**
```python
# ONNX v1
# Attributes: broadcast=1
# Explicit broadcast attribute required

# PyTorch
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.randn(1, 5)
# Manual broadcasting or use broadcast attribute
if broadcast:
    C = C.expand(3, 5)
Y = torch.matmul(A, B) + C
```

**v6 Approach:**
```python
# ONNX v6
# No broadcast attribute - automatic unidirectional broadcasting
# More intuitive and flexible

# PyTorch
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.randn(1, 5)  # Automatically broadcasted
Y = torch.matmul(A, B) + C  # PyTorch handles broadcasting automatically
```

**Key Difference:** v6 removes the need for explicit `broadcast` attribute and uses automatic unidirectional broadcasting, making the operator more intuitive and aligned with modern tensor operations.

### Comparison: v7 vs v9 - Optional C Input

**v7 Approach:**
```python
# ONNX v7
# Input C: Required (3 inputs always)
# Must provide C even if it's zeros

# PyTorch
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.zeros(3, 5)  # Must provide C
Y = torch.matmul(A, B) + C
```

**v9 Approach:**
```python
# ONNX v9
# Input C: Optional (2-3 inputs)
# Can omit C for simpler models

# PyTorch
A = torch.randn(3, 4)
B = torch.randn(4, 5)
# C can be omitted
Y = torch.matmul(A, B)  # Equivalent to C=0
```

**Key Difference:** v9 makes C optional, allowing models to omit the bias term when not needed, reducing model complexity and memory usage.

### Comparison: v11 vs v13 - Type Support

**v11 Types:**
- Input/Output: `double`, `float`, `float16`, `int32`, `int64`, `uint32`, `uint64`

**v13 Types:**
- Input/Output: `bfloat16`, `double`, `float`, `float16`, `int32`, `int64`, `uint32`, `uint64`

**Key Difference:** v13 adds `bfloat16` support, which provides a good balance between memory efficiency (16-bit) and numerical range (similar to float32), making it useful for training and inference on modern hardware.

---

## PyTorch Mapping

### Direct Mapping

The ONNX Gemm operator maps directly to PyTorch's matrix multiplication operations:

```python
# ONNX Gemm
# Y = alpha * A' * B' + beta * C
# where A' = transpose(A) if transA else A
#       B' = transpose(B) if transB else B

# PyTorch Equivalent
A_prime = A.t() if transA else A
B_prime = B.t() if transB else B
Y = alpha * torch.matmul(A_prime, B_prime) + beta * C
```

### Using torch.nn.Linear

For the common case where `transA=0`, `transB=0`, `alpha=1.0`, `beta=1.0`, and C is a bias vector, Gemm can be mapped to `torch.nn.Linear`:

```python
# ONNX Gemm
# A: [batch, in_features]
# B: [in_features, out_features]  (weight matrix)
# C: [out_features]  (bias vector, broadcasted)
# Attributes: transA=0, transB=0, alpha=1.0, beta=1.0

# PyTorch Equivalent
linear = nn.Linear(
    in_features,      # From A.shape[1] or B.shape[0]
    out_features,     # From B.shape[1]
    bias=True         # If C input is present
)
# Set weight and bias
linear.weight.data = B.t()  # Note: Linear stores weight as [out_features, in_features]
linear.bias.data = C
output = linear(A)
```

### Mapping Table

| ONNX | PyTorch | Notes |
|------|---------|-------|
| `A` | `input` | Input tensor A |
| `B` | `weight` | Weight matrix (may need transpose) |
| `C` | `bias` | Bias tensor (optional, may need broadcasting) |
| `alpha` | Scalar multiplication | Applied to `A * B` product |
| `beta` | Scalar multiplication | Applied to `C` |
| `transA` | `A.t()` or `A` | Transpose A if non-zero |
| `transB` | `B.t()` or `B` | Transpose B if non-zero |
| `Y` | `output` | Output tensor |

### Key Differences

1. **Weight Storage:**
   - ONNX Gemm: B is `[K, N]` (or `[N, K]` if transposed)
   - PyTorch Linear: Weight is `[out_features, in_features]`
   - Mapping: `linear.weight = B.t()` if `transB=0`, or `linear.weight = B` if `transB=1`

2. **Bias Broadcasting:**
   - ONNX: C can be broadcasted to `[M, N]`
   - PyTorch: Bias is typically `[out_features]` and is automatically broadcasted
   - Mapping: Extract bias from C (e.g., `C[0]` if C is `[1, N]`)

3. **Alpha and Beta:**
   - ONNX: Attributes `alpha` and `beta` are scalars
   - PyTorch: No direct equivalent in `nn.Linear`, must use manual computation
   - Mapping: Apply scaling manually: `Y = alpha * linear(A) + beta * C`

4. **Type Support:**
   - ONNX v1: Float types only
   - ONNX v11+: Float + Integer types
   - ONNX v13+: Float + Integer + bfloat16 types
   - PyTorch: Supports all numeric types

---

## Implementation Considerations

### Forge TIR Mapping

The Forge implementation maps ONNX Gemm to TIR operations. The typical mapping involves:

1. **Transpose Operations:** If `transA` or `transB` are non-zero, apply transpose operations
2. **Matrix Multiplication:** Use TIR matrix multiplication operation
3. **Scalar Multiplication:** Apply `alpha` to the matrix product and `beta` to C
4. **Broadcasting:** Handle unidirectional broadcasting of C to match the output shape
5. **Addition:** Add the scaled matrix product and scaled C

### Handling Different Versions

1. **v1:** Extract `broadcast` attribute and handle broadcasting explicitly
2. **v6-v7:** Automatic unidirectional broadcasting (no `broadcast` attribute)
3. **v9+:** Handle optional C input (check if C is provided or use scalar 0)
4. **v11+:** Support integer types in addition to float types
5. **v13+:** Support bfloat16 type

### Shape Inference

The output shape `(M, N)` can be inferred from:
- `M`: From `A.shape[0]` if `transA=0`, or `A.shape[1]` if `transA != 0`
- `N`: From `B.shape[1]` if `transB=0`, or `B.shape[0]` if `transB != 0`
- `K`: Must match between A and B (from `A.shape[1]` and `B.shape[0]` if not transposed, or vice versa)

### Broadcasting Rules

For tensor C to be unidirectionally broadcastable to `(M, N)`:
- C can have shape `(M, N)` - exact match
- C can have shape `(1, N)` - broadcast along first dimension
- C can have shape `(M, 1)` - broadcast along second dimension
- C can have shape `(1, 1)` - broadcast along both dimensions
- C can have shape `(N,)` - broadcast to `(1, N)` then to `(M, N)`
- C can be a scalar - broadcast to `(M, N)`

### Optimization Opportunities

1. **Identity Operations:** If `alpha=1.0` and `beta=0.0` and C is omitted, the operation simplifies to just matrix multiplication
2. **Zero C:** If `beta=0.0` or C is omitted, the addition step can be skipped
3. **Unit Alpha:** If `alpha=1.0`, the scalar multiplication can be skipped
4. **Fused Operations:** Combine transpose, matmul, and addition into a single fused operation when possible

---

## References

- [ONNX Gemm Operator Documentation](https://onnx.ai/onnx/operators/onnx__Gemm.html)
- [BLAS Level 3 Specification](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3)
- [PyTorch Linear Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [PyTorch Matrix Multiplication Documentation](https://pytorch.org/docs/stable/generated/torch.matmul.html)
- [ONNX Broadcasting Specification](https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md)

