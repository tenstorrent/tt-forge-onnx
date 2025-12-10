# PyTorch Pad Operation - Complete Explanation with Examples

## Overview

`torch.nn.functional.pad` pads a tensor with specified values. The padding can be applied to different dimensions with different amounts and modes.

## Function Signature

```python
torch.nn.functional.pad(input, pad, mode='constant', value=0.0)
```

### Parameters:
- **input**: Input tensor of any shape
- **pad**: Tuple of padding sizes. Format: `(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back, ...)`
  - For 1D: `(pad_left, pad_right)`
  - For 2D: `(pad_left, pad_right, pad_top, pad_bottom)`
  - For 3D: `(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)`
  - The order is: **last dimension first, then second-to-last, etc.**
- **mode**: Padding mode (`'constant'`, `'reflect'`, `'replicate'`, `'circular'`)
- **value**: Fill value for `'constant'` mode (default: 0.0)

---

## Padding Modes

### 1. `'constant'` Mode (Default)
Pads with a constant value.

**Example:**
```python
import torch
import torch.nn.functional as F

# 1D tensor
x = torch.tensor([1, 2, 3])
print(f"Input: {x}, shape: {x.shape}")  # Input: tensor([1, 2, 3]), shape: torch.Size([3])

# Pad left=1, right=2 with value=0
padded = F.pad(x, (1, 2), mode='constant', value=0)
print(f"Output: {padded}, shape: {padded.shape}")  # Output: tensor([0, 1, 2, 3, 0, 0]), shape: torch.Size([6])
# Input shape: [3] → Output shape: [3 + 1 + 2] = [6]

# 2D tensor
x = torch.tensor([[1, 2], [3, 4]])
print(f"Input shape: {x.shape}")  # Input shape: torch.Size([2, 3])

# Pad: left=1, right=1, top=1, bottom=1 with value=0
padded = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)
print(f"Output shape: {padded.shape}")  # Output shape: torch.Size([4, 4])
print(f"Output:\n{padded}")
# Input shape: [2, 3] → Output shape: [2+1+1, 3+1+1] = [4, 4]
# Result:
# tensor([[0, 0, 0, 0],
#         [0, 1, 2, 0],
#         [0, 3, 4, 0],
#         [0, 0, 0, 0]])

# With custom value
padded = F.pad(x, (1, 1, 1, 1), mode='constant', value=5)
print(f"Output shape: {padded.shape}")  # Output shape: torch.Size([4, 4])
print(f"Output:\n{padded}")
# Result:
# tensor([[5, 5, 5, 5],
#         [5, 1, 2, 5],
#         [5, 3, 4, 5],
#         [5, 5, 5, 5]])
```

### 2. `'reflect'` Mode
Pads by reflecting the tensor values at the boundaries (mirroring).

**Example:**
```python
# 1D tensor
x = torch.tensor([1, 2, 3, 4])
print(f"Input: {x}, shape: {x.shape}")  # Input: tensor([1, 2, 3, 4]), shape: torch.Size([4])

# Pad left=2, right=2
padded = F.pad(x, (2, 2), mode='reflect')
print(f"Output: {padded}, shape: {padded.shape}")  # Output: tensor([3, 2, 1, 2, 3, 4, 3, 2]), shape: torch.Size([8])
# Input shape: [4] → Output shape: [4 + 2 + 2] = [8]
# Explanation: Reflects at boundaries
# Original: [1, 2, 3, 4]
# Left pad (2): Reflect [1, 2] -> [2, 1] but actually [3, 2] (mirrors from edge)
# Right pad (2): Reflect [3, 4] -> [4, 3] but actually [3, 2] (mirrors from edge)

# 2D tensor
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
print(f"Input shape: {x.shape}")  # Input shape: torch.Size([3, 3])

# Pad: left=1, right=1, top=1, bottom=1
padded = F.pad(x, (1, 1, 1, 1), mode='reflect')
print(f"Output shape: {padded.shape}")  # Output shape: torch.Size([5, 5])
print(f"Output:\n{padded}")
# Input shape: [3, 3] → Output shape: [3+1+1, 3+1+1] = [5, 5]
# Result:
# tensor([[5, 4, 5, 6, 5],
#         [2, 1, 2, 3, 2],
#         [5, 4, 5, 6, 5],
#         [8, 7, 8, 9, 8],
#         [5, 4, 5, 6, 5]])
```

### 3. `'replicate'` Mode
Pads by replicating the edge values.

**Example:**
```python
# 1D tensor
x = torch.tensor([1, 2, 3, 4])
print(f"Input: {x}, shape: {x.shape}")  # Input: tensor([1, 2, 3, 4]), shape: torch.Size([4])

# Pad left=2, right=2
padded = F.pad(x, (2, 2), mode='replicate')
print(f"Output: {padded}, shape: {padded.shape}")  # Output: tensor([1, 1, 1, 2, 3, 4, 4, 4]), shape: torch.Size([8])
# Input shape: [4] → Output shape: [4 + 2 + 2] = [8]
# Left edge (1) is replicated, right edge (4) is replicated

# 2D tensor
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
print(f"Input shape: {x.shape}")  # Input shape: torch.Size([3, 3])

# Pad: left=1, right=1, top=1, bottom=1
padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
print(f"Output shape: {padded.shape}")  # Output shape: torch.Size([5, 5])
print(f"Output:\n{padded}")
# Input shape: [3, 3] → Output shape: [3+1+1, 3+1+1] = [5, 5]
# Result:
# tensor([[1, 1, 2, 3, 3],
#         [1, 1, 2, 3, 3],
#         [4, 4, 5, 6, 6],
#         [7, 7, 8, 9, 9],
#         [7, 7, 8, 9, 9]])
```

### 4. `'circular'` Mode
Pads by wrapping around (treating the tensor as periodic).

**Example:**
```python
# 1D tensor
x = torch.tensor([1, 2, 3, 4])
print(f"Input: {x}, shape: {x.shape}")  # Input: tensor([1, 2, 3, 4]), shape: torch.Size([4])

# Pad left=2, right=2
padded = F.pad(x, (2, 2), mode='circular')
print(f"Output: {padded}, shape: {padded.shape}")  # Output: tensor([3, 4, 1, 2, 3, 4, 1, 2]), shape: torch.Size([8])
# Input shape: [4] → Output shape: [4 + 2 + 2] = [8]
# Wraps around: ...4, 1, 2, 3, 4, 1, 2...

# 2D tensor
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
print(f"Input shape: {x.shape}")  # Input shape: torch.Size([3, 3])

# Pad: left=1, right=1, top=1, bottom=1
padded = F.pad(x, (1, 1, 1, 1), mode='circular')
print(f"Output shape: {padded.shape}")  # Output shape: torch.Size([5, 5])
print(f"Output:\n{padded}")
# Input shape: [3, 3] → Output shape: [3+1+1, 3+1+1] = [5, 5]
# Wraps around both dimensions
```

---

## Examples for Different Dimensions

### 1D Tensor (Shape: `[N]`)

```python
import torch
import torch.nn.functional as F

x = torch.tensor([1, 2, 3])
print(f"Input: {x}, shape: {x.shape}")
# Input: tensor([1, 2, 3]), shape: torch.Size([3])

# Pad left=2, right=1
padded = F.pad(x, (2, 1), mode='constant', value=0)
print(f"Output: {padded}, shape: {padded.shape}")
# Output: tensor([0, 0, 1, 2, 3, 0]), shape: torch.Size([6])
# Input shape: [3] → Output shape: [3 + 2 + 1] = [6]

# Different padding on each side
padded = F.pad(x, (3, 1), mode='constant', value=5)
print(f"Output: {padded}, shape: {padded.shape}")
# Output: tensor([5, 5, 5, 1, 2, 3, 5]), shape: torch.Size([7])
# Input shape: [3] → Output shape: [3 + 3 + 1] = [7]
```

**Pad tuple format for 1D:** `(pad_left, pad_right)`

---

### 2D Tensor (Shape: `[H, W]` or `[N, C]`)

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(f"Input shape: {x.shape}")  # Input shape: torch.Size([2, 3])
print(f"Input:\n{x}")
# Input:
# tensor([[1, 2, 3],
#         [4, 5, 6]])

# Pad: left=1, right=2, top=1, bottom=1
# Format: (pad_left, pad_right, pad_top, pad_bottom)
padded = F.pad(x, (1, 2, 1, 1), mode='constant', value=0)
print(f"Output shape: {padded.shape}")  # Output shape: torch.Size([4, 6])
print(f"Output:\n{padded}")
# Input shape: [2, 3] → Output shape: [2+1+1, 3+1+2] = [4, 6]
# Output:
# tensor([[0, 0, 0, 0, 0, 0],
#         [0, 1, 2, 3, 0, 0],
#         [0, 4, 5, 6, 0, 0],
#         [0, 0, 0, 0, 0, 0]])

# Asymmetric padding
padded = F.pad(x, (2, 1, 0, 2), mode='constant', value=0)
# Left=2, Right=1, Top=0, Bottom=2
print(f"Output shape: {padded.shape}")  # Output shape: torch.Size([4, 6])
print(f"Output:\n{padded}")
# Input shape: [2, 3] → Output shape: [2+0+2, 3+2+1] = [4, 6]
# Output:
# tensor([[0, 0, 1, 2, 3, 0],
#         [0, 0, 4, 5, 6, 0],
#         [0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0]])
```

**Pad tuple format for 2D:** `(pad_left, pad_right, pad_top, pad_bottom)`
- **Order**: Last dimension (width) first, then second-to-last (height)

---

### 3D Tensor (Shape: `[D, H, W]` or `[C, H, W]`)

```python
x = torch.tensor([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]])
print(f"Input shape: {x.shape}")  # Input shape: torch.Size([2, 2, 2])
print(f"Input:\n{x}")
# Input:
# tensor([[[1, 2],
#          [3, 4]],
#         [[5, 6],
#          [7, 8]]])

# Pad: left=1, right=1, top=1, bottom=1, front=1, back=1
# Format: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
print(f"Output shape: {padded.shape}")  # Output shape: torch.Size([4, 4, 4])
# Input shape: [2, 2, 2] → Output shape: [2+1+1, 2+1+1, 2+1+1] = [4, 4, 4]

# Different padding for each dimension
padded = F.pad(x, (2, 1, 1, 2, 0, 1), mode='constant', value=0)
# W: left=2, right=1
# H: top=1, bottom=2
# D: front=0, back=1
print(f"Output shape: {padded.shape}")  # Output shape: torch.Size([3, 5, 5])
# Input shape: [2, 2, 2] → Output shape: [2+0+1, 2+1+2, 2+2+1] = [3, 5, 5]
```

**Pad tuple format for 3D:** `(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)`
- **Order**: Last dimension (width) first, then height, then depth

---

### 4D Tensor (Shape: `[N, C, H, W]` - Batch of Images)

```python
# Batch of 2 images, 1 channel, 3x3
x = torch.tensor([[[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]],
                  [[[10, 11, 12],
                    [13, 14, 15],
                    [16, 17, 18]]]])
print(f"Input shape: {x.shape}")  # Input shape: torch.Size([2, 1, 3, 3])
print(f"Input[0, 0]:\n{x[0, 0]}")
# Input[0, 0]:
# tensor([[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]])

# Pad: W(left, right), H(top, bottom), C(front, back), N(front, back)
# Format: (pad_W_left, pad_W_right, pad_H_top, pad_H_bottom, 
#          pad_C_front, pad_C_back, pad_N_front, pad_N_back)
padded = F.pad(x, (1, 1, 1, 1, 0, 0, 0, 0), mode='constant', value=0)
print(f"Output shape: {padded.shape}")  # Output shape: torch.Size([2, 1, 5, 5])
# Input shape: [2, 1, 3, 3] → Output shape: [2+0+0, 1+0+0, 3+1+1, 3+1+1] = [2, 1, 5, 5]
# Only padded width and height dimensions
print(f"Output[0, 0]:\n{padded[0, 0]}")
# Output[0, 0]:
# tensor([[0, 0, 0, 0, 0],
#         [0, 1, 2, 3, 0],
#         [0, 4, 5, 6, 0],
#         [0, 7, 8, 9, 0],
#         [0, 0, 0, 0, 0]])
```

**Pad tuple format for 4D:** `(pad_W_left, pad_W_right, pad_H_top, pad_H_bottom, pad_C_front, pad_C_back, pad_N_front, pad_N_back)`

---

## Key Points

### 1. **Padding Order is Reversed**
The padding tuple starts with the **last dimension** and works backwards:
- For 2D: `(left, right, top, bottom)` → Last dim (width) first, then height
- For 3D: `(left, right, top, bottom, front, back)` → Last dim first, then second-to-last, etc.

### 2. **Common Use Cases**

**Image Padding (2D/4D):**
```python
# Add border around image
image = torch.randn(1, 3, 224, 224)  # [N, C, H, W]
print(f"Input shape: {image.shape}")  # Input shape: torch.Size([1, 3, 224, 224])
padded = F.pad(image, (1, 1, 1, 1, 0, 0, 0, 0), mode='reflect')
print(f"Output shape: {padded.shape}")  # Output shape: torch.Size([1, 3, 226, 226])
# Input shape: [1, 3, 224, 224] → Output shape: [1, 3, 224+1+1, 224+1+1] = [1, 3, 226, 226]
# Adds 1-pixel border on all sides
```

**Sequence Padding (1D/2D):**
```python
# Pad sequence to fixed length
sequence = torch.tensor([1, 2, 3])
print(f"Input: {sequence}, shape: {sequence.shape}")  # Input: tensor([1, 2, 3]), shape: torch.Size([3])
padded = F.pad(sequence, (0, 5), mode='constant', value=0)
print(f"Output: {padded}, shape: {padded.shape}")  # Output: tensor([1, 2, 3, 0, 0, 0, 0, 0]), shape: torch.Size([8])
# Input shape: [3] → Output shape: [3 + 0 + 5] = [8]
# Pads right side to make length 8
```

**3D Volume Padding:**
```python
# Pad 3D volume
volume = torch.randn(32, 32, 32)  # [D, H, W]
print(f"Input shape: {volume.shape}")  # Input shape: torch.Size([32, 32, 32])
padded = F.pad(volume, (2, 2, 2, 2, 2, 2), mode='constant', value=0)
print(f"Output shape: {padded.shape}")  # Output shape: torch.Size([36, 36, 36])
# Input shape: [32, 32, 32] → Output shape: [32+2+2, 32+2+2, 32+2+2] = [36, 36, 36]
# Adds 2-voxel border on all sides
```

### 3. **Mode Selection Guide**

- **`'constant'`**: Default, good for most cases, supports custom fill value
- **`'reflect'`**: Good for images (preserves edge information), no edge duplication
- **`'replicate'`**: Simple edge extension, duplicates edge values
- **`'circular'`**: Useful for periodic signals, wraps around

### 4. **Output Shape Calculation**

For a tensor of shape `[D1, D2, ..., Dn]` with padding `(p1, p2, ..., p2n)`:
- Output shape: `[D1 + p(2n-1) + p(2n), D2 + p(2n-3) + p(2n-2), ..., Dn + p1 + p2]`

**Example:**
```python
# 2D: [H, W] with pad (left, right, top, bottom)
# Output: [H + top + bottom, W + left + right]

x = torch.randn(5, 10)  # [H=5, W=10]
padded = F.pad(x, (2, 3, 1, 1), mode='constant')
# Output shape: [5+1+1, 10+2+3] = [7, 15]
```

---

## Complete Example: All Modes for 2D

```python
import torch
import torch.nn.functional as F

x = torch.tensor([[1, 2],
                  [3, 4]])

print(f"Input shape: {x.shape}")  # Input shape: torch.Size([2, 2])
print("Input:")
print(x)
# Input:
# tensor([[1, 2],
#         [3, 4]])
print()

# Constant mode
padded_const = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)
print(f"Constant mode - Output shape: {padded_const.shape}")  # Output shape: torch.Size([4, 4])
print("Constant (value=0):")
print(padded_const)
# Input shape: [2, 2] → Output shape: [2+1+1, 2+1+1] = [4, 4]
# tensor([[0, 0, 0, 0],
#         [0, 1, 2, 0],
#         [0, 3, 4, 0],
#         [0, 0, 0, 0]])
print()

# Reflect mode
padded_reflect = F.pad(x, (1, 1, 1, 1), mode='reflect')
print(f"Reflect mode - Output shape: {padded_reflect.shape}")  # Output shape: torch.Size([4, 4])
print("Reflect:")
print(padded_reflect)
# Input shape: [2, 2] → Output shape: [4, 4]
print()

# Replicate mode
padded_replicate = F.pad(x, (1, 1, 1, 1), mode='replicate')
print(f"Replicate mode - Output shape: {padded_replicate.shape}")  # Output shape: torch.Size([4, 4])
print("Replicate:")
print(padded_replicate)
# Input shape: [2, 2] → Output shape: [4, 4]
print()

# Circular mode
padded_circular = F.pad(x, (1, 1, 1, 1), mode='circular')
print(f"Circular mode - Output shape: {padded_circular.shape}")  # Output shape: torch.Size([4, 4])
print("Circular:")
print(padded_circular)
# Input shape: [2, 2] → Output shape: [4, 4]
```

---

## ONNX Pad vs PyTorch Pad

**Important Note**: ONNX Pad uses a different format:
- ONNX: `pads = [begin_0, begin_1, ..., begin_n, end_0, end_1, ..., end_n]`
- PyTorch: `pad = (end_n, begin_n, end_(n-1), begin_(n-1), ..., end_0, begin_0)`

The converter needs to handle this format conversion when translating ONNX Pad to PyTorch pad.

