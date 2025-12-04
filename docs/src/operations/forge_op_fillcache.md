# forge.op.FillCache

## Overview

FillCache op writes the input into the cache tensor starting at the specified update index.

## Function Signature

```python
forge.op.FillCache(name: str, cache: Tensor, input: Tensor, batch_offset: int) -> Tensor
```

## Parameters

- **name** (`str`): Unique op name.

- **cache** (`Tensor`): 4D cache tensor of shape [B, H, S_total, D]
- **input** (`Tensor`): 4D input tensor of shape [B, H, S_input, D]

- **batch_offset** (`int`, default: `0`): Offset in the batch dimension.

## Returns

- **result** (`Tensor`): Tensor

## Related Operations

*Related operations will be automatically linked here in future updates.*

