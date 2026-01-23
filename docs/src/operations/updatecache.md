# forge.op.UpdateCache

## Overview

UpdateCache writes a single token (S=1) slice into the cache tensor on specified index.

## Function Signature

```python
forge.op.UpdateCache(
    name: str,
    cache: Tensor,
    input: Tensor,
    update_index: int,
    batch_offset: int = 0
) -> Tensor
```

## Parameters

- **name** (`str`): str Unique op name.

- **cache** (`Tensor`): Tensor 4D cache tensor of shape [B, H, S_total, D]

- **input** (`Tensor`): Tensor 4D input tensor of shape [B, H, 1, D]

- **update_index** (`int`): update_index parameter

- **batch_offset** (`int`, default: `0`): int Offset in the batch dimension.

## Returns

- **result** (`Tensor`): Output tensor
