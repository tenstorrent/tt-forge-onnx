# forge.op.Where

## Overview



## Function Signature

```python
forge.op.Where(name: str, condition: Tensor, x: Tensor, y: Tensor) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **condition** (`Tensor`): Tensor When True (nonzero), yield x, else y

- **x** (`Tensor`): Tensor value(s) if true

- **y** (`Tensor`): Tensor value(s) if false

## Returns

- **result** (`Tensor`): Parameters name: str Op name, unique to the module, or leave blank to autoset condition: Tensor When True (nonzero), yield x, else y x: Tensor value(s) if true y: Tensor value(s) if false Tensor Forge tensor
