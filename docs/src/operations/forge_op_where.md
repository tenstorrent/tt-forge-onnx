# forge.op.Where

Where

## Function Signature

```python
forge.op.Where(name: str, condition: Tensor, x: Tensor, y: Tensor) -> Tensor
```

## Parameters

- **condition** (Tensor): When True (nonzero), yield x, else y
- **x** (Tensor): value(s) if true
- **y** (Tensor): value(s) if false
## Returns

- **result** (Output tensor): Tensor

