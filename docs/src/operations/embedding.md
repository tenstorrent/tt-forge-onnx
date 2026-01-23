# forge.op.Embedding

## Overview

Embedding lookup

## Function Signature

```python
forge.op.Embedding(
    name: str,
    indices: Tensor,
    embedding_table: Union[(Tensor, Parameter)]
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **indices** (`Tensor`): Tensor Integer tensor, the elements of which are used to index into the embedding table

- **embedding_table** (`Union[(Tensor, Parameter)]`): Tensor Dictionary of embeddings

## Returns

- **result** (`Tensor`): Output tensor
