# forge.op.Embedding

Embedding lookup

## Function Signature

```python
forge.op.Embedding(name: str, indices: Tensor, embedding_table: Union[Tensor, Parameter]) -> Tensor
```

## Parameters

- **indices** (Tensor): Integer tensor, the elements of which are used to index into the embedding table
- **embedding_table** (Union[Tensor, Parameter]): Dictionary of embeddings
## Returns

- **result** (Output tensor): Tensor

