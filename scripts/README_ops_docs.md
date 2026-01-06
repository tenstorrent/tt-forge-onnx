# Forge Operations Documentation Generator

This directory contains scripts for automatically generating documentation for Forge operations.

## Quick Start

```bash
# Generate all operation documentation
python scripts/generate_ops_docs.py
```

## How It Works

1. **Automatic Discovery**: The generator scans `forge/forge/op/*.py` files to discover all operations (functions starting with uppercase letters).

2. **Docstring Parsing**: It parses NumPy-style docstrings to extract:
   - Operation overview/description
   - Parameter descriptions with types
   - Return value descriptions
   - Mathematical definitions
   - Related operations

3. **Enhancement Layer**: Additional documentation (e.g., mathematical formulas, related ops) can be added via `scripts/operation_enhancements.json`.

4. **Markdown Generation**: Creates clean markdown files for each operation and an index page.

## Adding New Operations

**No manual documentation needed!** Simply:

1. Add your operation function to `forge/forge/op/*.py`
2. Write a proper docstring following the standard format
3. Run `python scripts/generate_ops_docs.py`

The documentation will be automatically generated.

## Docstring Standard

See `docs/FORGE_DOCSTRING_STANDARD.md` for the complete docstring format. Here's a quick example:

```python
def MyOperation(
    name: str,
    operandA: Tensor,
    param: int = 1,
) -> Tensor:
    """
    Brief one-line description of the operation.

    Detailed description with more context about the operation,
    its use cases, and important behavior notes.

    Parameters
    ----------
    name : str
        Name identifier for this operation in the computation graph.

    operandA : Tensor
        Input tensor of shape `(N, C, H, W)` where:
        - `N` is the batch size
        - `C` is the number of channels
        - `H` is the height
        - `W` is the width

    param : int, optional
        Description of the parameter.
        Default: `1`

    Returns
    -------
    Tensor
        Output tensor with description of shape and meaning.

    Mathematical Definition
    -----------------------
    output[i] = f(input[i])

    See Also
    --------
    forge.op.RelatedOp : Description of related operation
    """
```

## Files

| File | Purpose |
|------|---------|
| `generate_ops_docs.py` | Main documentation generator script |
| `discover_operations.py` | Automatic operation discovery from source files |
| `operation_enhancements.json` | Additional documentation enhancements |
| `README_ops_docs.md` | This file |

## Output

The generator creates:
- `docs/src/operations.md` - Index page with all operations by category
- `docs/src/operations/*.md` - Individual operation documentation pages

## Error Handling

The generator will **fail fast** if:
- The operation directory (`forge/forge/op/`) doesn't exist
- No operations are discovered
- Critical parsing errors occur

Warnings are issued for:
- Missing docstrings
- Non-critical parsing issues

## Enhancements File

The `operation_enhancements.json` file allows adding extra documentation that can't be extracted from docstrings:

```json
{
  "operations": {
    "Abs": {
      "mathematical_definition": "abs(x) = |x|",
      "related_operations": [
        {"name": "Relu", "description": "ReLU activation"}
      ]
    }
  }
}
```

**Note**: The goal is to migrate all documentation to source docstrings. Use this file only when necessary.
