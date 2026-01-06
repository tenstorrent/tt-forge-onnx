# Forge Operations Documentation

This directory contains automatically generated documentation for all Forge operations.

## File Naming Convention

Each operation has its own markdown file named after the operation:
- `abs.md` - Abs operation
- `relu.md` - Relu operation
- `conv2d.md` - Conv2d operation
- etc.

## Regenerating Documentation

To regenerate all documentation:

```bash
python scripts/generate_ops_docs.py
```

## Adding New Operations

When you add a new operation to `forge/forge/op/*.py`:

1. Ensure your function name starts with an uppercase letter
2. Add a proper NumPy-style docstring (see `docs/FORGE_DOCSTRING_STANDARD.md`)
3. Run the documentation generator

The documentation will be automatically created.

## Documentation Structure

Each operation page includes:
- **Overview**: What the operation does
- **Function Signature**: Python API with type hints
- **Parameters**: Detailed parameter descriptions
- **Returns**: Return value description
- **Mathematical Definition**: Formula (when applicable)
- **Related Operations**: Links to related ops

## Source of Truth

All documentation is sourced from:
1. Operation function docstrings in `forge/forge/op/*.py`
2. Enhancement data in `scripts/operation_enhancements.json`

The docstrings are the primary source. The enhancement file provides supplementary information that cannot be easily extracted from code.
