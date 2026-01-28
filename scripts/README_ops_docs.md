# Forge Operations Documentation Generator

This directory contains scripts for automatically generating documentation for Forge operations.

## Quick Start

```bash
# Generate all operation documentation
python scripts/generate_ops_docs.py
```

## Documentation

For complete documentation on the operations documentation generator, including:
- Command-line options
- Docstring standards
- Enhancement file format
- Error handling

See the **Tools** section in the documentation: [`docs/src/tools.md`](../docs/src/tools.md#operations-documentation-generator)

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
