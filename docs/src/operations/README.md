# Operations Documentation

This directory contains automatically generated documentation for all supported operations.

## Structure

Each operation has its own markdown file with:
- Function signature
- Parameter descriptions
- Return value descriptions
- Mathematical definitions (where applicable)
- Code examples
- Implementation details

## Generating Documentation

To regenerate the documentation after adding or updating operations:

```bash
python scripts/generate_ops_docs.py
```

See `scripts/README_ops_docs.md` for more details on how to add new operations.

## Index

The main operations index is available at [../operations.md](../operations.md).

