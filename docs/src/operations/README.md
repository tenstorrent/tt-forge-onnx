# Operations Documentation

This directory contains automatically generated documentation for all supported operations.

## Structure

Each operation has its own markdown file with:
- Function signature (extracted from Python source code)
- Parameter descriptions (from docstrings)
- Return value descriptions
- Mathematical definitions (where applicable)
- Code examples
- Implementation details

## Automatic Discovery

Operations are **automatically discovered** from Python files in `forge/forge/op/*.py`. The documentation generator:

1. Scans all Python files in `forge/forge/op/` directory
2. Identifies operation functions (functions starting with uppercase letters)
3. Extracts function signatures, docstrings, and parameter information
4. Infers operation categories from file names
5. Generates documentation pages in PyTorch-style format

**No manual entry is required** for basic operations - simply define your operation function with a proper docstring in the appropriate file.

## Generating Documentation

To regenerate the documentation after adding or updating operations:

```bash
python scripts/generate_ops_docs.py
```

This will:
- Automatically discover all operations from `forge/forge/op/*.py`
- Generate individual operation pages in `docs/src/operations/`
- Generate the index page at `docs/src/operations.md`

See `scripts/README_ops_docs.md` for more details on how the discovery system works and how to add new operations.

## Index

The main operations index is available at [../operations.md](../operations.md).

