# Operations Documentation Generator

This directory contains scripts for automatically generating operation documentation in PyTorch-style format.

## Overview

The documentation generator creates:
1. **Index Page** (`docs/src/operations.md`): A categorized list of all operations with short descriptions
2. **Individual Operation Pages** (`docs/src/operations/*.md`): Detailed documentation for each operation

## Files

- `generate_ops_docs.py`: Main script that generates the documentation
- `ops_data.py`: Data file containing operation definitions with descriptions, parameters, examples, etc.

## Usage

### Generate Documentation

From the project root directory:

```bash
python scripts/generate_ops_docs.py
```

This will:
- Load operation definitions from `scripts/ops_data.py`
- Generate individual operation pages in `docs/src/operations/`
- Generate the index page at `docs/src/operations.md`

### Adding New Operations

To add a new operation, edit `scripts/ops_data.py` and add a new `Operation` object to the `get_all_operations()` function:

```python
Operation(
    name="ttir.new_op",
    short_name="new_op",
    category="Elementwise Operations",
    description="Short description of the operation.",
    detailed_description="Detailed description...",
    operands=[
        Operand("input", "ranked tensor of any type values", "Description of input"),
        Operand("output", "ranked tensor of any type values", "Description of output")
    ],
    results=[Operand("result", "ranked tensor of any type values", "Description of result")],
    attributes=[
        Attribute("attr_name", "attr_type", "Description", "default_value")
    ],
    examples=["Example code here"],
    notes=["Additional notes"]
)
```

### Operation Categories

Operations are automatically categorized based on their names. Categories include:
- Elementwise Operations
- Convolution Functions
- Pooling Functions
- Normalization Functions
- Tensor Manipulation
- Reduction Operations
- Linear Functions
- Attention Mechanisms
- Embedding Functions
- Memory Operations
- Creation Operations
- Quantization Operations
- Conditional Operations
- Collective Operations
- Other Operations

To add a new category or change categorization, edit the `CATEGORIES` dictionary in `scripts/generate_ops_docs.py`.

## Documentation Structure

Each operation page includes:
- **Title**: Full operation name (e.g., `ttir.abs`)
- **Description**: Short and detailed descriptions
- **Function Signature**: Python-style function signature
- **Parameters**: Input operands and attributes with types and descriptions
- **Returns**: Output operands with descriptions
- **Mathematical Definition**: Mathematical formula (if applicable)
- **Examples**: Code examples showing usage
- **Notes**: Additional implementation details or warnings

## Integration with mdBook

The generated documentation is integrated into the mdBook documentation system:
- The index page is referenced in `docs/src/SUMMARY.md`
- Individual pages are linked from the index page
- The documentation follows mdBook markdown conventions

## Future Enhancements

Potential improvements:
1. Parse operations directly from MLIR definitions
2. Extract examples from test files
3. Generate parameter tables automatically
4. Add cross-references between related operations
5. Include performance characteristics
6. Add version information for each operation

