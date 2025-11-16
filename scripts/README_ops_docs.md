# Operations Documentation Generator

This directory contains scripts for automatically generating operation documentation in PyTorch-style format.

## Overview

The documentation generator creates:
1. **Index Page** (`docs/src/operations.md`): A categorized list of all operations with short descriptions
2. **Individual Operation Pages** (`docs/src/operations/*.md`): Detailed documentation for each operation

## Files

- `generate_ops_docs.py`: Main script that generates the documentation
- `discover_operations.py`: Automatic operation discovery from `forge/forge/op/*.py` files
- `ops_data.py`: Fallback data file containing manually curated operation definitions (used only if automatic discovery fails)

## Usage

### Generate Documentation

From the project root directory:

```bash
python scripts/generate_ops_docs.py
```

This will:
- **Automatically discover** operations from `forge/forge/op/*.py` files
- Extract function signatures, docstrings, and parameter information
- Generate individual operation pages in `docs/src/operations/`
- Generate the index page at `docs/src/operations.md`
- Fall back to `scripts/ops_data.py` only if automatic discovery fails

### Adding New Operations

**Operations are automatically discovered** from the Python files in `forge/forge/op/`. To add a new operation to the documentation, simply:

1. **Define your operation function** in the appropriate file (e.g., `forge/forge/op/eltwise_unary.py`):
   ```python
   def NewOp(name: str, operandA: Tensor) -> Tensor:
       """
       Short description of the operation.
       
       Optional detailed description can go here.
       
       Parameters
       ----------
       name: str
           Op name, unique to the module, or leave blank to autoset
       
       operandA: Tensor
           Description of the first operand
       
       Returns
       -------
       Tensor
           Description of the return value
       """
       return op(OpType.NewOp, name, operandA).get_tensor()
   ```

2. **Use a proper docstring** (preferably NumPy-style with Parameters and Returns sections)

3. **Run the generator**: `python scripts/generate_ops_docs.py`

The generator will automatically:
- Discover the new operation
- Extract its name, parameters, types, and descriptions
- Infer its category based on the file it resides in
- Generate a new markdown page for it

**Note**: The function name must start with an uppercase letter to be recognized as an operation.

### Operation Categories

Operations are **automatically categorized** based on the file they reside in. The category mapping is defined in `discover_operations.py`:

- `eltwise_unary.py`, `eltwise_binary.py`, `eltwise_nary.py` → **Elementwise Operations**
- `convolution.py` → **Convolution Functions**
- `pooling.py` → **Pooling Functions**
- `reduce.py` → **Reduction Operations**
- `matmul.py` → **Linear Functions**
- `tm.py` → **Tensor Manipulation**
- `nn.py` → **Normalization Functions**
- `resize.py` → **Resize Operations**
- `embedding.py` → **Embedding Functions**
- `kv_cache.py` → **Memory Operations**
- `constant.py` → **Creation Operations**
- `misc.py` → **Other Operations**
- `loss.py` → **Loss Functions**

To add a new category or change categorization, edit the `FILE_TO_CATEGORY` dictionary in `scripts/discover_operations.py`.

## Documentation Structure

Each operation page includes:
- **Title**: Full operation name (e.g., `forge.op.Abs`)
- **Description**: Short and detailed descriptions (extracted from docstring)
- **Function Signature**: Python-style function signature (extracted from source code)
- **Parameters**: Input operands and attributes with types and descriptions (from docstring)
- **Returns**: Output operands with descriptions (from docstring)
- **Mathematical Definition**: Mathematical formula (if applicable)
- **Examples**: Code examples showing usage
- **Notes**: Additional implementation details or warnings

## Integration with mdBook

The generated documentation is integrated into the mdBook documentation system:
- The index page is referenced in `docs/src/SUMMARY.md`
- Individual pages are linked from the index page
- The documentation follows mdBook markdown conventions

## How It Works

1. **Discovery Phase** (`discover_operations.py`):
   - Scans `forge/forge/op/*.py` files
   - Uses Python AST to parse function definitions
   - Extracts docstrings, signatures, and parameter information
   - Infers categories from file names

2. **Conversion Phase** (`generate_ops_docs.py`):
   - Converts discovered operations to documentation format
   - Cleans up descriptions and removes redundant type prefixes
   - Handles edge cases (missing docstrings, incorrect descriptions)

3. **Generation Phase** (`generate_ops_docs.py`):
   - Creates individual markdown pages for each operation
   - Generates categorized index page
   - Formats in PyTorch-style documentation format

## Future Enhancements

Potential improvements:
1. Extract examples from test files automatically
2. Generate parameter tables automatically
3. Add cross-references between related operations
4. Include performance characteristics
5. Add version information for each operation
6. Support for extracting mathematical definitions from docstrings

