# Tools

This page covers setup of various tools that can help you with development of TT-Forge-ONNX. The sections include:
* [Pre-commit](#pre-commit)
* [mdbook](#mdbook)
* [Cross Correlate Models and Ops and Export Model Variants Unique Op Configuration](#cross-correlate-models-and-ops-and-export-model-variants-unique-op-configuration)
* [Usage](#usage)

## Pre-commit

TT-Forge-ONNX defines various pre-commit hooks that check the code for formatting, licensing issues, etc.

To install pre-commit, run the following command:

```sh
source env/activate
pip install pre-commit
```

After installing pre-commit, you can install the hooks by running:

```sh
pre-commit install
```

Now, each time you run `git commit` the pre-commit hooks (checks) will be executed.

If you already committed before installing the pre-commit hooks, you can run it on all files to catch up:

```sh
pre-commit run --all-files
```

For more information visit [pre-commit](https://pre-commit.com/).

## mdbook

TT-Forge-ONNX uses `mdbook` to generate the documentation. To install `mdbook` on Ubuntu, run the following commands:

```sh
sudo apt install cargo
cargo install mdbook
```

>**NOTE:** If you do not want to install `mdbook` via cargo (Rust package manager), consult the [Official mdbook Installation Guide](https://rust-lang.github.io/mdBook/cli/index.html).

## Gather Unique Ops Configuration

The model's unique ops configuration can be gathered, and the results can be printed to the console and saved as a CSV or XLSX file.

1. **FORGE_EXTRACT_UNIQUE_OP_CONFIG_AT**
   - By setting this flag to one of the following options, the model's unique ops configuration can be extracted at a specific compilation stage or across all stages:

     - **`FORGE_EXTRACT_UNIQUE_OP_CONFIG_AT = ALL`**
       Extracts all the unique ops configurations present in the graph at every compilation stage.

     - **`FORGE_EXTRACT_UNIQUE_OP_CONFIG_AT = {GENERATE_INITIAL_GRAPH / POST_INITIAL_GRAPH_PASS / OPTIMIZED_GRAPH / AUTOGRAD / POST_AUTOGRAD_PASS / PRE_LOWERING_GRAPH}`**
       Extracts the unique ops configuration only at the specified compilation stage.

2. **FORGE_PRINT_UNIQUE_OP_CONFIG**
   - By setting this flag to `1`, all unique configurations will be printed to the console.

3. **FORGE_EXPORT_UNIQUE_OP_CONFIG_FILE_TYPE**
   - By setting this flag to `csv` or `xlsx`, all unique configurations will be exported as CSV or XLSX file. The file can be saved to the default path (for example, the current directory), or it can be saved to a specific path by setting the `FORGE_EXPORT_UNIQUE_OP_CONFIG_DIR_PATH` environment variable.

4. **FORGE_EXPORT_UNIQUE_OP_CONFIG_CSV_DELIMITER**
   - The delimiter for the csv file can be set by using this flag. Default delimiter : slash (i.e `/`)


> **Note:**
> The delimiter used in the CSV file will be a slash (`/`) to avoid potential parsing issues. Commas (`,`) and hyphen (`-`) may appear in the op shapes and attributes, which could lead to misinterpretation of the data.

## Cross Correlate Models and Ops and Export Model Variants Unique Op Configuration

The models and ops can be cross-correlated and model variants unique op configuration are exported as an XLSX file by running the `scripts/export_models_ops_correlation.py` Python script.

The script performs the following tasks:

1. Run all models until the compile depth specified by the user.
2. Export unique op requirements to a file (each model variants has its own directory, in that directory each compile depth has its own file).
3. Parse those unique op requirements and create a XLSX file that can be loaded into a google sheet.
   1. The XLSX file will contain list of models on X axis (i.e. columns) and list of ops on Y axis (i.e. rows/indices).
   2. Elements in between will contain a checkmark if the desired op from the Y axis (i.e., rows/indices) exists in the model on X axis (i.e., columns).
   3. Models will be sorted alphabetically.
   4. Ops will be sorted by the number of occurrences in the models.

### Usage

To run the script, use the following command:

```sh
python scripts/export_models_ops_correlation.py
```

### Required Options:

|                              **Option**                                   |                                                **Description**                                                   |
| :-----------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: |
| `-c`, `--compile_depth (GENERATE_INITIAL_GRAPH, PRE_LOWERING_PASS, etc.)` | Choose the compilation depth for extracting ops configuration for the models present in `pytest_directory_path`. |
| `-i`, `--pytest_directory_path`                                           | Specify the directory path containing models to test.                                                            |

### Optional Options:

|                              **Option**                                   |                                                **Description**                                                   |
| :-----------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: |
| `--cross_correlation_output_file_name`                                    | Specify the output XLSX file name for saving the cross correlation data between model variants and unique ops.   |
| `--models_unique_op_configs_output_file_name`                             | Specify the output XLSX file name for saving the Models unique op configurations.                                |
| `-o`, `--output_directory_path`                                           | Specify the output directory path for saving the XLSX/CSV file.                                                  |
| `--export_unique_op_config_file_type` (CSV, XLSX)                         | Specify the export unique op configuration file type                                                             |

### Example:

```sh
python scripts/export_models_ops_correlation.py --compile_depth GENERATE_INITIAL_GRAPH --pytest_directory_path forge/test/model_demos/high_prio/nlp/pytorch
```

## Operations Documentation Generator

The operations documentation generator automatically creates documentation for all Forge operations by parsing the source files in `forge/forge/op/*.py`.

### Quick Start

```sh
# Generate all operation documentation
python scripts/generate_ops_docs.py
```

### How It Works

1. **Automatic Discovery**: The generator scans `forge/forge/op/*.py` files to discover all operations (functions starting with uppercase letters).

2. **Docstring Parsing**: It parses NumPy-style docstrings to extract:
   - Operation overview/description
   - Parameter descriptions with types
   - Return value descriptions
   - Mathematical definitions
   - Related operations

3. **Enhancement Layer**: Additional documentation (e.g., mathematical formulas, related ops) can be added via `scripts/operation_enhancements.json`.

4. **Markdown Generation**: Creates clean markdown files for each operation and an index page.

### Command-Line Options

The generator supports the following command-line arguments:

| Option | Description | Default |
|--------|-------------|---------|
| `--op-dir` | Source directory for operations | `forge/forge/op/` |
| `--output-dir` | Output directory for operation docs | `docs/src/operations/` |
| `--index-file` | Output path for index page | `docs/src/operations.md` |
| `--enhancements` | Path to enhancements JSON file | `scripts/operation_enhancements.json` |
| `--no-cleanup` | Skip cleanup of stale documentation files | (cleanup enabled by default) |

Example with custom paths:

```sh
python scripts/generate_ops_docs.py --op-dir forge/forge/op --output-dir docs/src/operations
```

### Adding New Operations

**No manual documentation needed!** Simply:

1. Add your operation function to `forge/forge/op/*.py`
2. Write a proper docstring following the standard format (see `docs/FORGE_DOCSTRING_STANDARD.md`)
3. Run `python scripts/generate_ops_docs.py`

The documentation will be automatically generated.

### Docstring Standard

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
        Input tensor of shape `(N, C, H, W)`.

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

### Output Files

The generator creates:
- `docs/src/operations.md` - Index page with all operations by category
- `docs/src/operations/*.md` - Individual operation documentation pages

### Stale File Cleanup

The generator automatically removes documentation files for operations that no longer exist in the source code. This ensures the documentation stays in sync with the codebase. To disable this behavior, use the `--no-cleanup` flag.

### Enhancements File

The `scripts/operation_enhancements.json` file allows adding extra documentation that can't be extracted from docstrings:

```json
{
  "operations": {
    "Abs": {
      "description": "Enhanced description for the operation overview",
      "mathematical_definition": "abs(x) = |x|",
      "parameters": {
        "operandA": "Enhanced description for operandA parameter"
      },
      "related_operations": [
        {"name": "Relu", "description": "ReLU activation"}
      ]
    }
  }
}
```

Supported enhancement types:
- `description`: Override or supplement the operation overview
- `parameters`: Object mapping parameter names to enhanced descriptions
- `mathematical_definition`: Mathematical formula for the operation
- `related_operations`: List of related operations with descriptions

**Note**: The goal is to migrate all documentation to source docstrings. Use this file only when necessary.

### Error Handling

The generator will **fail fast** if:
- The operation directory doesn't exist
- No operations are discovered
- Critical parsing errors occur

Warnings are issued for:
- Missing docstrings
- Non-critical parsing issues
