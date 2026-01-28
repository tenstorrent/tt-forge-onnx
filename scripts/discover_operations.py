# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Automatic operation discovery from forge/forge/op/*.py files.

This module parses Python operation files to automatically discover
all operations and extract their documentation.

Usage:
    python scripts/discover_operations.py [options]

Options:
    --op-dir PATH    Source directory for operations (default: forge/forge/op/)
"""

import argparse
import ast
import sys
from contextlib import suppress
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


class OperationDiscoveryError(Exception):
    """Raised when operation discovery fails."""


@dataclass
class DiscoveredOperation:
    """Represents a discovered operation from the codebase."""

    name: str  # Function name (e.g., "Abs")
    module_path: str  # File path (e.g., "forge.forge.op.eltwise_unary")
    file_name: str  # File name (e.g., "eltwise_unary.py")
    docstring: str
    signature: str
    parameters: List[Dict[str, str]]  # List of {name, type, default, description}
    return_type: str
    return_description: str
    category: str  # Inferred from file name or function patterns


class OperationDiscoverer:
    """Discovers operations from forge/forge/op/*.py files."""

    # Map file names to categories
    FILE_TO_CATEGORY = {
        "eltwise_unary.py": "Elementwise Operations",
        "eltwise_binary.py": "Elementwise Operations",
        "eltwise_nary.py": "Elementwise Operations",
        "convolution.py": "Convolution Operations",
        "pooling.py": "Pooling Operations",
        "reduce.py": "Reduction Operations",
        "matmul.py": "Linear Operations",
        "tm.py": "Tensor Manipulation",
        "nn.py": "Normalization Operations",
        "resize.py": "Tensor Manipulation",
        "embedding.py": "Other Operations",
        "kv_cache.py": "Memory Operations",
        "constant.py": "Other Operations",
        "misc.py": "Other Operations",
        "loss.py": "Other Operations",
    }

    # Activation functions - override category
    ACTIVATION_FUNCTIONS = {"Relu", "LeakyRelu", "Sigmoid", "Tanh", "Gelu", "Silu", "HardSigmoid"}

    def __init__(self, project_root: Path, op_dir: Optional[Path] = None):
        self.project_root = project_root
        self.op_dir = op_dir or (project_root / "forge" / "forge" / "op")

    def _ast_to_string(self, node) -> str:
        """Convert AST node to string."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Attribute):
            return f"{self._ast_to_string(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._ast_to_string(node.value)}[{self._ast_to_string(node.slice)}]"
        elif isinstance(node, ast.Tuple):
            return "(" + ", ".join(self._ast_to_string(x) for x in node.elts) + ")"
        elif hasattr(ast, "unparse"):
            return ast.unparse(node)
        else:
            return str(node)

    def discover_all_operations(self) -> List[DiscoveredOperation]:
        """
        Discover all operations from the op directory.

        Raises
        ------
        OperationDiscoveryError
            If the operation directory does not exist or no operations are found.
        """
        # VALIDATION: Fail fast if directory doesn't exist
        if not self.op_dir.exists():
            raise OperationDiscoveryError(
                f"Operation directory not found: {self.op_dir}\n" f"Expected forge operations at: forge/forge/op/"
            )

        operations = []
        parse_errors = []

        # Get all Python files in op directory (except __init__.py and common.py)
        op_files = [f for f in self.op_dir.glob("*.py") if f.name not in ("__init__.py", "common.py")]

        if not op_files:
            raise OperationDiscoveryError(
                f"No operation files found in: {self.op_dir}\n" f"Expected Python files defining Forge operations."
            )

        print(f"Found {len(op_files)} operation files to parse")

        for op_file in op_files:
            try:
                file_ops = self._parse_file(op_file)
                operations.extend(file_ops)
                print(f"  [OK] {op_file.name}: {len(file_ops)} operations")
            except SyntaxError as e:
                error_msg = f"Syntax error in {op_file.name}: {e}"
                parse_errors.append(error_msg)
                print(f"  [FAIL] {op_file.name}: SYNTAX ERROR - {e}")
            except Exception as e:
                error_msg = f"Failed to parse {op_file.name}: {e}"
                parse_errors.append(error_msg)
                print(f"  [FAIL] {op_file.name}: ERROR - {e}")

        # VALIDATION: Report all parse errors
        if parse_errors:
            print(f"\nWarning: {len(parse_errors)} file(s) had parsing errors:")
            for err in parse_errors:
                print(f"  - {err}")

        # VALIDATION: Fail if no operations discovered
        if not operations:
            raise OperationDiscoveryError(
                f"No operations discovered from {len(op_files)} files.\n"
                f"Parse errors: {len(parse_errors)}\n"
                f"Ensure operation functions start with uppercase letters."
            )

        return operations

    def _parse_file(self, file_path: Path) -> List[DiscoveredOperation]:
        """Parse a single Python file to extract operations."""
        operations = []

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)  # Let SyntaxError propagate

        # Determine category from file name
        category = self.FILE_TO_CATEGORY.get(file_path.name, "Other Operations")

        # Find all function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Only process functions that start with uppercase (operation functions)
                if node.name and node.name[0].isupper():
                    op = self._parse_function(node, file_path, category)
                    if op:
                        operations.append(op)

        return operations

    def _parse_function(
        self, func_node: ast.FunctionDef, file_path: Path, category: str
    ) -> Optional[DiscoveredOperation]:
        """Parse a function node to extract operation information."""
        # Get docstring
        docstring = ast.get_docstring(func_node) or ""

        if not docstring:
            print(f"    Warning: {func_node.name} has no docstring")

        # Parse signature
        sig = self._get_function_signature(func_node)

        # Parse parameters
        parameters = self._parse_parameters(func_node, docstring)

        # Parse return type and description
        return_type = self._get_return_type(func_node)
        return_description = self._extract_return_description(docstring)

        # Get module path
        module_path = f"forge.forge.op.{file_path.stem}"

        # Override category for activation functions
        if func_node.name in self.ACTIVATION_FUNCTIONS:
            category = "Activation Functions"

        return DiscoveredOperation(
            name=func_node.name,
            module_path=module_path,
            file_name=file_path.name,
            docstring=docstring,
            signature=sig,
            parameters=parameters,
            return_type=return_type,
            return_description=return_description,
            category=category,
        )

    def _get_function_signature(self, func_node: ast.FunctionDef) -> str:
        """Get function signature as string."""
        args = []
        defaults_start = len(func_node.args.args) - len(func_node.args.defaults)

        for i, arg in enumerate(func_node.args.args):
            arg_str = arg.arg

            if arg.annotation:
                with suppress(Exception):
                    annotation = self._ast_to_string(arg.annotation)
                    arg_str += f": {annotation}"

            if i >= defaults_start:
                default_idx = i - defaults_start
                if default_idx < len(func_node.args.defaults):
                    with suppress(Exception):
                        default = func_node.args.defaults[default_idx]
                        default_str = self._ast_to_string(default)
                        arg_str += f" = {default_str}"

            args.append(arg_str)

        return_annotation = ""
        if func_node.returns:
            with suppress(Exception):
                return_annotation = f" -> {self._ast_to_string(func_node.returns)}"

        return f"forge.op.{func_node.name}({', '.join(args)}){return_annotation}"

    def _parse_parameters(self, func_node: ast.FunctionDef, docstring: str) -> List[Dict[str, str]]:
        """Parse function parameters from signature and docstring."""
        params = []
        param_descriptions = self._extract_parameter_descriptions(docstring)

        defaults_start = len(func_node.args.args) - len(func_node.args.defaults)

        for i, arg in enumerate(func_node.args.args):
            param_info = {
                "name": arg.arg,
                "type": "",
                "default": None,
                "description": param_descriptions.get(arg.arg, ""),
            }

            if arg.annotation:
                with suppress(Exception):
                    param_info["type"] = self._ast_to_string(arg.annotation)

            if i >= defaults_start:
                default_idx = i - defaults_start
                if default_idx < len(func_node.args.defaults):
                    with suppress(Exception):
                        default = func_node.args.defaults[default_idx]
                        param_info["default"] = self._ast_to_string(default)

            params.append(param_info)

        return params

    def _extract_parameter_descriptions(self, docstring: str) -> Dict[str, str]:
        """Extract parameter descriptions from NumPy-style docstring."""
        descriptions = {}

        if "Parameters" not in docstring:
            return descriptions

        lines = docstring.split("\n")
        in_params = False
        current_param = None
        current_desc = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            if stripped.startswith("Parameters"):
                in_params = True
                continue

            if in_params and (
                stripped.startswith("---")
                or stripped.startswith("Returns")
                or stripped.startswith("Mathematical")
                or stripped.startswith("See Also")
                or stripped.startswith("Notes")
                or stripped.startswith("Examples")
            ):
                if current_param:
                    descriptions[current_param] = " ".join(current_desc).strip()
                if not stripped.startswith("---"):
                    break
                continue

            if in_params:
                # Check for parameter line (not indented, contains ':')
                if stripped and not line.startswith(" ") and not line.startswith("\t"):
                    if ":" in stripped:
                        if current_param:
                            descriptions[current_param] = " ".join(current_desc).strip()

                        parts = stripped.split(":", 1)
                        param_name = parts[0].strip()
                        if param_name and not param_name.isupper() and len(param_name) < 50:
                            current_param = param_name
                            current_desc = [parts[1].strip()] if len(parts) > 1 and parts[1].strip() else []
                        else:
                            current_param = None
                    elif current_param is None:
                        continue
                elif current_param and stripped:
                    if line.startswith(" ") or line.startswith("\t"):
                        current_desc.append(stripped)

        if current_param:
            descriptions[current_param] = " ".join(current_desc).strip()

        return descriptions

    def _extract_return_description(self, docstring: str) -> str:
        """Extract return description from docstring."""
        if "Returns" not in docstring:
            return ""

        lines = docstring.split("\n")
        in_returns = False
        return_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Returns"):
                in_returns = True
                continue
            if in_returns:
                if stripped.startswith("---"):
                    continue
                if stripped and not line.startswith(" ") and not line.startswith("\t"):
                    # New section
                    if stripped.startswith(("Mathematical", "See Also", "Notes", "Examples")):
                        break
                if stripped:
                    return_lines.append(stripped)

        return " ".join(return_lines).strip()

    def _get_return_type(self, func_node: ast.FunctionDef) -> str:
        """Get return type annotation."""
        if func_node.returns:
            with suppress(Exception):
                return self._ast_to_string(func_node.returns)
        return ""


def discover_operations(project_root: Path, op_dir: Optional[Path] = None) -> List[DiscoveredOperation]:
    """
    Discover all operations from the codebase.

    Args:
        project_root: Path to the project root directory
        op_dir: Optional path to the operations directory (default: forge/forge/op/)

    Returns:
        List of discovered operations

    Raises:
        OperationDiscoveryError: If discovery fails
    """
    discoverer = OperationDiscoverer(project_root, op_dir)
    return discoverer.discover_all_operations()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Discover Forge operations from source files.", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--op-dir", type=Path, default=None, help="Source directory for operations (default: forge/forge/op/)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    project_root = Path(__file__).parent.parent

    try:
        operations = discover_operations(project_root, args.op_dir)
        print(f"\nDiscovered {len(operations)} operations:")

        # Group by category
        by_category = {}
        for op in operations:
            if op.category not in by_category:
                by_category[op.category] = []
            by_category[op.category].append(op)

        for category in sorted(by_category.keys()):
            print(f"\n{category}:")
            for op in sorted(by_category[category], key=lambda x: x.name):
                print(f"  - {op.name}")

    except OperationDiscoveryError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
