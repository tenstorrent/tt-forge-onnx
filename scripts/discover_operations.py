#!/usr/bin/env python3
"""
Automatic operation discovery from forge/forge/op/*.py files.

This module parses Python operation files to automatically discover
all operations and extract their documentation.
"""

import ast
import inspect
import importlib.util
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys
from dataclasses import dataclass


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
    
    def _ast_to_string(self, node):
        """Convert AST node to string (fallback for Python < 3.9)."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Attribute):
            return f"{self._ast_to_string(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._ast_to_string(node.value)}[{self._ast_to_string(node.slice)}]"
        elif isinstance(node, ast.Union):
            return "Union[" + ", ".join(self._ast_to_string(x) for x in node.elts) + "]"
        elif isinstance(node, ast.Tuple):
            return "(" + ", ".join(self._ast_to_string(x) for x in node.elts) + ")"
        else:
            return str(node)
    
    # Map file names to categories (matching manager's requirements)
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
        "resize.py": "Resize Operations",
        "embedding.py": "Embedding Functions",
        "kv_cache.py": "Memory Operations",
        "constant.py": "Creation Operations",
        "misc.py": "Other Operations",
        "loss.py": "Loss Functions",
    }
    
    # Activation functions that should be in separate category
    ACTIVATION_FUNCTIONS = {
        "Relu", "LeakyRelu", "Sigmoid", "Tanh", "Gelu", "Silu", "HardSigmoid"
    }
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.op_dir = project_root / "forge" / "forge" / "op"
    
    def discover_all_operations(self) -> List[DiscoveredOperation]:
        """Discover all operations from the op directory."""
        operations = []
        
        if not self.op_dir.exists():
            print(f"Warning: Operation directory not found: {self.op_dir}")
            return operations
        
        # Get all Python files in op directory (except __init__.py)
        op_files = [f for f in self.op_dir.glob("*.py") if f.name != "__init__.py"]
        
        for op_file in op_files:
            try:
                file_ops = self._parse_file(op_file)
                operations.extend(file_ops)
            except Exception as e:
                print(f"Warning: Failed to parse {op_file.name}: {e}")
                continue
        
        return operations
    
    def _parse_file(self, file_path: Path) -> List[DiscoveredOperation]:
        """Parse a single Python file to extract operations."""
        operations = []
        
        # Read and parse the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"Warning: Syntax error in {file_path.name}: {e}")
            return operations
        
        # Determine category from file name
        category = self.FILE_TO_CATEGORY.get(file_path.name, "Other Operations")
        
        # Find all function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Only process functions that start with uppercase (operation functions)
                if node.name[0].isupper():
                    op = self._parse_function(node, file_path, category)
                    if op:
                        operations.append(op)
        
        return operations
    
    def _parse_function(self, func_node: ast.FunctionDef, file_path: Path, category: str) -> Optional[DiscoveredOperation]:
        """Parse a function node to extract operation information."""
        # Get docstring
        docstring = ast.get_docstring(func_node) or ""
        
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
            category=category
        )
    
    def _get_function_signature(self, func_node: ast.FunctionDef) -> str:
        """Get function signature as string."""
        args = []
        for arg in func_node.args.args:
            arg_str = arg.arg
            # Try to get annotation if available
            if arg.annotation:
                try:
                    if hasattr(ast, 'unparse'):
                        annotation = ast.unparse(arg.annotation)
                    else:
                        # Fallback for Python < 3.9
                        annotation = self._ast_to_string(arg.annotation)
                    arg_str += f": {annotation}"
                except:
                    pass
            args.append(arg_str)
        
        # Add defaults
        defaults_start = len(func_node.args.args) - len(func_node.args.defaults)
        for i, default in enumerate(func_node.args.defaults):
            if i >= defaults_start:
                try:
                    default_idx = i - defaults_start
                    if default_idx < len(func_node.args.defaults):
                        default = func_node.args.defaults[default_idx]
                        if hasattr(ast, 'unparse'):
                            default_str = ast.unparse(default)
                        elif isinstance(default, ast.Constant):
                            default_str = repr(default.value)
                        else:
                            default_str = repr(default)
                        args[i] += f"={default_str}"
                except:
                    pass
        
        # Get return type
        return_annotation = ""
        if func_node.returns:
            try:
                if hasattr(ast, 'unparse'):
                    return_annotation = f" -> {ast.unparse(func_node.returns)}"
                else:
                    return_annotation = f" -> {self._ast_to_string(func_node.returns)}"
            except:
                pass
        
        # Use forge.op. prefix for consistency with documentation
        return f"forge.op.{func_node.name}({', '.join(args)}){return_annotation}"
    
    def _parse_parameters(self, func_node: ast.FunctionDef, docstring: str) -> List[Dict[str, str]]:
        """Parse function parameters from signature and docstring."""
        params = []
        
        # Extract parameter descriptions from docstring
        param_descriptions = self._extract_parameter_descriptions(docstring)
        
        defaults_start = len(func_node.args.args) - len(func_node.args.defaults)
        
        for i, arg in enumerate(func_node.args.args):
            param_info = {
                "name": arg.arg,
                "type": "",
                "default": None,
                "description": param_descriptions.get(arg.arg, "")
            }
            
            # Get type annotation
            if arg.annotation:
                try:
                    if hasattr(ast, 'unparse'):
                        param_info["type"] = ast.unparse(arg.annotation)
                    else:
                        param_info["type"] = self._ast_to_string(arg.annotation)
                except:
                    pass
            
            # Get default value
            if i >= defaults_start:
                default_idx = i - defaults_start
                if default_idx < len(func_node.args.defaults):
                    try:
                        default = func_node.args.defaults[default_idx]
                        if hasattr(ast, 'unparse'):
                            param_info["default"] = ast.unparse(default)
                        elif isinstance(default, ast.Constant):
                            param_info["default"] = repr(default.value)
                        else:
                            param_info["default"] = repr(default)
                    except:
                        pass
            
            params.append(param_info)
        
        return params
    
    def _extract_parameter_descriptions(self, docstring: str) -> Dict[str, str]:
        """Extract parameter descriptions from NumPy-style docstring."""
        descriptions = {}
        
        if "Parameters" not in docstring:
            return descriptions
        
        # Find Parameters section
        lines = docstring.split('\n')
        in_params = False
        current_param = None
        current_desc = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check for Parameters section start
            if stripped.startswith("Parameters"):
                in_params = True
                # Skip the "----------" line that usually follows
                continue
            
            # Check for section end markers
            if in_params and (stripped.startswith("---") or 
                             (stripped.startswith("Returns") and i > 0 and lines[i-1].strip() == "")):
                if current_param:
                    descriptions[current_param] = ' '.join(current_desc).strip()
                if stripped.startswith("Returns"):
                    break
                continue
            
            if in_params:
                # Check if this is a parameter name line
                # Format: "name : type" or just "name:" at start of line (not indented)
                if stripped and not line.startswith(' ') and not line.startswith('\t'):
                    # This might be a parameter name
                    if ':' in stripped:
                        # Save previous parameter
                        if current_param:
                            descriptions[current_param] = ' '.join(current_desc).strip()
                        
                        # Start new parameter
                        parts = stripped.split(':', 1)
                        param_name = parts[0].strip()
                        # Only treat as parameter if it looks like a name (not a section header)
                        if param_name and not param_name.isupper() and len(param_name) < 50:
                            current_param = param_name
                            current_desc = [parts[1].strip()] if len(parts) > 1 and parts[1].strip() else []
                        else:
                            current_param = None
                    elif current_param is None:
                        # Might be a section header, skip
                        continue
                elif current_param and stripped:
                    # Continuation of description (indented)
                    if line.startswith(' ') or line.startswith('\t'):
                        current_desc.append(stripped)
                    elif stripped.startswith("Returns"):
                        # End of Parameters section
                        if current_param:
                            descriptions[current_param] = ' '.join(current_desc).strip()
                        break
        
        # Save last parameter
        if current_param:
            descriptions[current_param] = ' '.join(current_desc).strip()
        
        return descriptions
    
    def _extract_return_description(self, docstring: str) -> str:
        """Extract return description from docstring."""
        if "Returns" not in docstring:
            return ""
        
        lines = docstring.split('\n')
        in_returns = False
        return_lines = []
        
        for line in lines:
            if line.strip().startswith("Returns"):
                in_returns = True
                continue
            if in_returns:
                if line.strip().startswith("---") or (line.strip() and not line.startswith(' ')):
                    break
                if line.strip():
                    return_lines.append(line.strip())
        
        return ' '.join(return_lines).strip()
    
    def _get_return_type(self, func_node: ast.FunctionDef) -> str:
        """Get return type annotation."""
        if func_node.returns:
            try:
                if hasattr(ast, 'unparse'):
                    return ast.unparse(func_node.returns)
                else:
                    return self._ast_to_string(func_node.returns)
            except:
                return ""
        return ""


def discover_operations(project_root: Path) -> List[DiscoveredOperation]:
    """
    Discover all operations from the codebase.
    
    Args:
        project_root: Path to the project root directory
        
    Returns:
        List of discovered operations
    """
    discoverer = OperationDiscoverer(project_root)
    return discoverer.discover_all_operations()


if __name__ == "__main__":
    # Test the discoverer
    project_root = Path(__file__).parent.parent
    operations = discover_operations(project_root)
    
    print(f"Discovered {len(operations)} operations:")
    for op in operations:
        print(f"  - {op.name} ({op.category}) from {op.file_name}")

