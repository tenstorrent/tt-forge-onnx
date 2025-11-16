#!/usr/bin/env python3
"""
Script to generate operation documentation in PyTorch-style format.

This script generates:
1. An index page with operations grouped by categories
2. Individual pages for each operation with detailed information
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path
import json


@dataclass
class Operand:
    """Represents an operand (input/output) of an operation."""
    name: str
    description: str
    type: str = "ranked tensor of any type values"


@dataclass
class Attribute:
    """Represents an attribute of an operation."""
    name: str
    mlir_type: str
    description: str
    default: Optional[str] = None


@dataclass
class Operation:
    """Represents a complete operation definition."""
    name: str  # e.g., "forge.op.Abs"
    short_name: str  # e.g., "abs"
    category: str
    description: str
    detailed_description: str = ""
    mathematical_definition: str = ""
    traits: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    operands: List[Operand] = field(default_factory=list)
    results: List[Operand] = field(default_factory=list)
    attributes: List[Attribute] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    signature: str = ""  # Python function signature


# Operation categories mapping
CATEGORIES = {
    "Elementwise Operations": [
        "abs", "add", "atan", "atan2", "bitwise_and", "bitwise_not", "bitwise_or", "bitwise_xor",
        "cbrt", "ceil", "cos", "div", "eq", "erf", "erfc", "exp", "expm1", "floor", "ge", "gelu",
        "gt", "hardsigmoid", "isfinite", "le", "leaky_relu", "log", "log1p", "logical_and",
        "logical_left_shift", "logical_not", "logical_or", "logical_right_shift", "logical_xor",
        "lt", "maximum", "minimum", "multiply", "neg", "ne", "pow", "reciprocal", "relu", "relu6",
        "remainder", "rsqrt", "sigmoid", "sign", "silu", "sin", "sqrt", "subtract", "tan", "tanh"
    ],
    "Convolution Functions": [
        "conv2d", "conv_transpose2d", "convolution"
    ],
    "Pooling Functions": [
        "avg_pool2d", "global_avg_pool2d", "max_pool2d", "max_pool2d_with_indices", "pooling"
    ],
    "Normalization Functions": [
        "batch_norm_inference", "batch_norm_training", "layer_norm", "rms_norm"
    ],
    "Tensor Manipulation": [
        "broadcast", "concat", "concatenate_heads", "gather", "index", "index_select", "pad",
        "permute", "reshape", "reverse", "slice_dynamic", "slice_static", "squeeze", "transpose",
        "unsqueeze", "upsample2d"
    ],
    "Reduction Operations": [
        "argmax", "cumsum", "max", "mean", "min", "prod", "reduce_and", "reduce_or", "sum"
    ],
    "Linear Functions": [
        "linear", "matmul", "dot_general"
    ],
    "Attention Mechanisms": [
        "scaled_dot_product_attention", "scaled_dot_product_attention_decode",
        "split_query_key_value_and_split_heads"
    ],
    "Embedding Functions": [
        "embedding", "embedding_backward"
    ],
    "Memory Operations": [
        "alloc", "dealloc", "empty", "fill_cache", "paged_update_cache", "update_cache"
    ],
    "Creation Operations": [
        "arange", "constant", "full", "ones", "rand", "zeros"
    ],
    "Quantization Operations": [
        "quantize", "quantize_unrolled", "dequantize", "dequantize_unrolled",
        "requantize", "requantize_unrolled"
    ],
    "Conditional Operations": [
        "where", "clamp_scalar", "clamp_tensor"
    ],
    "Collective Operations": [
        "all_gather", "all_reduce", "all_to_all", "collective_broadcast", "collective_permute",
        "reduce_scatter", "mesh_shard"
    ],
    "Other Operations": [
        "softmax", "sort", "get_dimension_size", "to_layout", "ttnn_metal_layout_cast",
        "typecast"
    ]
}


def parse_operation_from_text(text: str) -> Optional[Operation]:
    """
    Parse operation information from the provided text format.
    This is a simplified parser - in a real scenario, you'd parse from MLIR or structured data.
    """
    # This is a placeholder - you would implement actual parsing logic here
    # For now, we'll use the structured data approach
    return None


def sanitize_filename(name: str) -> str:
    """Convert operation name to a valid filename."""
    # Remove 'ttir.' prefix if present
    name = name.replace("ttir.", "")
    # Replace dots and special chars with underscores
    name = re.sub(r'[^\w-]', '_', name)
    return name.lower()


def get_category_for_op(op_name: str) -> str:
    """Get the category for an operation based on its name."""
    short_name = op_name.replace("ttir.", "").lower()
    for category, ops in CATEGORIES.items():
        if short_name in ops:
            return category
    return "Other Operations"


def generate_operation_page(op: Operation, output_dir: Path) -> None:
    """Generate a markdown page for a single operation."""
    filename = sanitize_filename(op.name)
    filepath = output_dir / f"{filename}.md"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # Title
        f.write(f"# {op.name}\n\n")
        
        # Short description
        f.write(f"{op.description}\n\n")
        
        # Detailed description
        if op.detailed_description:
            f.write(f"{op.detailed_description}\n\n")
        
        # Function signature (PyTorch-style)
        f.write("## Function Signature\n\n")
        f.write("```python\n")
        if op.signature:
            # Use the actual signature from the source code
            f.write(f"{op.signature}\n")
        else:
            # Fallback: construct signature from operands and attributes
            f.write(f"{op.name}(")
            params = []
            
            # Add operands first (required parameters)
            for operand in op.operands:
                if operand.name != "output":  # output is typically the last parameter
                    params.append(operand.name)
            
            # Add attributes (optional parameters)
            for attr in op.attributes:
                param_str = f"{attr.name}"
                if attr.default:
                    param_str += f"={attr.default}"
                params.append(param_str)
            
            # Add output operand last if present
            for operand in op.operands:
                if operand.name == "output":
                    params.append(operand.name)
            
            f.write(", ".join(params) if params else "")
            f.write(")\n")
        f.write("```\n\n")
        
        # Parameters (PyTorch-style)
        if op.operands or op.attributes:
            f.write("## Parameters\n\n")
            
            if op.operands:
                for operand in op.operands:
                    if operand.name != "output":  # Output is shown separately
                        type_str = operand.type if operand.type else "Tensor"
                        f.write(f"- **{operand.name}** ({type_str}): {operand.description}\n")
                if op.attributes:  # Only add blank line if there are attributes too
                    f.write("\n")
            
            if op.attributes:
                for attr in op.attributes:
                    default_str = f" (default: {attr.default})" if attr.default else ""
                    type_str = attr.mlir_type if attr.mlir_type else "Any"
                    f.write(f"- **{attr.name}** ({type_str}){default_str}: {attr.description}\n")
                f.write("\n")
        
        # Returns (PyTorch-style)
        if op.results:
            f.write("## Returns\n\n")
            for result in op.results:
                type_str = result.type if result.type else "Tensor"
                # Clean up return description (remove redundant "Output tensor: Tensor" patterns)
                desc = result.description
                if desc:
                    # Remove patterns like "Output tensor: Tensor" or "Forge tensor"
                    if ":" in desc:
                        parts = desc.split(":", 1)
                        if parts[0].strip().lower() in ["output tensor", "tensor", "forge tensor"]:
                            desc = parts[1].strip() if len(parts) > 1 else ""
                    # Remove standalone type words at the start
                    if desc.split()[0] in ["Tensor", "Output", "Forge"]:
                        desc_words = desc.split()
                        if len(desc_words) > 1:
                            desc = ' '.join(desc_words[1:])
                desc = desc or "Output tensor"
                f.write(f"- **{result.name}** ({type_str}): {desc}\n")
            f.write("\n")
        
        # Mathematical definition
        if op.mathematical_definition:
            f.write("## Mathematical Definition\n\n")
            f.write(f"{op.mathematical_definition}\n\n")
        
        # Examples
        if op.examples:
            f.write("## Examples\n\n")
            for example in op.examples:
                f.write(f"```python\n{example}\n```\n\n")
        
        # Notes
        if op.notes:
            f.write("## Notes\n\n")
            for note in op.notes:
                f.write(f"{note}\n\n")
        
        # Traits and Interfaces (if needed)
        if op.traits or op.interfaces:
            f.write("## Implementation Details\n\n")
            if op.traits:
                f.write("**Traits:** " + ", ".join(op.traits) + "\n\n")
            if op.interfaces:
                f.write("**Interfaces:** " + ", ".join(op.interfaces) + "\n\n")


def generate_index_page(operations: List[Operation], output_dir: Path) -> None:
    """Generate the main index page with operations grouped by categories."""
    filepath = output_dir / "operations.md"
    
    # Group operations by category
    ops_by_category: Dict[str, List[Operation]] = {}
    for op in operations:
        category = get_category_for_op(op.name)
        if category not in ops_by_category:
            ops_by_category[category] = []
        ops_by_category[category].append(op)
    
    # Sort categories
    sorted_categories = sorted(ops_by_category.keys())
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# Operations Reference\n\n")
        f.write("This page provides a comprehensive reference for all supported operations.\n\n")
        f.write("Operations are organized by category. Click on an operation name to view detailed documentation.\n\n")
        
        for category in sorted_categories:
            f.write(f"## {category}\n\n")
            ops = sorted(ops_by_category[category], key=lambda x: x.short_name)
            
            for op in ops:
                filename = sanitize_filename(op.name)
                f.write(f"- [{op.name}](./operations/{filename}.md)\n")
                f.write(f"  {op.description}\n\n")
        
        f.write("\n---\n\n")
        f.write("*This documentation is automatically generated from operation definitions.*\n")


def convert_discovered_to_operation(discovered) -> Operation:
    """Convert a DiscoveredOperation to an Operation."""
    # Extract short description from docstring (first line before Parameters/Returns)
    docstring = discovered.docstring.strip()
    
    # Find the first meaningful line (skip empty lines, get text before Parameters/Returns)
    lines = docstring.split('\n')
    short_desc = discovered.name  # Default
    detailed_desc = ""
    
    # Collect all description lines before Parameters/Returns
    desc_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Stop at Parameters or Returns section
        if stripped.startswith("Parameters") or stripped.startswith("Returns"):
            break
        if stripped and not stripped.startswith("---"):
            desc_lines.append(stripped)
    
    # Use first line as short description, rest as detailed
    if desc_lines:
        short_desc = desc_lines[0]
        if len(desc_lines) > 1:
            detailed_desc = "\n\n".join(desc_lines[1:])
    
    # Clean up descriptions
    short_desc = short_desc.strip()
    detailed_desc = detailed_desc.strip()
    
    # If description is too short or seems wrong, try to infer from function name
    if len(short_desc) < 3 or short_desc.lower() in ["sigmoid", "tm", "op"]:
        # Try to create a better description from function name
        op_name_lower = discovered.name.lower()
        if op_name_lower == "abs":
            short_desc = "Elementwise absolute value operation"
        elif op_name_lower in ["add", "subtract", "multiply", "divide"]:
            short_desc = f"Elementwise {op_name_lower} of two tensors"
        elif op_name_lower == "relu":
            short_desc = "Rectified Linear Unit (ReLU) activation"
        elif op_name_lower == "sigmoid":
            short_desc = "Sigmoid activation function"
        elif op_name_lower == "conv2d":
            short_desc = "2D convolution operation"
        elif op_name_lower == "matmul":
            short_desc = "Matrix multiplication operation"
        # Keep the original if we couldn't improve it
    
    # Convert parameters to operands and attributes
    operands = []
    attributes = []
    
    for param in discovered.parameters:
        # Skip 'name' parameter (it's for op naming, not a tensor)
        if param["name"] == "name":
            continue
        
        # Determine if it's an operand (Tensor) or attribute
        param_type = param.get("type", "").lower()
        is_tensor = "tensor" in param_type or "parameter" in param_type
        
        # Clean up description (remove extra whitespace, newlines, and redundant type prefixes)
        desc = param.get("description", "").strip()
        desc = ' '.join(desc.split())  # Normalize whitespace
        
        # Remove redundant type prefixes from description (e.g., "Tensor First operand" -> "First operand")
        type_words = ["Tensor", "Parameter", "Optional", "Union", "List", "Tuple", "int", "float", "bool", "str", "Tenor"]
        desc_words = desc.split()
        if desc_words and desc_words[0] in type_words:
            # Check if first word is a type (possibly followed by comma or colon)
            first_word = desc_words[0].rstrip(',:')
            if first_word in type_words and len(desc_words) > 1:
                # Remove the type word
                desc = ' '.join(desc_words[1:])
        
        # Remove redundant phrases like "optional Optional" or "Tensor Tensor"
        desc = desc.replace("optional Optional", "Optional").replace("Tensor Tensor", "Tensor")
        desc = desc.replace("Tenor, optional", "").replace("Tenor, Optional", "Optional").replace("Tenor,", "").strip()
        if desc.startswith(", "):
            desc = desc[2:]
        desc = desc.strip()
        
        if is_tensor:
            operands.append(Operand(
                name=param["name"],
                type=param.get("type", "Tensor"),
                description=desc or f"{param['name']} tensor"
            ))
        else:
            attributes.append(Attribute(
                name=param["name"],
                mlir_type=param.get("type", "Any"),
                description=desc or f"{param['name']} parameter",
                default=param.get("default")
            ))
    
    # Add output operand
    if discovered.return_type:
        operands.append(Operand(
            name="output",
            type=discovered.return_type,
            description=discovered.return_description or "Output tensor"
        ))
    
    # Convert name to ttir format (lowercase with underscores)
    short_name = discovered.name.lower()
    full_name = f"forge.op.{discovered.name}"
    
    return Operation(
        name=full_name,
        short_name=short_name,
        category=discovered.category,
        description=short_desc,
        detailed_description=detailed_desc,
        operands=operands,
        results=[Operand("result", discovered.return_type or "Tensor", "Output tensor")],
        attributes=attributes,
        examples=[],  # Could be extracted from docstring examples if present
        signature=discovered.signature  # Store the actual Python signature
    )


def load_operations_from_codebase(project_root: Path) -> List[Operation]:
    """
    Automatically discover operations from forge/forge/op/*.py files.
    This is the primary method - operations are discovered automatically.
    """
    try:
        import sys
        script_dir = Path(__file__).parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        from discover_operations import discover_operations, DiscoveredOperation
        
        print("Discovering operations from codebase...")
        discovered_ops = discover_operations(project_root)
        
        if not discovered_ops:
            print("Warning: No operations discovered. Falling back to manual data.")
            return load_operations_from_data(project_root)
        
        print(f"Discovered {len(discovered_ops)} operations from codebase")
        
        # Convert to Operation format
        operations = []
        for discovered in discovered_ops:
            try:
                op = convert_discovered_to_operation(discovered)
                operations.append(op)
            except Exception as e:
                print(f"Warning: Failed to convert {discovered.name}: {e}")
                continue
        
        return operations
        
    except ImportError as e:
        print(f"Warning: Could not import operation discoverer: {e}")
        print("Falling back to manual operation data.")
        return load_operations_from_data(project_root)


def load_operations_from_data(project_root: Path) -> List[Operation]:
    """
    Load operations from a structured data source (fallback method).
    This imports from ops_data.py which contains manually curated operation definitions.
    """
    import sys
    import importlib.util
    from pathlib import Path
    
    # Get the ops_data.py file path
    script_dir = Path(__file__).parent
    ops_data_path = script_dir / "ops_data.py"
    
    try:
        # Load the module directly from file
        spec = importlib.util.spec_from_file_location("ops_data", ops_data_path)
        if spec is None or spec.loader is None:
            raise ImportError("Could not load ops_data module")
        ops_data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ops_data)
        return ops_data.get_all_operations()
    except (ImportError, FileNotFoundError, AttributeError) as e:
        print(f"Warning: Could not import ops_data: {e}")
        print("Using minimal example operations.")
        # Fallback to minimal examples
        return [
            Operation(
                name="forge.op.Abs",
                short_name="abs",
                category="Elementwise Operations",
                description="Elementwise absolute value operation.",
                operands=[
                    Operand("operandA", "Tensor", "The input tensor"),
                    Operand("output", "Tensor", "The output tensor")
                ],
                results=[Operand("result", "Tensor", "The result tensor")]
            )
        ]


def main():
    """Main function to generate all documentation."""
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    docs_dir = project_root / "docs" / "src"
    ops_docs_dir = docs_dir / "operations"
    
    # Create operations directory if it doesn't exist
    ops_docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load operations - try automatic discovery first, fall back to manual data
    print("Loading operations...")
    operations = load_operations_from_codebase(project_root)
    print(f"Loaded {len(operations)} operations")
    
    # Generate individual operation pages
    print("Generating operation pages...")
    for op in operations:
        generate_operation_page(op, ops_docs_dir)
        print(f"  Generated: {op.name}")
    
    # Generate index page
    print("Generating index page...")
    generate_index_page(operations, docs_dir)
    print("  Generated: operations.md")
    
    print("\nDocumentation generation complete!")
    print(f"Output directory: {ops_docs_dir}")


if __name__ == "__main__":
    main()

