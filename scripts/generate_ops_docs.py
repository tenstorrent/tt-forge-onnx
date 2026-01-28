# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Script to generate operation documentation from Forge operation source files.

This script:
1. Discovers operations from forge/forge/op/*.py files
2. Parses docstrings to extract documentation
3. Generates markdown documentation pages
4. Creates an index page with operations grouped by category
5. Cleans up stale documentation files for removed operations

All operation information is sourced from the actual Python source files.
Enhanced descriptions (e.g., mathematical definitions) are loaded from
scripts/operation_enhancements.json.

Usage:
    python scripts/generate_ops_docs.py [options]

Options:
    --op-dir PATH        Source directory for operations (default: forge/forge/op/)
    --output-dir PATH    Output directory for operation docs (default: docs/src/operations/)
    --index-file PATH    Output path for index page (default: docs/src/operations.md)
    --enhancements PATH  Path to enhancements JSON file (default: scripts/operation_enhancements.json)
    --no-cleanup         Skip cleanup of stale documentation files
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
from pathlib import Path


class DocumentationGenerationError(Exception):
    """Raised when documentation generation fails."""


@dataclass
class Operand:
    """Represents an operand (input/output) of an operation."""

    name: str
    description: str
    type: str = "Tensor"


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
    short_name: str  # e.g., "Abs"
    category: str
    description: str
    detailed_description: str = ""
    mathematical_definition: str = ""
    operands: List[Operand] = field(default_factory=list)
    results: List[Operand] = field(default_factory=list)
    attributes: List[Attribute] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    signature: str = ""
    related_operations: List[Dict[str, str]] = field(default_factory=list)


def sanitize_filename(name: str) -> str:
    """
    Convert operation name to a clean filename.

    Examples:
        "forge.op.Abs" -> "abs"
        "Resize2d" -> "resize2d"
    """
    # Remove 'forge.op.' prefix if present
    name = name.replace("forge.op.", "")
    return name.lower()


def load_enhancements(enhancements_path: Path) -> Dict:
    """
    Load operation enhancements from JSON file.

    The enhancements file supports the following fields per operation:
    - description: Override/supplement the operation overview
    - parameters: Dict mapping parameter names to enhanced descriptions
    - mathematical_definition: Mathematical formula for the operation
    - related_operations: List of related operations with descriptions
    """
    if enhancements_path.exists():
        try:
            with open(enhancements_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("operations", {})
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load enhancements: {e}")

    return {}


def generate_operation_page(op: Operation, output_dir: Path, enhancements: Dict) -> None:
    """Generate a markdown page for a single operation."""
    filename = sanitize_filename(op.name)
    filepath = output_dir / f"{filename}.md"

    # Get enhancements for this operation
    op_enhancements = enhancements.get(op.short_name, {})
    param_enhancements = op_enhancements.get("parameters", {})

    # Build content in a list, then join and normalize at the end
    lines = []

    # Title
    lines.append(f"# {op.name}")
    lines.append("")

    # Overview section
    lines.append("## Overview")
    lines.append("")
    # Use enhanced description if available, otherwise use docstring description
    if op_enhancements.get("description"):
        overview = op_enhancements["description"]
    else:
        overview = op.description
        if op.detailed_description:
            if overview and not overview.endswith("."):
                overview += "."
            overview += "\n\n" + op.detailed_description
    lines.append(overview)
    lines.append("")

    # Function signature
    lines.append("## Function Signature")
    lines.append("")
    lines.append("```python")
    if op.signature:
        sig = op.signature
        # Format multi-line for readability
        if len(sig) > 80 and "(" in sig and ")" in sig:
            sig = _format_signature(sig)
        lines.append(sig)
    else:
        lines.append(f"{op.name}(...)")
    lines.append("```")
    lines.append("")

    # Parameters section
    if op.operands or op.attributes:
        lines.append("## Parameters")
        lines.append("")

        # Write name parameter first
        name_attr = next((a for a in op.attributes if a.name == "name"), None)
        if name_attr:
            # Check for enhanced description first
            desc = (
                param_enhancements.get("name")
                or name_attr.description
                or "Name identifier for this operation in the computation graph."
            )
            lines.append(f"- **name** (`str`): {desc}")
            lines.append("")

        # Write operands (tensor inputs)
        for operand in op.operands:
            if operand.name not in ("output", "result"):
                type_str = operand.type or "Tensor"
                # Check for enhanced description first
                desc = param_enhancements.get(operand.name) or operand.description or f"{operand.name} tensor"
                lines.append(f"- **{operand.name}** (`{type_str}`): {desc}")
                lines.append("")

        # Write other attributes
        for attr in op.attributes:
            if attr.name != "name":
                type_str = attr.mlir_type or "Any"
                default_str = f", default: `{attr.default}`" if attr.default else ""
                # Check for enhanced description first
                desc = param_enhancements.get(attr.name) or attr.description or f"{attr.name} parameter"
                lines.append(f"- **{attr.name}** (`{type_str}`{default_str}): {desc}")
                lines.append("")

    # Returns section
    if op.results:
        lines.append("## Returns")
        lines.append("")
        for result in op.results:
            type_str = result.type or "Tensor"
            desc = result.description or "Output tensor"
            lines.append(f"- **{result.name}** (`{type_str}`): {desc}")
            lines.append("")

    # Mathematical Definition (from enhancements or docstring)
    math_def = op_enhancements.get("mathematical_definition") or op.mathematical_definition
    if math_def:
        lines.append("## Mathematical Definition")
        lines.append("")
        lines.append(math_def)
        lines.append("")

    # Notes
    if op.notes:
        lines.append("## Notes")
        lines.append("")
        for note in op.notes:
            lines.append(f"- {note}")
        lines.append("")

    # Related Operations (from enhancements)
    related_ops = op_enhancements.get("related_operations", []) or op.related_operations
    if related_ops:
        lines.append("## Related Operations")
        lines.append("")
        for related in related_ops:
            related_name = related.get("name", "")
            related_desc = related.get("description", "")
            related_file = sanitize_filename(related_name)
            lines.append(f"- [forge.op.{related_name}](./{related_file}.md): {related_desc}")

    # Join lines and ensure exactly one trailing newline
    content = "\n".join(lines).rstrip() + "\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def _format_signature(sig: str) -> str:
    """Format a long signature for readability."""
    if "(" not in sig or ")" not in sig:
        return sig

    func_name = sig.split("(")[0]
    rest = sig[len(func_name) :]

    # Find closing paren
    paren_count = 0
    split_idx = -1
    for i, char in enumerate(rest):
        if char == "(":
            paren_count += 1
        elif char == ")":
            paren_count -= 1
            if paren_count == 0:
                split_idx = i
                break

    if split_idx <= 0:
        return sig

    params_str = rest[1:split_idx]
    return_part = rest[split_idx + 1 :]

    # Split parameters respecting nested brackets
    params = []
    current = ""
    depth = 0
    for char in params_str:
        current += char
        if char in "[(":
            depth += 1
        elif char in "])":
            depth -= 1
        elif char == "," and depth == 0:
            params.append(current[:-1].strip())
            current = ""
    if current.strip():
        params.append(current.strip())

    # Format with indentation
    formatted = f"{func_name}(\n"
    for i, param in enumerate(params):
        comma = "," if i < len(params) - 1 else ""
        formatted += f"    {param}{comma}\n"
    formatted += f"){return_part}"

    return formatted


def generate_index_page(operations: List[Operation], output_dir: Path) -> None:
    """Generate the main index page with operations grouped by categories."""
    filepath = output_dir / "operations.md"

    # Group operations by category
    ops_by_category: Dict[str, List[Operation]] = {}
    for op in operations:
        if op.category not in ops_by_category:
            ops_by_category[op.category] = []
        ops_by_category[op.category].append(op)

    # Category order
    category_order = [
        "Elementwise Operations",
        "Convolution Operations",
        "Pooling Operations",
        "Normalization Operations",
        "Tensor Manipulation",
        "Reduction Operations",
        "Linear Operations",
        "Activation Functions",
        "Memory Operations",
        "Other Operations",
    ]

    sorted_categories = [c for c in category_order if c in ops_by_category]
    for c in sorted(ops_by_category.keys()):
        if c not in sorted_categories:
            sorted_categories.append(c)

    category_descriptions = {
        "Elementwise Operations": "Mathematical operations applied element-wise",
        "Convolution Operations": "Convolution and related transformations",
        "Pooling Operations": "Pooling and downsampling operations",
        "Normalization Operations": "Batch and layer normalization",
        "Tensor Manipulation": "Reshaping, slicing, and tensor operations",
        "Reduction Operations": "Aggregation and reduction operations",
        "Linear Operations": "Matrix multiplication and linear transformations",
        "Activation Functions": "Non-linear activation functions",
        "Memory Operations": "Cache and memory management operations",
        "Other Operations": "Miscellaneous operations",
    }

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Forge Operations Reference\n\n")
        f.write(
            "Welcome to the Forge Operations Reference. This page provides a comprehensive guide to all supported operations in the Forge framework.\n\n"
        )

        # Overview
        f.write("## Overview\n\n")
        f.write(
            "Forge operations are organized into logical categories based on their functionality. Each operation is documented with detailed information including function signatures, parameters, examples, and usage notes.\n\n"
        )

        # Quick Navigation
        f.write("## Quick Navigation\n\n")
        for category in sorted_categories:
            anchor = category.lower().replace(" ", "-")
            desc = category_descriptions.get(category, "Operations in this category")
            f.write(f"- [{category}](#{anchor}) - {desc}\n")
        f.write("\n---\n\n")

        # Category sections
        for category in sorted_categories:
            f.write(f"## {category}\n\n")

            desc = category_descriptions.get(category, "")
            if desc:
                f.write(f"{desc}.\n\n")

            # Table
            ops = sorted(ops_by_category[category], key=lambda x: x.short_name)
            if ops:
                f.write("| Operation | Description | Link |\n")
                f.write("|-----------|-------------|------|\n")
                for op in ops:
                    filename = sanitize_filename(op.name)
                    short_desc = op.description[:80] + "..." if len(op.description) > 80 else op.description
                    f.write(f"| **{op.short_name}** | {short_desc} | [{op.name}](./operations/{filename}.md) |\n")
                f.write("\n")

        # Documentation Structure
        f.write("---\n\n")
        f.write("## Documentation Structure\n\n")
        f.write("Each operation documentation page includes:\n\n")
        f.write("- **Overview**: Brief description of what the operation does\n")
        f.write("- **Function Signature**: Python API signature with type hints\n")
        f.write("- **Parameters**: Detailed parameter descriptions with types and defaults\n")
        f.write("- **Returns**: Return value description\n")
        f.write("- **Mathematical Definition**: Mathematical formula (where applicable)\n")
        f.write("- **Related Operations**: Links to related operations\n\n")
        f.write("---\n\n")
        f.write(
            "*This documentation is automatically generated from operation definitions in `forge/forge/op/*.py`. For the most up-to-date information, refer to the source code.*\n"
        )


def cleanup_stale_files(output_dir: Path, valid_operations: Set[str]) -> int:
    """
    Remove documentation files for operations that no longer exist.

    Args:
        output_dir: Directory containing operation markdown files
        valid_operations: Set of valid operation short names (lowercase)

    Returns:
        Number of stale files removed
    """
    removed_count = 0

    if not output_dir.exists():
        return removed_count

    for md_file in output_dir.glob("*.md"):
        # Extract operation name from filename (e.g., "abs.md" -> "abs")
        op_name = md_file.stem.lower()

        # Skip non-operation files
        if op_name in ("readme", "index", "operations"):
            continue

        # Check if this operation still exists
        if op_name not in valid_operations:
            try:
                md_file.unlink()
                print(f"      Removed stale file: {md_file.name}")
                removed_count += 1
            except OSError as e:
                print(f"      Warning: Could not remove {md_file.name}: {e}")

    return removed_count


def convert_discovered_to_operation(discovered) -> Operation:
    """Convert a DiscoveredOperation to an Operation."""
    docstring = discovered.docstring.strip()
    lines = docstring.split("\n")

    # Extract description (first paragraph before Parameters)
    desc_lines = []
    detailed_lines = []
    in_detailed = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("Parameters", "Returns", "Mathematical", "See Also", "Notes", "Examples")):
            break
        if stripped.startswith("---"):
            continue
        if stripped:
            if not desc_lines:
                desc_lines.append(stripped)
            else:
                in_detailed = True
                detailed_lines.append(stripped)
        elif desc_lines and not in_detailed:
            in_detailed = True

    short_desc = " ".join(desc_lines)
    detailed_desc = "\n\n".join(" ".join(detailed_lines[i : i + 1]) for i in range(len(detailed_lines)))

    # Extract mathematical definition from docstring if present
    math_def = ""
    if "Mathematical Definition" in docstring:
        in_math = False
        math_lines = []
        for line in lines:
            if "Mathematical Definition" in line:
                in_math = True
                continue
            if in_math:
                if line.strip().startswith("---"):
                    continue
                if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                    if line.strip().startswith(("See Also", "Notes", "Examples", "Parameters", "Returns")):
                        break
                if line.strip():
                    math_lines.append(line.strip())
        math_def = "\n".join(math_lines)

    # Extract related operations from docstring
    related_ops = []
    if "See Also" in docstring:
        in_see_also = False
        for line in lines:
            if "See Also" in line:
                in_see_also = True
                continue
            if in_see_also:
                if line.strip().startswith("---"):
                    continue
                if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                    if line.strip().startswith(("Notes", "Examples", "Parameters", "Returns", "Mathematical")):
                        break
                if "forge.op." in line:
                    parts = line.strip().split(":")
                    if len(parts) >= 2:
                        op_name = parts[0].replace("forge.op.", "").strip()
                        op_desc = parts[1].strip()
                        related_ops.append({"name": op_name, "description": op_desc})

    # Convert parameters to operands and attributes
    operands = []
    attributes = []

    for param in discovered.parameters:
        param_type = param.get("type", "").lower()
        is_tensor = "tensor" in param_type

        desc = param.get("description", "").strip()
        desc = " ".join(desc.split())  # Normalize whitespace

        if is_tensor:
            operands.append(
                Operand(
                    name=param["name"], type=param.get("type", "Tensor"), description=desc or f"{param['name']} tensor"
                )
            )
        else:
            attributes.append(
                Attribute(
                    name=param["name"],
                    mlir_type=param.get("type", "Any"),
                    description=desc or f"{param['name']} parameter",
                    default=param.get("default"),
                )
            )

    # Add result
    results = []
    if discovered.return_type:
        results.append(Operand("result", discovered.return_description or "Output tensor", discovered.return_type))

    return Operation(
        name=f"forge.op.{discovered.name}",
        short_name=discovered.name,
        category=discovered.category,
        description=short_desc,
        detailed_description=detailed_desc,
        mathematical_definition=math_def,
        operands=operands,
        results=results,
        attributes=attributes,
        signature=discovered.signature,
        related_operations=related_ops,
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate documentation for Forge operations from source files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
      Generate documentation with default paths.

  %(prog)s --op-dir forge/forge/op --output-dir docs/src/operations
      Generate documentation with custom paths.

  %(prog)s --no-cleanup
      Generate documentation without removing stale files.
""",
    )

    parser.add_argument(
        "--op-dir", type=Path, default=None, help="Source directory for operations (default: forge/forge/op/)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for operation docs (default: docs/src/operations/)",
    )
    parser.add_argument(
        "--index-file", type=Path, default=None, help="Output path for index page (default: docs/src/operations.md)"
    )
    parser.add_argument(
        "--enhancements",
        type=Path,
        default=None,
        help="Path to enhancements JSON file (default: scripts/operation_enhancements.json)",
    )
    parser.add_argument("--no-cleanup", action="store_true", help="Skip cleanup of stale documentation files")

    return parser.parse_args()


def main():
    """Main function to generate all documentation."""
    args = parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Set paths from args or use defaults
    op_dir = args.op_dir or (project_root / "forge" / "forge" / "op")
    ops_docs_dir = args.output_dir or (project_root / "docs" / "src" / "operations")
    index_file = args.index_file or (project_root / "docs" / "src" / "operations.md")
    enhancements_path = args.enhancements or (script_dir / "operation_enhancements.json")
    do_cleanup = not args.no_cleanup

    # Create output directory
    ops_docs_dir.mkdir(parents=True, exist_ok=True)

    # Import and run discovery
    print("=" * 60)
    print("Forge Operations Documentation Generator")
    print("=" * 60)

    sys.path.insert(0, str(script_dir))

    print(f"\n[1/5] Discovering operations from {op_dir}...")
    try:
        from discover_operations import discover_operations

        discovered_ops = discover_operations(project_root, op_dir)
        print(f"      Discovered {len(discovered_ops)} operations")
    except ImportError as e:
        print(f"\nERROR: Could not import discover_operations: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Operation discovery failed:\n{e}", file=sys.stderr)
        sys.exit(1)

    # Load enhancements
    print(f"\n[2/5] Loading operation enhancements from {enhancements_path.name}...")
    enhancements = load_enhancements(enhancements_path)
    print(f"      Loaded enhancements for {len(enhancements)} operations")

    # Convert discovered operations
    print("\n[3/5] Converting and generating operation pages...")
    operations = []
    errors = []

    for discovered in discovered_ops:
        try:
            op = convert_discovered_to_operation(discovered)
            operations.append(op)
            generate_operation_page(op, ops_docs_dir, enhancements)
            print(f"      [OK] {op.short_name}.md")
        except Exception as e:
            errors.append(f"{discovered.name}: {e}")
            print(f"      [FAIL] {discovered.name}: {e}")

    if errors:
        print(f"\n      Warning: {len(errors)} operation(s) had conversion errors")

    # Cleanup stale files
    if do_cleanup:
        print("\n[4/5] Cleaning up stale documentation files...")
        valid_ops = {sanitize_filename(op.name) for op in operations}
        removed = cleanup_stale_files(ops_docs_dir, valid_ops)
        if removed > 0:
            print(f"      Removed {removed} stale file(s)")
        else:
            print("      No stale files found")
    else:
        print("\n[4/5] Skipping cleanup (--no-cleanup specified)")

    # Generate index page
    print("\n[5/5] Generating index page...")
    generate_index_page(operations, index_file.parent)
    print(f"      [OK] {index_file.name}")

    # Summary
    print("\n" + "=" * 60)
    print("Documentation generation complete!")
    print(f"  Total operations: {len(operations)}")
    print(f"  Output directory: {ops_docs_dir}")
    print(f"  Index page: {index_file}")

    if errors:
        print(f"\n  Errors: {len(errors)}")
        for err in errors:
            print(f"    - {err}")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
