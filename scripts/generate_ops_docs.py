# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Script to generate operation documentation from Forge operation source files.

This script:
1. Discovers operations from forge/forge/op/*.py files
2. Parses docstrings to extract documentation
3. Generates a single docs/src/operations.md containing:
   - Category summary tables (with in-page anchor links)
   - Full per-operation detail sections (signature, parameters, returns, etc.)
4. Removes any stale individual files left over in docs/src/operations/

All operation information is sourced from the actual Python source files.
Enhanced descriptions (e.g., mathematical definitions) are loaded from
scripts/operation_enhancements.json.

Usage:
    python scripts/generate_ops_docs.py [options]

Options:
    --op-dir PATH        Source directory for operations (default: forge/forge/op/)
    --output-dir PATH    Legacy individual-file directory to clean up (default: docs/src/operations/)
    --index-file PATH    Output path for the combined page (default: docs/src/operations.md)
    --enhancements PATH  Path to enhancements JSON file (default: scripts/operation_enhancements.json)
    --no-cleanup         Skip cleanup of the legacy docs/src/operations/ directory
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict
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
    """
    if enhancements_path.exists():
        try:
            with open(enhancements_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("operations", {})
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load enhancements: {e}")

    return {}


def _build_operation_section(op: Operation, enhancements: Dict) -> str:
    """
    Build the markdown detail section for a single operation.

    The section heading uses only the short name (e.g. ``### Abs``) so that
    mdBook generates a clean in-page anchor (``#abs``) that can be linked from
    the category summary tables above.
    """
    op_enhancements = enhancements.get(op.short_name, {})
    param_enhancements = op_enhancements.get("parameters", {})

    lines = []

    # Section heading — short name gives anchor #abs, #relu, etc.
    lines.append(f"### {op.short_name}")
    lines.append("")

    # Overview
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
    lines.append("**Function Signature**")
    lines.append("")
    lines.append("```python")
    if op.signature:
        sig = op.signature
        if len(sig) > 80 and "(" in sig and ")" in sig:
            sig = _format_signature(sig)
        lines.append(sig)
    else:
        lines.append(f"{op.name}(...)")
    lines.append("```")
    lines.append("")

    # Parameters
    if op.operands or op.attributes:
        lines.append("**Parameters**")
        lines.append("")

        name_attr = next((a for a in op.attributes if a.name == "name"), None)
        if name_attr:
            desc = (
                param_enhancements.get("name")
                or name_attr.description
                or "Name identifier for this operation in the computation graph."
            )
            lines.append(f"- **name** (`str`): {desc}")
            lines.append("")

        for operand in op.operands:
            if operand.name not in ("output", "result"):
                type_str = operand.type or "Tensor"
                desc = param_enhancements.get(operand.name) or operand.description or f"{operand.name} tensor"
                lines.append(f"- **{operand.name}** (`{type_str}`): {desc}")
                lines.append("")

        for attr in op.attributes:
            if attr.name != "name":
                type_str = attr.mlir_type or "Any"
                default_str = f", default: `{attr.default}`" if attr.default else ""
                desc = param_enhancements.get(attr.name) or attr.description or f"{attr.name} parameter"
                lines.append(f"- **{attr.name}** (`{type_str}`{default_str}): {desc}")
                lines.append("")

    # Returns
    if op.results:
        lines.append("**Returns**")
        lines.append("")
        for result in op.results:
            type_str = result.type or "Tensor"
            desc = result.description or "Output tensor"
            lines.append(f"- **{result.name}** (`{type_str}`): {desc}")
            lines.append("")

    # Mathematical Definition
    math_def = op_enhancements.get("mathematical_definition") or op.mathematical_definition
    if math_def:
        lines.append("**Mathematical Definition**")
        lines.append("")
        lines.append(math_def)
        lines.append("")

    # Notes
    if op.notes:
        lines.append("**Notes**")
        lines.append("")
        for note in op.notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


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


def generate_operations_md(operations: List[Operation], output_file: Path, enhancements: Dict) -> None:
    """
    Generate a single self-contained operations.md that contains:

    1. Overview and quick-navigation section.
    2. Per-category summary tables whose Link column uses in-page anchors
       (e.g. ``#abs``) instead of external file links.
    3. Full per-operation detail sections (signature, parameters, returns,
       mathematical definition, related operations) appended after the tables.
    """
    # Group operations by category
    ops_by_category: Dict[str, List[Operation]] = {}
    for op in operations:
        if op.category not in ops_by_category:
            ops_by_category[op.category] = []
        ops_by_category[op.category].append(op)

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

    with open(output_file, "w", encoding="utf-8") as f:
        # ── Header ────────────────────────────────────────────────────────────
        f.write("# Forge Operations Reference\n\n")
        f.write(
            "Welcome to the Forge Operations Reference. This page provides a comprehensive guide to all supported operations in the Forge framework.\n\n"
        )

        # ── Overview ──────────────────────────────────────────────────────────
        f.write("## Overview\n\n")
        f.write(
            "Forge operations are organized into logical categories based on their functionality. Each operation is documented with detailed information including function signatures, parameters, examples, and usage notes.\n\n"
        )

        # ── Quick Navigation ──────────────────────────────────────────────────
        f.write("## Quick Navigation\n\n")
        for category in sorted_categories:
            anchor = category.lower().replace(" ", "-")
            desc = category_descriptions.get(category, "Operations in this category")
            f.write(f"- [{category}](#{anchor}) - {desc}\n")
        f.write("\n---\n\n")

        # ── Category summary tables ───────────────────────────────────────────
        # Links point to in-page anchors generated by the detail sections below.
        for category in sorted_categories:
            f.write(f"## {category}\n\n")

            desc = category_descriptions.get(category, "")
            if desc:
                f.write(f"{desc}.\n\n")

            ops = sorted(ops_by_category[category], key=lambda x: x.short_name)
            if ops:
                f.write("| Operation | Description | Link |\n")
                f.write("|-----------|-------------|------|\n")
                for op in ops:
                    short_desc = op.description[:80] + "..." if len(op.description) > 80 else op.description
                    anchor = op.short_name.lower()
                    f.write(f"| **{op.short_name}** | {short_desc} | [{op.name}](#{anchor}) |\n")
                f.write("\n")

        # ── Operation detail sections ─────────────────────────────────────────
        f.write("---\n\n")
        f.write("## Operation Details\n\n")

        all_ops = sorted(operations, key=lambda x: x.short_name.lower())
        for op in all_ops:
            f.write(_build_operation_section(op, enhancements))

        # ── Footer ────────────────────────────────────────────────────────────
        f.write(
            "*This documentation is automatically generated from operation definitions in `forge/forge/op/*.py`. For the most up-to-date information, refer to the source code.*\n"
        )


def cleanup_operations_dir(ops_dir: Path) -> int:
    """
    Remove all individual operation markdown files from the legacy
    ``docs/src/operations/`` directory.

    Operation documentation has been consolidated into ``operations.md``, so
    the separate per-operation files are no longer needed.

    Args:
        ops_dir: Path to the legacy ``docs/src/operations/`` directory.

    Returns:
        Number of files removed.
    """
    removed_count = 0

    if not ops_dir.exists():
        return removed_count

    for md_file in ops_dir.glob("*.md"):
        try:
            md_file.unlink()
            print(f"      Removed: {md_file.name}")
            removed_count += 1
        except OSError as e:
            print(f"      Warning: Could not remove {md_file.name}: {e}")

    # Remove the directory itself if now empty
    try:
        ops_dir.rmdir()
        print(f"      Removed directory: {ops_dir.name}/")
    except OSError:
        pass  # Not empty or already gone — leave it

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
        if stripped.startswith(("Parameters", "Returns", "Mathematical", "Notes", "Examples")):
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
                    if line.strip().startswith(("Notes", "Examples", "Parameters", "Returns")):
                        break
                if line.strip():
                    math_lines.append(line.strip())
        math_def = "\n".join(math_lines)

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

    # Import and run discovery
    print("=" * 60)
    print("Forge Operations Documentation Generator")
    print("=" * 60)

    sys.path.insert(0, str(script_dir))

    print(f"\n[1/4] Discovering operations from {op_dir}...")
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
    print(f"\n[2/4] Loading operation enhancements from {enhancements_path.name}...")
    enhancements = load_enhancements(enhancements_path)
    print(f"      Loaded enhancements for {len(enhancements)} operations")

    # Convert discovered operations
    print("\n[3/4] Converting operations...")
    operations = []
    errors = []

    for discovered in discovered_ops:
        try:
            op = convert_discovered_to_operation(discovered)
            operations.append(op)
            print(f"      [OK] {op.short_name}")
        except Exception as e:
            errors.append(f"{discovered.name}: {e}")
            print(f"      [FAIL] {discovered.name}: {e}")

    if errors:
        print(f"\n      Warning: {len(errors)} operation(s) had conversion errors")

    # Clean up the legacy per-operation files
    if do_cleanup:
        print(f"\n[4/4] Cleaning up legacy {ops_docs_dir.name}/ directory...")
        removed = cleanup_operations_dir(ops_docs_dir)
        if removed > 0:
            print(f"      Removed {removed} file(s)")
        else:
            print("      Nothing to remove")
    else:
        print(f"\n[4/4] Skipping cleanup of {ops_docs_dir.name}/ (--no-cleanup specified)")

    # Generate the combined operations.md
    print(f"\nGenerating {index_file.name}...")
    generate_operations_md(operations, index_file, enhancements)
    print(f"      [OK] {index_file.name}")

    # Summary
    print("\n" + "=" * 60)
    print("Documentation generation complete!")
    print(f"  Total operations: {len(operations)}")
    print(f"  Output file:      {index_file}")

    if errors:
        print(f"\n  Errors: {len(errors)}")
        for err in errors:
            print(f"    - {err}")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
