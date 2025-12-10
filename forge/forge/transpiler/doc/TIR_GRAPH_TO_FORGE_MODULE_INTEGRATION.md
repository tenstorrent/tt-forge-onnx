# TIR Graph to Forge Module: Integration Guide

## Executive Summary

This document provides a comprehensive guide for generating Forge modules from TIRGraph (Transpiler Intermediate Representation), designed to integrate seamlessly into the existing Forge compilation pipeline. This creates an alternative path to the TVM Relay IR path, enabling direct conversion from framework models (ONNX, PyTorch, etc.) to Forge modules via the TIR transpiler.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [TIRGraph Structure](#tirgraph-structure)
3. [TIRGraph → Python Forge Module Generation](#tirgraph--python-forge-module-generation)
4. [Integration into Forge Compilation Pipeline](#integration-into-forge-compilation-pipeline)
5. [Comparison with TVM Path](#comparison-with-tvm-path)
6. [Implementation Details](#implementation-details)
7. [Code Generation Details](#code-generation-details)
8. [Parameter and Constant Handling](#parameter-and-constant-handling)
9. [Operation Mapping System](#operation-mapping-system)
10. [Example Flows](#example-flows)

---

## 1. High-Level Architecture

### 1.1 Dual-Path Compilation

The Forge compiler supports two paths for generating Forge modules:

```
Path 1 (Existing): TVM Relay IR Path
Framework Model → TVM Relay IR → JSON Graphs → Python Forge Module

Path 2 (New): TIR Transpiler Path
Framework Model → TIRGraph → Python Forge Module
```

### 1.2 Complete Flow Diagram

```
User Code
    ↓
compile_main(module, inputs, ..., compiler_cfg=CompilerConfig(use_tir_transpiler=True))
    ↓
wrap_module() → Module (OnnxModule, PyTorchModule, etc.)
    ↓
forge_compile_from_context()
    ↓
generate_initial_graph()
    ↓
convert_to_forge_module() [TIR Path]
    ↓
ONNXToForgeTranspiler.transpile() → TIRGraph
    ↓
generate_forge_module_from_tir() [NEW]
    ↓
TIRCodeGenerator.generate() → Python Code String
    ↓
Dynamic Import → ForgeModule
    ↓
generate_graph() → Forge Graph
    ↓
[Compilation Passes]
    ↓
run_mlir_compiler() → Binary
    ↓
CompiledModel
```

### 1.3 Key Components

1. **TIRGraph**: Intermediate representation graph (`transpiler/core/graph.py`)
2. **TIRNode**: Individual operation nodes (`transpiler/ir/nodes.py`)
3. **TIRCodeGenerator**: Code generation from TIRGraph (`transpiler/codegen/generator.py`)
4. **Frontend Transpilers**: Convert framework models to TIRGraph (e.g., `ONNXToForgeTranspiler`)

---

## 2. TIRGraph Structure

### 2.1 TIRGraph Class (`transpiler/core/graph.py:16`)

**Purpose:** Represents a computational graph in Transpiler Intermediate Representation.

**Key Attributes:**
```python
class TIRGraph:
    name: str                                    # Graph name
    nodes: List[TIRNode]                         # Operation nodes
    inputs: List[str]                            # Input tensor names
    outputs: List[str]                           # Output tensor names
    params: Dict[str, torch.Tensor]             # Trainable weights
    constants: Dict[str, torch.Tensor]           # Non-trainable constants
    producer_map: Dict[str, str]                 # Output → Producer node
    consumer_map: Dict[str, List[str]]           # Input → Consumer nodes
```

**Key Methods:**
- `add_node(node: TIRNode)`: Add a node to the graph
- `get_topological_sort() -> List[TIRNode]`: Get nodes in execution order
- `run(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]`: Execute graph

### 2.2 TIRNode Class (`transpiler/ir/nodes.py:11`)

**Purpose:** Represents a single operation in the TIR graph.

**Key Attributes:**
```python
class TIRNode:
    name: str                                    # Node name
    op_type: str                                 # Operation type (e.g., "Conv2d", "Add")
    inputs: List[str]                            # Input tensor names
    outputs: List[str]                           # Output tensor names
    input_tensors: Dict[str, TensorInfo]         # Input metadata
    output_tensors: Dict[str, TensorInfo]        # Output metadata
    attrs: Dict[str, Any]                        # PyTorch-compatible attributes
    forge_attrs: Dict[str, Any]                  # Forge-specific attributes
    forge_op_function_name: str                  # Forge op function name
```

**Key Methods:**
- `emit() -> Dict[str, Any]`: Returns dictionary for code generation
- `eval(input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]`: Execute operation

### 2.3 TIRGraph Example

```python
# Example TIRGraph structure
tir_graph = TIRGraph(name="simple_model")
tir_graph.inputs = ["input"]
tir_graph.outputs = ["output"]
tir_graph.params = {
    "conv_weight": torch.randn(3, 1, 3, 3),
    "conv_bias": torch.randn(3),
}
tir_graph.nodes = [
    Conv2dNode(name="conv1", inputs=["input"], outputs=["conv1_out"], ...),
    ReluNode(name="relu1", inputs=["conv1_out"], outputs=["output"], ...),
]
```

---

## 3. TIRGraph → Python Forge Module Generation

### 3.1 Tensor Conversion Requirements

**Critical:** Different compilation paths require different input tensor formats.

#### 3.1.1 TVM Path Tensor Conversion

**Location:** `compile.py:1012`, `tvm_to_python.py:1825`

**Conversion:**
```python
# In convert_to_forge_module() for TVM path
pytorch_inputs = to_pt_tensors(module_inputs)  # Convert to PyTorch tensors
forge_module, dev_types, module_inputs = generate_forge_module(
    module,
    pytorch_inputs,  # TVM expects PyTorch tensors
    compiler_cfg,
    module.name,
    verify_cfg,
)
```

**Why:** TVM Relay IR frontends (e.g., `relay.frontend.from_onnx`) work with PyTorch tensors internally.

#### 3.1.2 TIR Transpiler Path Tensor Conversion

**Location:** `transpiler/codegen/tir_to_forge.py` (NEW)

**Conversion:**
```python
# In convert_to_forge_module() for TIR path
if isinstance(module, OnnxModule):
    # Convert inputs to numpy arrays for ONNX transpiler
    import numpy as np
    onnx_inputs = []
    for inp in module_inputs:
        if isinstance(inp, torch.Tensor):
            onnx_inputs.append(inp.detach().cpu().numpy())
        elif isinstance(inp, np.ndarray):
            onnx_inputs.append(inp)
        else:
            onnx_inputs.append(np.array(inp))
    
    # ONNX transpiler works with numpy arrays
    transpiler = ONNXToForgeTranspiler(validate_model=True)
    tir_graph = transpiler.transpile(module.module)  # Uses numpy internally
```

**Why:** ONNX models use numpy arrays, and the transpiler needs to match ONNX's native format.

**After TIRGraph Generation:**
```python
# Convert back to PyTorch tensors for Forge module consistency
forge_inputs = to_pt_tensors(onnx_inputs)  # Convert numpy → PyTorch
```

### 3.2 Current Implementation: `generate_forge_module()` (`transpiler/codegen/generator.py:8`)

**Purpose:** Generate Python code string from TIRGraph.

**Current Implementation:**
```python
def generate_forge_module(graph: TIRGraph, class_name="GeneratedForgeModule") -> str:
    """
    Generates a Python string for the Forge module by traversing the graph.
    
    Note: This is a basic implementation. See TIRCodeGenerator for enhanced version.
    """
    lines = []
    lines.append("import torch")
    lines.append("import forge")
    lines.append("")
    lines.append(f"class {class_name}(forge.Module):")
    lines.append(f"    def __init__(self, name='{graph.name}'):")
    lines.append(f"        super().__init__(name=name)")
    
    # Add Parameters
    for name, tensor in graph.params.items():
        shape_str = str(tuple(tensor.shape))
        lines.append(f"        self.add_parameter('{name}', forge.Parameter(shape={shape_str}))")
    
    # Add Constants
    for name, tensor in graph.constants.items():
        shape_str = str(tuple(tensor.shape))
        lines.append(f"        self.add_parameter('{name}', forge.Parameter(shape={shape_str}, requires_grad=False))")
    
    # Forward Method
    all_initializers = set(graph.params.keys()) | set(graph.constants.keys())
    forward_args = [inp for inp in graph.inputs if inp not in all_initializers]
    args_str = ", ".join(forward_args)
    lines.append(f"    def forward(self, {args_str}):")
    
    # Operations
    sorted_nodes = graph.get_topological_sort()
    for node in sorted_nodes:
        op_info = node.emit()
        # ... format inputs and attributes ...
        lines.append(f"        {output_name} = {op_info['function_name']}({call_args})")
    
    # Return
    if len(graph.outputs) == 1:
        lines.append(f"        return {graph.outputs[0]}")
    else:
        out_str = ", ".join(graph.outputs)
        lines.append(f"        return {out_str}")
    
    return "\n".join(lines)
```

### 3.3 Enhanced Implementation Requirements

To match the TVM path's capabilities, the TIR code generator needs:

1. **Parameter Loading Support**: Method to load parameters from framework models (ONNX, PyTorch, etc.)
2. **Device Placement**: Support for CPU/Device partitioning (future enhancement)
3. **Memory Management**: Input deletion for memory optimization (`delete_inputs` flag)
4. **Source Layer Tracking**: Track original framework layer names for debugging
5. **Error Handling**: Robust error handling and validation
6. **Code Formatting**: Proper indentation and formatting (matching ForgeWriter style)
7. **Data Format Handling**: Proper conversion of dtypes to Forge DataFormat
8. **Multi-Output Support**: Enhanced support for operations with multiple outputs
9. **Submodule Support**: Support for nested modules (future enhancement)
10. **Verification Hooks**: Integration with verification system

---

## 4. Integration into Forge Compilation Pipeline

### 4.1 Entry Point: `generate_forge_module_from_tir()` (NEW)

**Purpose:** Main entry point for TIR-based Forge module generation, mirroring `generate_forge_module()` from TVM path.

**Location:** `forge/forge/transpiler/codegen/tir_to_forge.py` (NEW FILE)

**Function Signature:**
```python
def generate_forge_module_from_tir(
    tir_graph: TIRGraph,
    framework_mod: Module,
    compiler_cfg: CompilerConfig = None,
    graph_name: str = None,
    verify_cfg: DeprecatedVerifyConfig = None,
    clean_later: bool = False,
    input_names: List[str] = [],
) -> Tuple[List[ForgeModule], List[str], List[torch.Tensor]]:
    """
    Generate Forge modules from TIRGraph.
    
    This function mirrors the TVM path's generate_forge_module() but works with TIRGraph
    instead of TVM Relay IR. It handles:
    - Code generation from TIRGraph
    - File I/O for generated Python modules
    - Dynamic module import and instantiation
    - Parameter loading from framework models
    - Verification (if enabled)
    
    Args:
        tir_graph: TIRGraph to convert to Forge module
        framework_mod: Original framework module (for parameter loading)
        compiler_cfg: Compiler configuration
        graph_name: Name for the generated module
        verify_cfg: Verification configuration
        clean_later: Whether to clean up generated files later
        input_names: Optional input names (for consistency with TVM path)
        
    Returns:
        Tuple of (forge_modules, devices, forge_inputs)
        - forge_modules: List of ForgeModule instances (typically one)
        - devices: List of device types (typically ["TTDevice"])
        - forge_inputs: List of input tensors (converted to PyTorch format)
        
    Raises:
        ValueError: If TIRGraph is invalid or code generation fails
        ImportError: If generated module cannot be imported
        AssertionError: If parameter loading fails
    """
```

**Line-by-Line Implementation:**

```python
def generate_forge_module_from_tir(
    tir_graph: TIRGraph,
    framework_mod: Module,
    compiler_cfg: CompilerConfig = None,
    graph_name: str = None,
    verify_cfg: DeprecatedVerifyConfig = None,
    clean_later: bool = False,
    input_names: List[str] = [],
) -> Tuple[List[ForgeModule], List[str], List[torch.Tensor]]:
    """
    Generate Forge modules from TIRGraph with full feature support.
    """
    import os
    import sys
    import importlib.util
    from forge.module import ForgeModule
    from forge.tensor import to_pt_tensors
    from forge.verify.config import _get_global_verify_config
    
    # Lines 1-15: Initialize configurations
    if compiler_cfg is None:
        compiler_cfg = CompilerConfig()
    if verify_cfg is None:
        verify_cfg = _get_global_verify_config()
    
    if graph_name is None:
        graph_name = tir_graph.name
    
    # Lines 16-25: Get framework outputs for verification (if needed)
    if verify_cfg is not None and verify_cfg.verify_forge_codegen_vs_framework:
        # Run framework model to get golden outputs for comparison
        sample_inputs = _prepare_sample_inputs(tir_graph, framework_mod)
        framework_outputs = framework_mod.cpu_eval_forward(*sample_inputs)
    else:
        framework_outputs = None
    
    # Lines 26-35: Generate Python code from TIRGraph
    module_name = graph_name
    class_name = _to_pascal_case(module_name)
    
    # Determine framework type for code generation
    framework = _get_framework_type(framework_mod)
    
    # Generate Python code string using enhanced TIRCodeGenerator
    code_generator = TIRCodeGenerator(
        tir_graph=tir_graph,
        class_name=class_name,
        framework=framework,
        compiler_cfg=compiler_cfg,
    )
    python_code = code_generator.generate()
    
    # Lines 36-50: Write Python code to file
    module_directory = "generated_modules"
    os.makedirs(module_directory, exist_ok=True)
    filename = f"{module_name}.py"
    file_path = os.path.join(module_directory, filename)
    
    with open(file_path, "w") as f:
        f.write(python_code)
    
    logger.info(f"Generated Forge module code: {file_path}")
    
    # Lines 51-65: Dynamically import and instantiate module
    sys.path.append(".")
    module = import_from_path(module_name, file_path)
    TestClass = getattr(module, class_name)
    
    # Create ForgeModule instance
    forge_mod = TestClass(module_name)
    
    # Lines 66-80: Load parameters from framework model
    try:
        forge_mod.process_framework_parameters(framework_mod.module)
    except Exception as e:
        logger.error(f"Failed to load parameters from framework model: {e}")
        raise
    
    # Verify all parameters were loaded
    missing_params = [
        param for param in forge_mod.get_parameters() 
        if param.value() is None
    ]
    if missing_params:
        missing_names = [p.get_name() for p in missing_params]
        raise AssertionError(
            f"Could not retrieve parameters from framework and TIR: {missing_names}"
        )
    
    logger.info(f"Successfully loaded {len(forge_mod.get_parameters())} parameters")
    
    # Lines 81-95: Prepare inputs (convert to PyTorch tensors for consistency)
    forge_inputs = _prepare_forge_inputs(tir_graph, framework_mod)
    forge_inputs = to_pt_tensors(forge_inputs)  # Ensure PyTorch format
    
    # Lines 96-110: Verify generated module (if enabled)
    if verify_cfg is not None and verify_cfg.verify_forge_codegen_vs_framework:
        if framework_outputs is None:
            logger.warning("Verification enabled but framework outputs not available")
        else:
            forge_outputs = _get_forge_outputs([forge_mod], forge_inputs)
            verify_framework_vs_forge_codegen(
                framework_outputs, 
                forge_outputs, 
                verify_cfg=verify_cfg
            )
            logger.info("Verification passed: Forge module matches framework outputs")
    
    # Lines 111-120: Cleanup (if needed)
    if not compiler_cfg.tir_retain_python_files and not clean_later:
        cleanup_temporary_files([filename])
        logger.debug(f"Cleaned up temporary file: {filename}")
    
    return [forge_mod], ["TTDevice"], forge_inputs

def _get_framework_type(framework_mod: Module) -> str:
    """Determine framework type from module."""
    from forge.module import OnnxModule, PyTorchModule, TFLiteModule
    
    if isinstance(framework_mod, OnnxModule):
        return "onnx"
    elif isinstance(framework_mod, PyTorchModule):
        return "pytorch"
    elif isinstance(framework_mod, TFLiteModule):
        return "tflite"
    else:
        return "unknown"
```

### 4.2 Integration into `convert_to_forge_module()` (`compile.py:990`)

**Purpose:** Add TIR path option to existing conversion function with proper tensor conversion.

**Current Implementation:**
```python
def convert_to_forge_module(
    module: AnyModule,
    module_inputs: Union[AnyTensor, List[AnyTensor]],
    compiler_cfg: CompilerConfig,
    verify_cfg: DeprecatedVerifyConfig,
) -> ForgeModule:
    from .tvm_to_python import generate_forge_module  # TVM path
    # ...
    forge_module, dev_types, module_inputs = generate_forge_module(
        module,
        to_pt_tensors(module_inputs),  # Convert to PyTorch tensors for TVM
        compiler_cfg,
        module.name,
        verify_cfg,
    )
    return forge_module[0], module_inputs
```

**Enhanced Implementation:**
```python
def convert_to_forge_module(
    module: AnyModule,
    module_inputs: Union[AnyTensor, List[AnyTensor]],
    compiler_cfg: CompilerConfig,
    verify_cfg: DeprecatedVerifyConfig,
) -> ForgeModule:
    """
    Convert framework module to Forge module.
    
    Supports two paths:
    1. TVM Path: Framework Model → TVM Relay IR → JSON Graphs → Forge Module
    2. TIR Path: Framework Model → TIRGraph → Forge Module
    
    Tensor Conversion:
    - TVM Path: Inputs converted to PyTorch tensors (to_pt_tensors)
    - TIR Path: Inputs converted to framework-native format (numpy for ONNX)
    """
    # Determine which path to use
    use_tir_transpiler = (
        compiler_cfg.use_tir_transpiler and 
        compiler_cfg.compile_tir_to_python
    )
    use_tvm_path = compiler_cfg.compile_tvm_to_python and not use_tir_transpiler
    
    if use_tir_transpiler:
        # TIR Path: Framework Model → TIRGraph → Forge Module
        from .transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
        from .transpiler.codegen.tir_to_forge import generate_forge_module_from_tir
        
        # Convert framework model to TIRGraph
        if isinstance(module, OnnxModule):
            # Convert inputs to numpy arrays for ONNX transpiler
            import numpy as np
            onnx_inputs = []
            for inp in module_inputs:
                if isinstance(inp, torch.Tensor):
                    onnx_inputs.append(inp.detach().cpu().numpy())
                elif isinstance(inp, np.ndarray):
                    onnx_inputs.append(inp)
                else:
                    # Try to convert to numpy
                    onnx_inputs.append(np.array(inp))
            
            transpiler = ONNXToForgeTranspiler(
                validate_model=True, 
                debug=compiler_cfg.tir_enable_debug
            )
            tir_graph = transpiler.transpile(module.module)
        else:
            raise NotImplementedError(
                f"TIR transpiler not yet supported for {type(module)}. "
                f"Currently only ONNX models are supported."
            )
        
        # Generate Forge module from TIRGraph
        forge_modules, dev_types, forge_inputs = generate_forge_module_from_tir(
            tir_graph=tir_graph,
            framework_mod=module,
            compiler_cfg=compiler_cfg,
            graph_name=module.name,
            verify_cfg=verify_cfg,
        )
        
        # Convert forge_inputs back to torch tensors for consistency
        module_inputs = to_pt_tensors(forge_inputs)
        return forge_modules[0], module_inputs
        
    elif use_tvm_path:
        # TVM Path: Framework Model → TVM Relay IR → Forge Module (existing)
        from .tvm_to_python import generate_forge_module
        
        # Convert inputs to PyTorch tensors for TVM path
        pytorch_inputs = to_pt_tensors(module_inputs)
        
        forge_module, dev_types, module_inputs = generate_forge_module(
            module,
            pytorch_inputs,
            compiler_cfg,
            module.name,
            verify_cfg,
        )
        return forge_module[0], module_inputs
    else:
        raise ValueError(
            "Either compile_tvm_to_python or compile_tir_to_python must be True. "
            f"Current config: compile_tvm_to_python={compiler_cfg.compile_tvm_to_python}, "
            f"compile_tir_to_python={compiler_cfg.compile_tir_to_python}"
        )
```

### 4.2.1 Integration into `generate_initial_graph()` (`compile.py:612`)

**Purpose:** Update the graph generation function to support both paths.

**Current Implementation:**
```python
def generate_initial_graph(context: CompileContext) -> CompileDepth:
    modules_ = []
    if context.compiler_cfg.compile_tvm_to_python and context.graph is None:
        module_inputs = context.inputs
        for module in context.modules:
            if not isinstance(module, ForgeModule):
                module, module_inputs = convert_to_forge_module(
                    module,
                    module_inputs,
                    context.compiler_cfg,
                    context.verify_cfg,
                )
                assert isinstance(module, ForgeModule)
                context.inputs = module_inputs
            modules_.append(module)
    # ... rest of function
```

**Enhanced Implementation:**
```python
def generate_initial_graph(context: CompileContext) -> CompileDepth:
    """
    Generates initial graph from the input framework.
    Supports both TVM and TIR transpiler paths.
    """
    modules_ = []
    
    # Check if we should convert to Forge module (either TVM or TIR path)
    should_convert = (
        (context.compiler_cfg.compile_tvm_to_python or 
         context.compiler_cfg.compile_tir_to_python) and 
        context.graph is None
    )
    
    if should_convert:
        module_inputs = context.inputs
        for module in context.modules:
            if not isinstance(module, ForgeModule):
                module, module_inputs = convert_to_forge_module(
                    module,
                    module_inputs,
                    context.compiler_cfg,
                    context.verify_cfg,
                )
                assert isinstance(module, ForgeModule)
                context.inputs = module_inputs
            modules_.append(module)
    
    # ... rest of function (graph generation, etc.)
```

### 4.3 CompilerConfig Enhancement

**Add TIR Transpiler Options:**
```python
# In config.py
@dataclass_json
@dataclass
class CompilerConfig:
    # ... existing fields ...
    
    # TVM Path Configuration (existing)
    compile_tvm_to_python: bool = True  # Generate Python code from TVM Relay IR
    
    # TIR Transpiler Path Configuration (new)
    compile_tir_to_python: bool = False  # Generate Python code from TIRGraph
    use_tir_transpiler: bool = False  # Use TIR transpiler instead of TVM path
    
    # TIR-specific options
    tir_retain_python_files: bool = False  # Keep generated Python files
    tir_enable_debug: bool = False  # Enable debug mode for TIR transpiler
```

**Configuration Logic:**
- If `compile_tvm_to_python=True` and `use_tir_transpiler=False`: Use TVM path (default)
- If `compile_tir_to_python=True` and `use_tir_transpiler=True`: Use TIR transpiler path
- Both flags can coexist, allowing users to choose the path per compilation

---

## 5. Comparison with TVM Path

### 5.1 Path Comparison Table

| Aspect | TVM Path | TIR Path |
|--------|----------|----------|
| **Input** | Framework Model | Framework Model |
| **Intermediate** | TVM Relay IR → JSON Graphs | TIRGraph |
| **Code Generation** | JSON Graphs → Python Code | TIRGraph → Python Code |
| **Output** | ForgeModule | ForgeModule |
| **Partitioning** | CPU/Device partitioning | Single device (can be extended) |
| **Optimization** | TVM Relay passes | TIR-level optimizations |
| **Complexity** | High (multiple stages) | Lower (direct conversion) |
| **Flexibility** | High (TVM ecosystem) | Medium (TIR-specific) |

### 5.2 Advantages of TIR Path

1. **Directness**: Fewer intermediate representations
2. **Simplicity**: Simpler code generation logic
3. **Framework-Specific**: Can leverage framework-specific optimizations
4. **Debugging**: Easier to debug (fewer transformation steps)
5. **Control**: More control over the conversion process

### 5.3 Advantages of TVM Path

1. **Maturity**: Well-tested and optimized
2. **Optimization**: Advanced TVM optimization passes
3. **Partitioning**: Built-in CPU/Device partitioning
4. **Ecosystem**: Access to TVM's operator library
5. **Multi-Framework**: Unified path for all frameworks

---

## 6. Implementation Details

### 6.1 TIRCodeGenerator Class (NEW)

**Location:** `forge/forge/transpiler/codegen/tir_generator.py` (NEW FILE)

**Purpose:** Enhanced code generator for TIRGraph, matching ForgeWriter capabilities.

**Class Structure:**
```python
class TIRCodeGenerator:
    """Generates Python Forge module code from TIRGraph."""
    
    def __init__(
        self,
        tir_graph: TIRGraph,
        class_name: str,
        framework: str = "onnx",
        compiler_cfg: CompilerConfig = None,
    ):
        self.tir_graph = tir_graph
        self.class_name = class_name
        self.framework = framework
        self.compiler_cfg = compiler_cfg or CompilerConfig()
        self.lines = []
        self.indent = 0
        self.param_names = []
        self.const_names = []
    
    def generate(self) -> str:
        """Generate complete Python module code."""
        self.write_header()
        self.write_class_definition()
        self.write_forward()
        self.write_param_parser()
        return "\n".join(self.lines)
    
    def write_header(self):
        """Write imports and module header."""
        self.wl("import torch")
        self.wl("import forge")
        self.wl("import forge.op")
        self.wl("from forge import ForgeModule")
        self.wl("")
        self.wl("from loguru import logger")
        self.wl("")
    
    def write_class_definition(self):
        """Write ForgeModule class definition."""
        self.wl(f"class {self.class_name}(ForgeModule):")
        self.indent += 1
        self.wl("def __init__(self, name):")
        self.indent += 1
        self.wl("super().__init__(name)")
        self.wl("")
        
        # Add parameters
        for name, tensor in self.tir_graph.params.items():
            shape = tuple(tensor.shape)
            dtype = tensor.dtype
            self.param_names.append(name)
            self.wl(
                f'self.add_parameter("{name}", '
                f'forge.Parameter(*{shape}, requires_grad=True, '
                f'dev_data_format={self._get_forge_data_format(dtype, name)})'
            )
        
        # Add constants
        for name, tensor in self.tir_graph.constants.items():
            shape = tuple(tensor.shape)
            dtype = tensor.dtype
            self.const_names.append(name)
            self.wl(
                f'self.add_constant("{name}", '
                f'shape={shape}, dtype={self._get_pytorch_dtype(dtype, name)})'
            )
        
        self.indent = 0
        self.wl("")
    
    def write_forward(self):
        """Write forward method."""
        self.indent = 1
        
        # Get forward arguments (exclude params and constants)
        all_initializers = set(self.tir_graph.params.keys()) | set(self.tir_graph.constants.keys())
        forward_args = [inp for inp in self.tir_graph.inputs if inp not in all_initializers]
        args_str = ", ".join(forward_args)
        
        self.wl(f"def forward(self, {args_str}):")
        self.indent += 1
        
        # Get nodes in topological order
        sorted_nodes = self.tir_graph.get_topological_sort()
        
        for node in sorted_nodes:
            op_info = node.emit()
            
            # Format input names
            input_names = self._format_input_names(op_info['input_names'])
            
            # Format attributes
            attrs = op_info.get('args', {})
            attr_strs = []
            for k, v in attrs.items():
                if isinstance(v, str):
                    attr_strs.append(f"{k}='{v}'")
                else:
                    attr_strs.append(f"{k}={v}")
            attrs_str = ", ".join(attr_strs)
            
            # Format function call
            call_args = input_names
            if attrs_str:
                call_args += f", {attrs_str}"
            
            # Write operation call
            output_name = op_info['output_name']
            function_name = op_info['function_name']
            node_name = op_info['node_name']
            
            self.wl(f"# {node.op_type} -> {node_name}")
            self.wl(f'{output_name} = {function_name}("{node_name}", {call_args})')
            
            # Memory optimization: delete inputs if needed
            if self.compiler_cfg.enable_input_deletion:
                for input_name in op_info['input_names']:
                    if input_name not in all_initializers:
                        self.wl(f"{input_name}._value = None")
        
        # Write return statement
        if len(self.tir_graph.outputs) == 1:
            self.wl(f"return {self.tir_graph.outputs[0]}")
        else:
            out_str = ", ".join(self.tir_graph.outputs)
            self.wl(f"return {out_str}")
        
        self.indent = 0
        self.wl("")
    
    def write_param_parser(self):
        """Write parameter loading method."""
        self.indent = 1
        
        if self.framework == "pytorch" or self.framework == "paddle":
            self.wl("def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl("named_parameters = dict(model.state_dict().items())")
            self.wl("named_buffers = dict(model.named_buffers())")
            self.wl("named_parameters.update(named_buffers)")
            self.wl("")
            self.wl("for name, param in named_parameters.items():")
            self.indent += 1
            self.wl("if name in self.param_names:")
            self.indent += 1
            self.wl("self.get_parameter(name).set_value(param)")
            self.indent -= 2
        elif self.framework == "onnx":
            self.wl("def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl("import onnx")
            self.wl("import onnx.numpy_helper as numpy_helper")
            self.wl("")
            self.wl("for initializer in model.graph.initializer:")
            self.indent += 1
            self.wl("name = initializer.name")
            self.wl("if name in self.param_names:")
            self.indent += 1
            self.wl("tensor = torch.from_numpy(numpy_helper.to_array(initializer))")
            self.wl("self.get_parameter(name).set_value(tensor)")
            self.indent -= 2
        
        self.indent = 0
        self.wl("")
    
    def _format_input_names(self, input_names: List[str]) -> str:
        """Format input names for function call."""
        formatted = []
        for name in input_names:
            if name in self.param_names:
                formatted.append(f'self.get_parameter("{name}")')
            elif name in self.const_names:
                formatted.append(f'self.get_constant("{name}")')
            else:
                formatted.append(name)
        return ", ".join(formatted)
    
    def _get_forge_data_format(self, dtype: torch.dtype, name: str) -> str:
        """Convert PyTorch dtype to Forge DataFormat."""
        # Implementation similar to forge_df_from_str in python_codegen.py
        # ...
    
    def _get_pytorch_dtype(self, dtype: torch.dtype, name: str) -> str:
        """Convert PyTorch dtype to string representation."""
        # Implementation similar to pytorch_df_from_str in python_codegen.py
        # ...
    
    def wl(self, line: str):
        """Write line with current indentation."""
        indent_str = "    " * self.indent
        self.lines.append(indent_str + line)
```

### 6.2 Helper Functions

**Location:** `forge/forge/transpiler/codegen/tir_to_forge.py`

```python
def _prepare_sample_inputs(tir_graph: TIRGraph, framework_mod: Module) -> List[torch.Tensor]:
    """
    Prepare sample inputs for framework model execution.
    
    Extracts input shapes and dtypes from TIRGraph and creates random tensors.
    """
    sample_inputs = []
    for input_name in tir_graph.inputs:
        if input_name in tir_graph.input_tensors:
            tensor_info = tir_graph.input_tensors[input_name]
            shape = tensor_info.shape
            dtype = tensor_info.torch_dtype or torch.float32
            sample_inputs.append(torch.randn(*shape, dtype=dtype))
    return sample_inputs

def _prepare_forge_inputs(tir_graph: TIRGraph, framework_mod: Module) -> List[torch.Tensor]:
    """
    Prepare inputs for Forge module.
    
    Similar to _prepare_sample_inputs but ensures PyTorch tensor format.
    """
    return _prepare_sample_inputs(tir_graph, framework_mod)

def _get_forge_outputs(forge_modules: List[ForgeModule], inputs: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Get outputs from Forge modules.
    
    Handles both single and multiple outputs.
    """
    outputs = []
    for mod in forge_modules:
        mod_outputs = mod.forward(*inputs)
        if isinstance(mod_outputs, tuple):
            outputs.extend(mod_outputs)
        else:
            outputs.append(mod_outputs)
    return outputs

def _to_pascal_case(name: str) -> str:
    """Convert name to PascalCase."""
    return ''.join(word.capitalize() for word in name.split('_'))

def to_onnx_inputs(inputs: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
    """
    Convert inputs to numpy arrays for ONNX transpiler.
    
    This function handles the tensor conversion required for the TIR transpiler path.
    ONNX models work with numpy arrays, so we convert PyTorch tensors to numpy.
    
    Args:
        inputs: Input tensors in various formats (PyTorch, numpy, or mixed)
        
    Returns:
        List of numpy arrays
        
    Example:
        >>> pytorch_inputs = [torch.randn(1, 3, 224, 224)]
        >>> onnx_inputs = to_onnx_inputs(pytorch_inputs)
        >>> assert isinstance(onnx_inputs[0], np.ndarray)
    """
    import numpy as np
    
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    
    onnx_inputs = []
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            onnx_inputs.append(inp.detach().cpu().numpy())
        elif isinstance(inp, np.ndarray):
            onnx_inputs.append(inp)
        else:
            # Try to convert to numpy
            try:
                onnx_inputs.append(np.array(inp))
            except Exception as e:
                raise ValueError(
                    f"Cannot convert input to numpy array: {type(inp)}. "
                    f"Error: {e}"
                )
    
    return onnx_inputs

def to_framework_inputs(
    inputs: Union[np.ndarray, List[np.ndarray]], 
    framework: str
) -> Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]]:
    """
    Convert inputs to framework-specific format.
    
    Args:
        inputs: Input arrays (numpy or PyTorch)
        framework: Framework type ("onnx", "pytorch", etc.)
        
    Returns:
        Inputs in framework-specific format
    """
    if framework == "onnx":
        # ONNX uses numpy arrays
        return to_onnx_inputs(inputs) if not isinstance(inputs, (list, tuple)) else to_onnx_inputs(inputs)
    elif framework == "pytorch":
        # PyTorch uses torch tensors
        from forge.tensor import to_pt_tensors
        return to_pt_tensors(inputs)
    else:
        # Default: return as-is
        return inputs
```

---

## 7. Code Generation Details

### 7.1 Generated Code Structure

**Example Generated Code:**
```python
import torch
import forge
import forge.op
from forge import ForgeModule

from loguru import logger

class SimpleModel(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter("conv_weight", forge.Parameter(*(3, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("conv_bias", forge.Parameter(*(3,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_constant("scale", shape=(1,), dtype=torch.float32)
    
    def forward(self, input):
        # Conv2d -> conv1
        conv1_out = forge.op.Conv2d("conv1", input, self.get_parameter("conv_weight"), self.get_parameter("conv_bias"), kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        # Relu -> relu1
        output = forge.op.Relu("relu1", conv1_out)
        return output
    
    def process_framework_parameters(self, model):
        import onnx
        import onnx.numpy_helper as numpy_helper
        
        for initializer in model.graph.initializer:
            name = initializer.name
            if name in self.param_names:
                tensor = torch.from_numpy(numpy_helper.to_array(initializer))
                self.get_parameter(name).set_value(tensor)
```

### 7.2 Operation Call Format

**Standard Format:**
```python
output_name = forge.op.OperationName("node_name", input1, input2, ..., attr1=value1, attr2=value2)
```

**Examples:**
```python
# Conv2d
conv_out = forge.op.Conv2d("conv1", input, weight, bias, kernel_size=(3, 3), stride=(1, 1))

# MatMul
matmul_out = forge.op.Matmul("matmul1", input, weight)

# Add
add_out = forge.op.Add("add1", input1, input2)

# Reshape
reshape_out = forge.op.Reshape("reshape1", input, shape=(1, -1))
```

### 7.3 Parameter and Constant Access

**Parameters:**
```python
self.get_parameter("param_name")
```

**Constants:**
```python
self.get_constant("const_name")
```

---

## 8. Parameter and Constant Handling

### 8.1 Parameter Loading from Framework Models

**ONNX Models:**
```python
def process_framework_parameters(self, model):
    import onnx
    import onnx.numpy_helper as numpy_helper
    
    for initializer in model.graph.initializer:
        name = initializer.name
        if name in self.param_names:
            tensor = torch.from_numpy(numpy_helper.to_array(initializer))
            self.get_parameter(name).set_value(tensor)
```

**PyTorch Models:**
```python
def process_framework_parameters(self, model):
    named_parameters = dict(model.state_dict().items())
    named_buffers = dict(model.named_buffers())
    named_parameters.update(named_buffers)
    
    for name, param in named_parameters.items():
        if name in self.param_names:
            self.get_parameter(name).set_value(param)
```

### 8.2 Parameter vs Constant Distinction

**Parameters (Trainable):**
- `requires_grad=True` in framework model
- Stored in `TIRGraph.params`
- Added via `self.add_parameter()` with `requires_grad=True`

**Constants (Non-Trainable):**
- `requires_grad=False` in framework model
- Stored in `TIRGraph.constants`
- Added via `self.add_constant()` or `self.add_parameter()` with `requires_grad=False`

---

## 9. Operation Mapping System

### 9.1 TIRNode.emit() Method

**Purpose:** Returns dictionary for code generation.

**Structure:**
```python
{
    "function_name": "forge.op.Conv2d",      # Forge operation function
    "node_name": "conv1",                    # Node name
    "output_name": "conv1_out",              # Output tensor name
    "output_names": ["conv1_out"],           # All output names
    "input_names": ["input", "weight"],      # Input tensor names
    "input_shapes": [(1, 1, 28, 28), ...],  # Input shapes
    "input_dtypes": [torch.float32, ...],    # Input dtypes
    "args": {                                # Operation arguments
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (0, 0),
    }
}
```

### 9.2 Forge Operation Function Names

**Mapping from TIRNode.op_type to Forge function:**
- `Conv2d` → `forge.op.Conv2d`
- `MatMul` → `forge.op.Matmul`
- `Add` → `forge.op.Add`
- `Relu` → `forge.op.Relu`
- `Reshape` → `forge.op.Reshape`
- etc.

**Set in TIRNode.__init__():**
```python
self.forge_op_function_name = f"forge.op.{self.op_type}"
```

---

## 10. Example Flows

### 10.1 Complete ONNX → Forge Module Flow

**Step 1: ONNX Model to TIRGraph**
```python
# User code
onnx_model = onnx.load("model.onnx")
transpiler = ONNXToForgeTranspiler(validate_model=True)
tir_graph = transpiler.transpile(onnx_model)
```

**Step 2: TIRGraph to Python Code**
```python
# Internal
code_generator = TIRCodeGenerator(
    tir_graph=tir_graph,
    class_name="GeneratedModel",
    framework="onnx",
)
python_code = code_generator.generate()
```

**Step 3: Python Code to ForgeModule**
```python
# Internal
module = import_from_path("GeneratedModel", "generated_modules/GeneratedModel.py")
TestClass = getattr(module, "GeneratedModel")
forge_mod = TestClass("model")
forge_mod.process_framework_parameters(onnx_model)
```

**Step 4: Integration into Compilation Pipeline**
```python
# In compile_main
compiler_cfg = CompilerConfig(use_tir_transpiler=True)
compiled_model = compile_main(
    module=OnnxModule("model", onnx_model),
    sample_inputs=[torch.randn(1, 1, 28, 28)],
    compiler_cfg=compiler_cfg,
)
```

### 10.2 Comparison: TVM vs TIR Path

**TVM Path:**
```
ONNX Model
  → relay.frontend.from_onnx() → TVM Relay IR
  → compile_tvm_for_forge() → Partitioned Relay IR
  → extract_graphs() → JSON Graphs
  → compile_tvm_to_python() → Python Code
  → ForgeModule
```

**TIR Path:**
```
ONNX Model
  → ONNXToForgeTranspiler.transpile() → TIRGraph
  → TIRCodeGenerator.generate() → Python Code
  → ForgeModule
```

---

## 11. Integration Checklist

### 11.1 Implementation Steps

- [ ] **Step 1:** Enhance `TIRCodeGenerator` class
  - [ ] Add parameter loading support (ONNX, PyTorch, etc.)
  - [ ] Add proper code formatting (matching ForgeWriter style)
  - [ ] Add error handling and validation
  - [ ] Add memory management options (`delete_inputs` flag)
  - [ ] Add data format conversion (dtype → Forge DataFormat)
  - [ ] Add source layer tracking for debugging
  - [ ] Add multi-output operation support

- [ ] **Step 2:** Create `generate_forge_module_from_tir()` function
  - [ ] Implement main entry point with full error handling
  - [ ] Add file I/O handling with proper directory management
  - [ ] Add dynamic import logic (mirroring TVM path)
  - [ ] Add parameter loading with validation
  - [ ] Add verification support (framework vs Forge comparison)
  - [ ] Add cleanup logic for temporary files
  - [ ] Add logging for debugging

- [ ] **Step 3:** Enhance `CompilerConfig`
  - [ ] Add `compile_tir_to_python` flag (mirroring `compile_tvm_to_python`)
  - [ ] Add `use_tir_transpiler` flag for path selection
  - [ ] Add `tir_retain_python_files` flag (mirroring `retain_tvm_python_files`)
  - [ ] Add `tir_enable_debug` flag for debug mode
  - [ ] Add validation to ensure at least one path is enabled

- [ ] **Step 4:** Modify `convert_to_forge_module()`
  - [ ] Add TIR path option with proper tensor conversion
  - [ ] Add framework-specific routing (ONNX → numpy, PyTorch → torch, etc.)
  - [ ] Add tensor format conversion (numpy ↔ PyTorch)
  - [ ] Maintain backward compatibility with TVM path
  - [ ] Add error handling for unsupported frameworks
  - [ ] Add logging for path selection

- [ ] **Step 5:** Modify `generate_initial_graph()`
  - [ ] Update condition to check both `compile_tvm_to_python` and `compile_tir_to_python`
  - [ ] Ensure proper tensor format handling for each path
  - [ ] Add validation for configuration consistency

- [ ] **Step 6:** Tensor Conversion Utilities
  - [ ] Create `to_onnx_inputs()` helper for ONNX transpiler
  - [ ] Create `to_framework_inputs()` helper for framework-specific conversion
  - [ ] Ensure consistent conversion back to PyTorch format
  - [ ] Add validation for tensor shapes and dtypes

- [ ] **Step 7:** Testing
  - [ ] Unit tests for `TIRCodeGenerator`
  - [ ] Unit tests for `generate_forge_module_from_tir()`
  - [ ] Integration tests with ONNX models
  - [ ] Integration tests with PyTorch models (when supported)
  - [ ] Verification tests against framework outputs
  - [ ] Performance comparison with TVM path
  - [ ] Error handling tests (invalid graphs, missing parameters, etc.)
  - [ ] Tensor conversion tests (numpy ↔ PyTorch)

- [ ] **Step 8:** Documentation
  - [ ] Update user documentation with TIR path usage
  - [ ] Add examples for both paths
  - [ ] Add migration guide from TVM to TIR path
  - [ ] Add troubleshooting guide
  - [ ] Update API documentation

---

## 12. Summary

### 12.1 Key Points

1. **TIRGraph Structure**: Well-defined intermediate representation with nodes, params, constants
2. **Code Generation**: Direct conversion from TIRGraph to Python Forge module code
3. **Integration**: Seamless integration into existing compilation pipeline via `convert_to_forge_module()`
4. **Flexibility**: Can be enabled/disabled via `CompilerConfig.use_tir_transpiler`
5. **Compatibility**: Maintains same interface as TVM path

### 12.2 Benefits

1. **Simpler Pipeline**: Fewer transformation steps than TVM path
2. **Framework-Specific**: Can leverage framework-specific optimizations
3. **Easier Debugging**: Direct mapping from framework to Forge
4. **Extensibility**: Easy to add new frontends (PyTorch, TensorFlow, etc.)

### 12.3 Key Features and Implementation Highlights

#### 12.3.1 Dual-Path Compilation Support

**Feature:** Support for both TVM and TIR transpiler paths with configuration flags.

**Implementation:**
- `CompilerConfig.compile_tvm_to_python`: Enable TVM path (existing)
- `CompilerConfig.compile_tir_to_python`: Enable TIR path (new)
- `CompilerConfig.use_tir_transpiler`: Select TIR path when both enabled
- Automatic path selection based on configuration

**Benefits:**
- Users can choose the best path for their use case
- Both paths can coexist in the codebase
- Easy migration between paths

#### 12.3.2 Tensor Format Conversion

**Feature:** Automatic tensor format conversion for each compilation path.

**Implementation:**
- **TVM Path:** Converts inputs to PyTorch tensors (`to_pt_tensors()`)
- **TIR Path:** Converts inputs to framework-native format (numpy for ONNX)
- **Post-Conversion:** All paths convert back to PyTorch for consistency

**Key Functions:**
- `to_pt_tensors()`: PyTorch tensor conversion (existing)
- `to_onnx_inputs()`: Numpy array conversion for ONNX (new)
- `to_framework_inputs()`: Framework-specific conversion (new)

**Benefits:**
- Correct format for each compilation path
- Consistent output format (PyTorch tensors)
- Transparent to users

#### 12.3.3 Enhanced Code Generation

**Feature:** Full-featured code generator matching TVM path capabilities.

**Implementation:**
- Parameter loading from framework models
- Constant handling (non-trainable parameters)
- Memory management (input deletion)
- Source layer tracking
- Error handling and validation
- Proper code formatting

**Benefits:**
- Production-ready generated code
- Easy debugging with source layer tracking
- Memory-efficient execution

#### 12.3.4 Framework-Specific Support

**Feature:** Support for multiple frameworks with framework-specific optimizations.

**Current Support:**
- **ONNX:** Full support with numpy input conversion
- **PyTorch:** Planned (direct tensor support)

**Future Support:**
- TensorFlow
- JAX
- PaddlePaddle

**Benefits:**
- Unified interface for all frameworks
- Framework-specific optimizations
- Easy to add new frameworks

### 12.4 Future Enhancements

1. **CPU/Device Partitioning**: Add support for CPU fallback operations
2. **Optimization Passes**: Add TIR-level optimization passes
3. **Multi-Output Support**: Enhanced support for multi-output operations
4. **Memory Optimization**: Advanced memory management strategies
5. **Verification**: Enhanced verification and debugging tools
6. **PyTorch Frontend**: Direct PyTorch model support (without ONNX conversion)
7. **TensorFlow Frontend**: Direct TensorFlow model support
8. **Graph Optimization**: TIRGraph-level optimizations before code generation
9. **Submodule Support**: Support for nested modules and subgraphs
10. **Dynamic Shapes**: Support for dynamic input shapes

---

**Document Version:** 1.1  
**Last Updated:** 2025-01-19  
**Status:** Enhanced Design Document with Tensor Conversion and Configuration Details


