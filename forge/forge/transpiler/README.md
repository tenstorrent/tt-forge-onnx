# Forge Transpiler Architecture

## Motivation: Why We Need the Transpiler

Apache TVM is a comprehensive deep learning compiler stack that combines both runtime components (execution engine, memory management, device management) and compilation components (graph optimization, code generation, operator fusion). However, for Forge, this dual nature creates a fundamental mismatch: we already have our own runtime, so TVM's runtime components are unnecessary, while TVM's compilation path adds complexity through multiple intermediate representations and limits our control over the conversion process.

Additionally, TVM's compilation pipeline doesn't provide the transparency and debuggability needed to understand exactly how framework operations are converted to Forge operations, making it difficult to verify correctness and debug conversion issues. We need a lightweight, purpose-built transpiler that focuses solely on compilation—converting framework models to Forge modules—while providing direct control over the conversion pipeline, framework-specific optimizations, and explicit handling of framework version differences (such as ONNX opset versions).

## Overview

The Forge Transpiler is a direct, transparent, and debuggable compilation system that converts machine learning models from ONNX format (with planned support for PaddlePaddle, TensorFlow, and other frameworks) into executable Forge modules. Unlike the traditional TVM-based compilation path, the transpiler provides a streamlined conversion pipeline: Framework Model → TIRGraph (Transpiler Intermediate Representation) → Python Forge Module, eliminating unnecessary intermediate representations and reducing compilation overhead.

The transpiler architecture is organized into framework-specific frontends (currently ONNX) that convert framework models into a framework-agnostic TIRGraph—a computational graph representation that captures nodes, inputs, outputs, parameters, and constants. This TIRGraph is then processed by the TranspilerCodeGenerator to map TIR operations to Forge operations and generate executable Python Forge module code.

The system handles the complexity of model conversion through well-defined stages: model validation, shape inference, operation conversion using opset-aware converters, graph construction with proper topology and name sanitization, and finally code generation with memory optimization. Each stage is designed to be transparent and debuggable—the TIRGraph can be executed directly using PyTorch for validation, built-in debug mode compares outputs with ONNX Runtime, and the generated Python code is human-readable—while maintaining explicit opset-aware design that handles multiple ONNX opset versions through version-specific converter logic.

The transpiler is seamlessly integrated into the Forge compilation pipeline as an alternative path to TVM, allowing users to choose between the transpiler path (for transparency, faster compilation, and explicit opset handling) or the TVM path (for advanced graph optimizations and multi-framework support), with both paths producing the same ForgeModule output that proceeds through Forge's graph optimization passes, MLIR compilation, and binary generation. The system is built with extensibility in mind—new operations can be added by implementing converter classes, and the architecture supports future expansion to other frameworks beyond ONNX.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Transpiler Working - Detailed Walkthrough](#transpiler-working---detailed-walkthrough)
4. [Forge Compilation Pipeline](#forge-compilation-pipeline)
5. [Compiling MNIST Model: TVM vs Transpiler Path](#compiling-mnist-model-tvm-vs-transpiler-path)
6. [Testing](#testing)

---

## Quick Start

### Simple Example

Here's a minimal example to get started with the transpiler:

```python
import torch
import onnx
import forge
from forge.config import CompilerConfig

# Create a simple PyTorch model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 16, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

# Export to ONNX
model = SimpleModel()
dummy_input = torch.randn(1, 1, 28, 28)
onnx_path = "simple_model.onnx"
torch.onnx.export(model, dummy_input, onnx_path, opset_version=11)

# Load ONNX model
onnx_model = onnx.load(onnx_path)
framework_model = forge.OnnxModule("simple_model", onnx_model)

# Configure transpiler path
compiler_cfg = CompilerConfig(
    compile_transpiler_to_python=True,
    compile_tvm_to_python=False,
)

# Compile using transpiler
compiled_model = forge.compile(
    framework_model,
    sample_inputs=[dummy_input],
    module_name="simple_model",
    compiler_cfg=compiler_cfg,
)

# Run inference
output = compiled_model(dummy_input)
print(f"Output shape: {output.shape}")
```

### Directory Structure

The transpiler codebase is organized as follows:

```
forge/forge/transpiler/
├── frontends/onnx/          # ONNX-specific frontend
│   ├── engine.py            # Main transpiler engine
│   ├── converters/         # Operation converters
│   └── utils/               # ONNX utilities
├── core/                    # Framework-agnostic core
│   ├── graph.py             # TIRGraph implementation
│   ├── node.py              # TIRNode base class
│   └── types.py             # Type utilities
├── operations/              # TIR operation implementations
└── codegen/                 # Code generation
    ├── transpiler_generator.py
    └── transpiler_to_forge.py
```

---

## Architecture Overview

The transpiler architecture is organized into four main layers, each with distinct responsibilities. This layered design enables framework-agnostic graph representation while supporting framework-specific conversion logic, making it extensible to multiple ML frameworks beyond ONNX.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           TRANSPILER ARCHITECTURE                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                          FRONTEND LAYER                                │  │
│  │                    (Framework-Specific: ONNX)                          │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Engine (ONNXToForgeTranspiler)                                  │  │  │
│  │  │  - Orchestrates conversion pipeline                              │  │  │
│  │  │  - Model validation (ONNX schema checking)                       │  │  │
│  │  │  - Shape inference (fills missing tensor shapes)                 │  │  │
│  │  │  - Opset extraction & converter map building                     │  │  │
│  │  │  - Parameter/constant distinction (heuristic-based)              │  │  │
│  │  │  - Name sanitization & uniqueness enforcement                    │  │  │
│  │  │  - Debug mode support (ONNX Runtime comparison)                  │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Converters (OnnxOpConverter subclasses)                         │  │  │
│  │  │  - One converter per ONNX operation type (20+ converters)        │  │  │
│  │  │  - Opset-aware conversion via version-specific patterns          │  │  │
│  │  │  - Returns TIR nodes or constant results                         │  │  │
│  │  │  - Handles attribute conversion from ONNX to PyTorch format      │  │  │
│  │  │  - Supports operation decomposition (e.g., Gemm -> MatMul+Add)   │  │  │
│  │  │  - Examples: ConvConverter, ReluConverter, GemmConverter, etc.   │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Utils (utils/)                                                  │  │  │
│  │  │  - naming.py: Name sanitization and uniqueness enforcement       │  │  │
│  │  │  - attributes.py: Attribute extraction and value parsing         │  │  │
│  │  │  - validation.py: ONNX model schema validation                   │  │  │
│  │  │  - onnx_graph.py: Graph manipulation helpers                     │  │  │
│  │  │  - io_builder.py: Input/output tensor extraction                 │  │  │
│  │  │  - debug/validator.py: ONNX Runtime comparison utilities         │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                     │                                        │
│                                     │ Converts                               │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                          CORE LAYER                                    │  │
│  │                  (Framework-Agnostic: TIR)                             │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │  TIRGraph (graph.py)                                             │  │  │
│  │  │  - Computational graph representation                            │  │  │
│  │  │  - Manages nodes in execution order                              │  │  │
│  │  │  - Tracks topology with producer and consumer mappings           │  │  │
│  │  │  - Stores parameters (trainable weights) and constants           │  │  │
│  │  │  - Maintains name mappings between original and sanitized names  │  │  │
│  │  │  - Uses topological sort for execution order                     │  │  │
│  │  │  - Memory management via activation dependency computation       │  │  │
│  │  │  - Direct execution with PyTorch backend                         │  │  │
│  │  │  - Debug mode: compares outputs with ONNX Runtime                │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │  TIRNode (node.py)                                               │  │  │
│  │  │  - Base class for all operations                                 │  │  │
│  │  │  - Stores node attributes: name, op_type, inputs, outputs, attrs │  │  │
│  │  │  - Manages inputs and outputs as ordered dictionaries            │  │  │
│  │  │  - Executes operations for validation                            │  │  │
│  │  │  - Generates code metadata for Forge module generation           │  │  │
│  │  │  - Converts attributes to Forge format                           │  │  │
│  │  │  - Tracks Forge operation names (e.g., "Conv2d", "Relu")         │  │  │
│  │  │  - Maintains source tracking for original framework node names   │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │  TensorInfo                                                      │  │  │
│  │  │  - Shape and dtype information                                   │  │  │
│  │  │  - Type conversion utilities (ONNX <-> PyTorch)                  │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                     │                                        │
│                                     │ Uses                                   │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                       OPERATIONS LAYER                                 │  │
│  │              (PyTorch-Compatible Implementations)                      │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Operation Nodes (operations/)                                   │  │  │
│  │  │  - Arithmetic: AddNode, SubNode, MulNode, DivNode, MatMulNode    │  │  │
│  │  │  - Convolution: Conv1dNode, Conv2dNode, Conv3dNode               │  │  │
│  │  │  - Activation: ReluNode, SigmoidNode, TanhNode, SoftmaxNode,     │  │  │
│  │  │                LogSoftmaxNode, LeakyReluNode, DropoutNode        │  │  │
│  │  │  - Pooling: MaxPool1d/2d/3dNode, AveragePool1d/2d/3dNode         │  │  │
│  │  │  - Shape: ReshapeNode, TransposeNode, SqueezeNode, UnsqueezeNode │  │  │
│  │  │  - Reduction: ReduceSumNode, ReduceMeanNode, ReduceMaxNode       │  │  │
│  │  │  - Other: ConcatNode, ClipNode, CastNode, IdentityNode, FullNode │  │  │
│  │  │  - All implement PyTorch-based execution for validation          │  │  │
│  │  │  - All generate code metadata for Forge module generation        │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                     │                                        │
│                                     │ Generates                              │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        CODEGEN LAYER                                   │  │
│  │              (Python Forge Module Generation)                          │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │  TranspilerCodeGenerator (transpiler_generator.py)               │  │  │
│  │  │  - Generates complete Python ForgeModule class                   │  │  │
│  │  │  - Writes header with required imports                           │  │  │
│  │  │  - Generates class definition with parameter/const registration  │  │  │
│  │  │  - Writes forward method with operations in topological order    │  │  │
│  │  │  - Memory optimization: Reference counting algorithm             │  │  │
│  │  │  - Generates parameter parser method                             │  │  │
│  │  │  - Handles device-specific data type conversion                  │  │  │
│  │  │  - Matches ForgeWriter code structure for consistency            │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │  transpiler_to_forge (transpiler_to_forge.py)                    │  │  │
│  │  │  - Main entry point for transpiler-based module generation       │  │  │
│  │  │  - Generates TIRGraph via ONNXToForgeTranspiler                  │  │  │
│  │  │  - Generates code via TranspilerCodeGenerator                    │  │  │
│  │  │  - Writes generated modules to file system                       │  │  │
│  │  │  - Dynamically imports generated modules at runtime              │  │  │
│  │  │  - Loads parameters from framework models                        │  │  │
│  │  │  - Verifies outputs against framework models when enabled        │  │  │
│  │  │  - Returns generated Forge modules and inputs                    │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Through Layers

The conversion process flows through these layers in a sequential manner:

1. **Frontend Layer** receives an ONNX model and converts it to TIRGraph
2. **Core Layer** maintains the TIRGraph structure and provides graph operations
3. **Operations Layer** provides executable implementations of operations
4. **Codegen Layer** transforms TIRGraph into Python Forge module code

### Component Responsibilities

#### Frontend Layer (`frontends/onnx/`)

The frontend layer handles all ONNX-specific aspects of conversion:

- **Engine** (`engine.py`): The `ONNXToForgeTranspiler` class orchestrates the entire conversion process through a multi-stage pipeline. It validates the ONNX model structure and schema, runs shape inference to determine tensor shapes, extracts the opset version from model metadata, builds opset-specific converter maps, processes initializers (distinguishing trainable parameters from constants), converts each ONNX node using appropriate converters, and constructs the final TIRGraph with proper topology and name mappings. Supports debug mode for ONNX Runtime comparison and parameter freezing options.

- **Converters** (`converters/`): Each ONNX operation type has a corresponding converter class (e.g., `ConvConverter`, `ReluConverter`, `GemmConverter`) that inherits from `OnnxOpConverter`. These converters are opset-aware, explicitly handling different ONNX opset versions (e.g., `Conv` uses `group` in opset < 11, `groups` in opset >= 11). They convert ONNX operations into one or more TIR nodes, handling attribute conversion from ONNX format to PyTorch-compatible format, and performing operation decomposition when necessary (e.g., `Gemm` → `MatMul` + `Add`). Converters return either `List[TIRNode]` for normal operations or `ConstantResult` for constant values.

- **Utils** (`utils/`): Helper utilities organized by functionality: naming utilities (`sanitize_name()`, `ensure_unique_name()`, `generate_clean_variable_name()`) convert ONNX names to valid Python identifiers; attribute utilities (`extract_attributes()`, `extract_attr_value()`) extract and convert ONNX attributes to PyTorch-friendly names; validation utilities check ONNX model schema and structure; graph manipulation utilities handle input/output extraction and graph traversal; and debug utilities compare TIR outputs with ONNX Runtime for validation.

#### Core Layer (`core/`)

The core layer provides framework-agnostic abstractions that work across all frontends:

- **TIRGraph** (`graph.py`): Represents the computational graph in a framework-agnostic way. It maintains nodes in execution order, topology maps (`producer_map` and `consumer_map`) tracking tensor dependencies, parameters (trainable weights) and constants (non-trainable values), and bidirectional name mappings between original frontend names and sanitized names. The graph can be executed directly using PyTorch for validation via the `run()` method, supports topological sorting using Kahn's algorithm, computes activation dependencies for memory management, and includes debug mode for comparing outputs with ONNX Runtime.

- **TIRNode** (`node.py`): Base class for all operations in the TIRGraph. Stores node metadata (name, op_type, inputs/outputs as `OrderedDict[str, TensorInfo]`, attributes), provides execution interface via `eval()` method that executes operations using PyTorch, provides code generation interface via `emit()` method that returns operation metadata for Forge API calls, and handles attribute conversion from PyTorch-compatible format to Forge-specific format via `convert_attrs_to_forge_attrs()` method. Subclasses can override attribute conversion for custom transformations.

- **Types** (`types.py`): Provides `TensorInfo` class for representing tensor metadata (name, shape with support for dynamic dimensions, ONNX dtype, and derived PyTorch dtype). Includes `onnx_dtype_to_torch_dtype()` utility function that converts ONNX `TensorProto.DataType` integer enums to PyTorch dtypes, supporting FLOAT, INT32, INT64, BOOL, FLOAT16, DOUBLE, and other common types.

#### Operations Layer (`operations/`)

Operations are implemented as `TIRNode` subclasses using PyTorch, enabling direct execution for validation while maintaining framework-agnostic representation:

- **Operation Categories**: Includes arithmetic operations (`AddNode`, `SubNode`, `MulNode`, `DivNode`, `MatMulNode`), convolution operations (`Conv1dNode`, `Conv2dNode`, `Conv3dNode` - automatically selected based on input dimensions), activation functions (`ReluNode`, `SigmoidNode`, `TanhNode`, `SoftmaxNode`, `LogSoftmaxNode`, `LeakyReluNode`, `DropoutNode`), pooling operations (`MaxPool1d/2d/3dNode`, `AveragePool1d/2d/3dNode`), shape operations (`ReshapeNode`, `TransposeNode`, `SqueezeNode`, `UnsqueezeNode`), reduction operations (`ReduceSumNode`, `ReduceMeanNode`, `ReduceMaxNode`), and other operations (`ConcatNode`, `ClipNode`, `CastNode`, `IdentityNode`, `FullNode`).

- **Implementation Pattern**: All operations inherit from `TIRNode` and implement `eval()` method using PyTorch functions (e.g., `torch.nn.functional.relu()` for `ReluNode`), implement `emit()` method that returns metadata dictionary for code generation (typically uses base class implementation), and can optionally override `convert_attrs_to_forge_attrs()` for custom attribute transformations.

- **Usage**: Operations can be executed directly for validation (e.g., `node.eval({"input": tensor})` returns output dictionary), and they emit metadata that describes how they should be generated in Python code (e.g., `forge.op.Relu(...)`).

#### Codegen Layer (`codegen/`)

The codegen layer transforms `TIRGraph` into executable Python code that implements a `ForgeModule`:

- **TranspilerCodeGenerator** (`transpiler_generator.py`): Generates complete Python `ForgeModule` class from `TIRGraph`. It writes header with required imports (`torch`, `forge`, `forge.op`, `ForgeModule`), generates class definition with `__init__` method that registers parameters (trainable weights) and constants (non-trainable values) with proper device-specific data format handling, generates `forward()` method with operations in topological order formatted as Forge API calls (e.g., `forge.op.Conv2d(...)`), implements reference counting algorithm (`_compute_inputs_to_delete()`) to determine which intermediate activations can be safely deleted for memory optimization, and generates `process_framework_parameters()` method that loads weights from ONNX model into the Forge module. The generated code matches `ForgeWriter` structure for consistency.

- **transpiler_to_forge** (`transpiler_to_forge.py`): Orchestrates the complete pipeline from framework model to executable Forge module. It generates `TIRGraph` via `ONNXToForgeTranspiler.transpile()`, generates Python code via `TranspilerCodeGenerator.generate()`, writes generated code to file system (`generated_modules/{graph_name}.py`), dynamically imports the generated module at runtime using `importlib`, instantiates the `ForgeModule` class from imported module, loads parameters from ONNX model using `process_framework_parameters()`, and optionally verifies outputs by comparing Forge module outputs with framework outputs. Returns tuple of `(forge_modules, forge_inputs)` for integration with Forge compilation pipeline.

---

## Transpiler Working - Detailed Walkthrough

### High-Level Conversion Pipeline

The transpiler converts ONNX models to Forge modules through a well-defined pipeline with five main stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ONNX Model                                          │
│  (ModelProto: nodes, initializers, inputs, outputs, metadata)               │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              Stage 1: Model Validation & Preparation                        │
│  • Validate ONNX model structure and schema                                 │
│  • Run shape inference to determine tensor shapes                           │
│  • Extract opset version from model metadata                                │
│  • Build converter map for the specific opset version                       │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              Stage 2: Process Initializers                                  │
│  • Distinguish parameters (trainable weights) from constants                │
│  • Convert ONNX tensors to PyTorch tensors                                  │
│  • Store in TIRGraph.params (trainable) or TIRGraph.constants               │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              Stage 3: Convert ONNX Nodes → TIR Nodes                        │
│  For each ONNX node:                                                        │
│    1. Look up converter for operation type                                  │
│    2. Extract input/output tensor information (shapes, dtypes)              │
│    3. Extract node attributes                                               │
│    4. Call converter with opset version                                     │
│    5. Converter returns List[TIRNode] or ConstantResult                     │
│    6. Add nodes to graph with name sanitization                             │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              Stage 4: Build TIRGraph                                        │
│  • Sanitize input/output names (convert to valid Python identifiers)        │
│  • Build topology maps (producer/consumer relationships)                    │
│  • Compute activation dependencies for memory management                    │
│  • Store name mappings (original ↔ sanitized)                               │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TIRGraph                                            │
│  (Framework-agnostic intermediate representation)                           │
│  • Nodes in topological order                                               │
│  • Parameters and constants                                                 │
│  • Topology maps                                                            │
│  • Name mappings                                                            │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              Stage 5: Generate & Instantiate Forge Module                   │
│  • Generate ForgeModule class definition                                    │
│  • Generate __init__ with parameter/constant registration                   │
│  • Generate forward() method with operations in topological order           │
│  • Compute memory optimization (which activations to delete)                │
│  • Generate process_framework_parameters() method                           │
│  • Write code to file (generated_modules/model.py)                          │
│  • Dynamically import module using importlib                                │
│  • Instantiate ForgeModule class                                            │
│  • Load parameters from ONNX model                                          │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ForgeModule                                            │
│  (Ready for execution)                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Stage Explanations

#### Stage 1: Model Validation & Preparation

Before conversion begins, the transpiler validates and prepares the ONNX model through a systematic validation and setup process:

**Model Validation**: The ONNX model undergoes comprehensive validation to ensure it conforms to the ONNX specification:
- **Structural Validation**: Verifies that the model contains all required components including version information, opset definitions, and graph structure
- **Schema Validation**: Checks that all operations in the graph conform to ONNX operator schemas for the specified opset version, ensuring operations are used correctly
- **Type Checking**: Validates that tensor data types are consistent throughout the graph—input types must match what operations expect, and operation outputs must match what their consumers expect
- **Graph Integrity**: Verifies that the graph is well-formed with no dangling references, all tensor references point to valid sources, and all nodes have proper input/output connections
- **Early Error Detection**: If validation fails, the process stops immediately with a clear error message, preventing wasted computation on invalid models

**Shape Inference**: The transpiler determines tensor shapes throughout the graph by propagating shape information:
- **Shape Propagation**: Traverses the graph in execution order, starting from model inputs and propagating shape information through each operation to determine output shapes
- **Operation Shape Rules**: Each operation type has mathematical rules for computing output shapes from input shapes and operation parameters (e.g., convolution output height depends on input height, padding, kernel size, stride, and dilation)
- **Dynamic Dimensions**: Handles cases where dimensions cannot be determined statically (e.g., variable batch sizes or sequence lengths), marking them as dynamic for later handling
- **Graceful Degradation**: If shape inference encounters issues, the transpiler continues with available shape information rather than failing, using whatever shape data is present in the model

**Opset Extraction**: The ONNX opset version is identified from the model metadata:
- **Version Reading**: Extracts the primary opset version from the model's metadata, which determines which version of each operation specification to use
- **Default Behavior**: If no opset is explicitly specified, assumes the oldest supported version (opset 1) for backward compatibility
- **Version Handling**: While ONNX models can specify multiple opsets for different operation domains, the transpiler uses the primary opset for the main operation set
- **Version Impact**: Different opset versions may change attribute names, move attributes to become inputs, introduce new operations, or modify operation behavior, so the correct version must be identified

**Converter Map Building**: A lookup table is created mapping each ONNX operation type to its corresponding converter:
- **Map Creation**: Builds a dictionary that maps operation type names (like "Conv", "Relu", "Gemm") to their converter functions
- **Opset Binding**: Each converter is configured for the specific opset version, ensuring version-specific conversion logic is applied
- **Converter Coverage**: Registers converters for all supported operation categories including arithmetic operations, activation functions, convolutions, pooling operations, reduction operations, shape manipulation operations, and constant operations
- **Fast Lookup**: The map structure enables instant lookup of converters during node conversion, making the conversion process efficient even for large models with many operations

#### Stage 2: Process Initializers

Initializers are pre-computed tensor values stored in the ONNX model. The transpiler processes each initializer to distinguish trainable parameters from non-trainable constants and convert them to PyTorch tensors:

**Initializer Processing**: The transpiler iterates through all initializers in the ONNX model:
- **Tensor Extraction**: Extracts tensor data from the ONNX format, handling various storage methods (embedded data, external files, etc.)
- **Type Conversion**: Converts tensor data types from ONNX format to PyTorch format, mapping ONNX type enums to PyTorch dtype objects
- **Shape Preservation**: Maintains the original tensor dimensions and shape information from the ONNX model
- **Name Tracking**: Uses the tensor name as an identifier for storage and later reference

**Parameter vs Constant Distinction**: The transpiler uses heuristics to classify initializers as either trainable parameters or non-trainable constants:
- **Constant Indicators**:
  - Names containing "constant" (case-insensitive) typically indicate non-trainable values
  - Scalar tensors or single-element tensors are usually constants rather than weights
  - Integer or boolean type tensors that aren't weights or biases are often constants (like indices or masks)
  - Explicit naming patterns that suggest constant values
- **Parameter Indicators**:
  - Weight tensors typically have multi-dimensional shapes matching layer configurations (e.g., convolution weights have shape matching input/output channels and kernel dimensions)
  - Bias tensors are usually one-dimensional with shape matching output channels
  - Any tensor that doesn't match constant patterns is treated as a trainable parameter
- **Storage Separation**: Parameters are stored separately from constants, allowing the code generator to handle them differently (parameters need gradient tracking, constants don't)
- **Parameter Freezing Option**: The transpiler supports an option to treat all initializers as constants, which is useful for inference-only models where training is not needed

**Tensor Conversion**: During conversion, several transformations occur:
- **Dtype Mapping**: ONNX data types (like FLOAT, INT32, INT64, BOOL, FLOAT16, DOUBLE) are mapped to their PyTorch equivalents, ensuring type compatibility
- **Memory Layout**: Tensors are stored in contiguous memory using row-major (C-style) layout, matching ONNX's storage format
- **Device Placement**: Tensors are initially created on CPU; device placement for execution happens later in the Forge compilation pipeline

#### Stage 3: Convert ONNX Nodes → TIR Nodes

Each ONNX operation node is converted to one or more TIR nodes using opset-aware converters. This stage processes nodes sequentially, maintaining graph topology and name mappings:

**Node Conversion Process**: The transpiler iterates through all operation nodes in the ONNX graph in order:
- **Node Information Extraction**: For each node, extracts the operation type, input tensor names, output tensor names, and tracks the node's position in the graph

**Step 1: Converter Lookup**:
- **Operation Type Matching**: Looks up the appropriate converter for the operation type from the converter map built in Stage 1
- **Error Handling**: If no converter exists for an operation type, the conversion fails with a clear error message indicating the unsupported operation
- **Version-Specific Logic**: The converter retrieved is already configured for the specific opset version, ensuring the correct conversion logic is applied

**Step 2: Tensor Information Extraction**:
- **Input Tensor Metadata**: For each input to the operation, looks up tensor shape and type information from either shape inference results or initializer data
- **Output Tensor Metadata**: Determines output tensor shapes and types based on operation semantics and input shapes
- **Shape Calculation**: Uses operation-specific rules to compute output shapes (e.g., convolution output dimensions depend on input size, kernel size, padding, stride, and dilation)
- **Metadata Storage**: Creates tensor metadata objects that capture name, shape (which may include unknown dimensions), and data type information

**Step 3: Attribute Extraction**:
- **Attribute Reading**: Extracts all attributes associated with the operation node
- **Name Normalization**: Converts ONNX attribute names to PyTorch-compatible names (e.g., "axis" becomes "dim", "dilations" becomes "dilation")
- **Format Conversion**: Transforms attribute values from ONNX format to PyTorch format (e.g., lists become tuples, padding format changes from ONNX's begin/end pairs to PyTorch's symmetric pairs)
- **Type Conversion**: Converts attribute values to appropriate Python types (integers, floats, strings, or tensors)
- **Default Application**: Applies operation-specific default values when attributes are missing (e.g., convolution defaults to stride of 1, no padding, no dilation, single group)

**Step 4: Converter Invocation**:
- **Converter Execution**: Calls the converter with the node information, tensor metadata, and converted attributes
- **Opset-Aware Processing**: The converter uses the opset version to handle version-specific differences (e.g., older opsets use different attribute names than newer ones)
- **Attribute Transformation**: The converter transforms ONNX attributes to PyTorch-compatible formats, handling differences in representation
- **Operation Decomposition**: Some converters break complex operations into simpler ones (e.g., general matrix multiply becomes matrix multiplication plus addition, asymmetric padding becomes a padding operation followed by symmetric convolution)
- **TIR Node Creation**: The converter creates one or more TIR nodes representing the operation with all necessary information

**Step 5: Result Processing**:
- **Result Type Determination**: Checks whether the converter returned computational nodes or a constant value
- **TIR Nodes Handling**: For computational nodes:
  - Validates that nodes have all required information and valid connections
  - Sanitizes node names to be valid Python identifiers
  - Updates name mappings between original ONNX names and sanitized names
  - Handles operations that produce multiple outputs (like split operations)
  - Adds nodes to the graph, which automatically updates topology tracking
  - Records mappings for debug mode comparison
- **Constant Handling**: For constant values:
  - Extracts the constant tensor value and output name
  - Stores the constant directly in the graph's constants dictionary
  - Updates name mappings
  - Skips creating a computational node since constants don't need execution

**Opset-Specific Handling**: Different ONNX opset versions handle operations differently:
- **Convolution**: Older opsets use "group" attribute, newer opsets use "groups"
- **Reshape**: Older opsets have shape as an attribute, newer opsets have shape as an input tensor
- **Squeeze/Unsqueeze**: Older opsets have axes as attributes, newer opsets have axes as input tensors
- **Padding**: Older opsets use different attribute names than newer opsets

**Operation Decomposition**: Some operations are decomposed into simpler operations:
- **General Matrix Multiply**: Broken down into matrix multiplication followed by addition
- **Asymmetric Padding**: Split into a padding operation followed by symmetric convolution
- **Multi-Output Operations**: Operations that produce multiple outputs create multiple TIR nodes, one for each output

#### Stage 4: Build TIRGraph

The TIRGraph is constructed with proper topology, name mappings, and memory management information. This stage finalizes the graph structure and prepares it for execution or code generation:

**Graph Initialization**: The TIRGraph is created with all necessary data structures:
- **Basic Information**: Stores the graph name, framework identifier, debug mode setting, and reference to the original ONNX model for debugging
- **Node Collection**: Maintains a list of all operation nodes that will be populated during conversion
- **Parameter and Constant Storage**: Separate dictionaries store trainable parameters and non-trainable constants
- **Topology Tracking**: Creates maps to track which nodes produce which tensors and which nodes consume which tensors
- **Name Mapping**: Maintains bidirectional mappings between original ONNX names and sanitized Python-compatible names

**Name Sanitization Process**: ONNX names often contain characters invalid for Python identifiers, so they must be sanitized:
- **Input Name Processing**: For each graph input:
  - Extracts the original name from the ONNX model
  - Replaces invalid characters (colons, slashes, dots, dashes, spaces) with underscores
  - Removes consecutive underscores and trims leading/trailing underscores
  - Ensures names don't start with digits (prepends prefix if needed)
  - Ensures uniqueness by appending numeric suffixes if names collide
  - Preserves input names more closely since users may reference them
  - Stores mappings between original and sanitized names
- **Output Name Generation**: For graph outputs:
  - Generates clean, readable names based on operation type (e.g., "conv2d_0", "relu_1")
  - Uses operation type and a counter to ensure uniqueness
  - Makes generated code more readable and maintainable
- **Node Name Sanitization**: Each node's name is sanitized when added to the graph, ensuring all names are valid Python identifiers

**Topology Map Construction**: The graph builds maps to track tensor dependencies:
- **Producer Map**: Records which node produces each tensor, enabling quick lookup of tensor sources
- **Consumer Map**: Records all nodes that consume each tensor, enabling detection of multi-consumer scenarios
- **Incremental Building**: These maps are built incrementally as nodes are added, keeping the graph structure up-to-date
- **Topology Validation**: After all nodes are added, validates that:
  - All tensor references point to valid sources (inputs, initializers, or node outputs)
  - The graph contains no cycles (would prevent execution)
  - All graph outputs are actually produced by some node

**Activation Dependency Computation**: The graph computes which activations are needed by which operations:
- **Reverse Dependency Graph**: Builds a map showing which tensors depend on which other tensors
- **Dependency Tracking**: For each operation, tracks which input tensors are needed to compute each output tensor
- **Memory Management**: This information enables:
  - Garbage collection during graph execution (tensors can be deleted when no longer needed)
  - Memory optimization in code generation (reference counting determines when to free memory)
  - Activation lifetime tracking (knowing when an intermediate result can be safely discarded)

**Graph Finalization**: The graph undergoes final validation and setup:
- **Topological Sort**: Computes execution order ensuring all dependencies are satisfied, and verifies the graph is acyclic
- **Output Validation**: Verifies that all declared graph outputs are actually reachable and produced by valid nodes
- **Name Mapping Completion**: Finalizes all name mappings between original ONNX names and sanitized names
- **Debug Mode Setup**: If debug mode is enabled, stores the original ONNX model and builds mappings for comparing outputs with ONNX Runtime

#### Stage 5: Generate & Instantiate Forge Module

The TIRGraph is converted to executable Python code that implements a ForgeModule. This stage produces human-readable Python code and creates an executable module instance:

**Code Generation**: The code generator creates a complete Python class from the TIRGraph:
- **Header Generation**: Writes necessary Python imports including PyTorch, Forge framework, and Forge operations
- **Class Definition**: Generates a ForgeModule subclass with:
  - **Initialization Method**: Registers all parameters (trainable weights) and constants (non-trainable values) with their shapes and data types, handling device-specific data format requirements
  - **Forward Method**: Generates the computation graph with operations in topological order, formatted as Forge API calls
  - **Parameter Parser Method**: Generates code to load weights from the ONNX model into the Forge module

**Memory Optimization**: The code generator implements a reference counting algorithm to optimize memory usage:
- **Reference Counting**: Counts how many times each intermediate tensor is used by subsequent operations
- **Deletion Marking**: Identifies which tensors can be safely deleted after their last use
- **Code Insertion**: Inserts code to delete unused intermediate activations, reducing memory footprint during execution
- **Safety**: Only deletes intermediate activations, never model inputs, parameters, or constants

**File I/O**: The generated Python code is written to disk:
- **Directory Management**: Creates a directory for generated modules if it doesn't exist
- **File Writing**: Writes the complete Python code to a file with a name based on the model name
- **File Organization**: Generated files are stored in a dedicated directory for easy management

**Dynamic Import**: The generated module is loaded at runtime:
- **Module Loading**: Uses Python's import system to dynamically load the generated module file
- **Code Execution**: Executes the module code, which defines the ForgeModule class
- **Class Availability**: Makes the ForgeModule class available for instantiation

**Module Instantiation**: The ForgeModule class is instantiated:
- **Instance Creation**: Creates an instance of the generated ForgeModule class
- **Structure Initialization**: The initialization method sets up the module structure, registering parameters and constants (but not loading values yet)
- **Empty State**: At this point, the module has the correct structure but no weights loaded

**Parameter Loading**: Weights are transferred from the ONNX model to the Forge module:
- **Weight Extraction**: Extracts tensor values from the ONNX model's initializer list
- **Name Mapping**: Uses the name mappings to find the correct parameters/constants using original ONNX names
- **Value Assignment**: Loads the extracted values into the Forge module's parameter and constant storage
- **Special Handling**: Handles edge cases like scalar constants that may need reshaping
- **Error Handling**: Provides clear error messages if parameters or constants are missing

**Input Conversion**: Input tensors are converted to Forge format:
- **Tensor Transformation**: Converts PyTorch input tensors to Forge Tensor objects
- **Device Placement**: Sets up appropriate device placement for the tensors
- **Shape Preservation**: Maintains tensor shapes and data types during conversion

**Verification (Optional)**: The transpiler can verify correctness at multiple stages:
- **TIRGraph Verification**: Executes the TIRGraph and compares outputs with the original framework model
- **Forge Module Verification**: Executes the generated Forge module and compares outputs with the framework model
- **Tolerance-Based Comparison**: Uses numerical tolerance to account for floating-point differences
- **Error Reporting**: Provides detailed error messages if outputs don't match within tolerance

**Integration**: The generated ForgeModule integrates with the Forge compilation pipeline:
- **Pipeline Entry**: The module is returned to the Forge compilation system
- **Next Steps**: The module proceeds through Forge's graph optimization passes, MLIR compilation, and binary code generation
- **Configuration**: The transpiler path is controlled by compiler configuration settings

---

## Forge Compilation Pipeline

The Forge compilation system supports two parallel paths for converting framework models to executable binaries: the **TVM Path** and the **Transpiler Path**. Both paths converge at the ForgeModule stage and proceed through the same downstream compilation stages.

### Complete Compilation Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              User Code                                       │
│  forge.compile(onnx_model, sample_inputs, compiler_cfg=...)                  │
└──────────────────────────────┬──────────────────────────────────────────────-┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        forge.compile()                                       │
│  • Wraps model in OnnxModule                                                 │
│  • Creates CompilerConfig (if not provided)                                  │
│  • Creates VerifyConfig (if not provided)                                    │
└──────────────────────────────┬─────────────────────────────────────────────--┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                  convert_to_forge_module()                                   │
│  ┌──────────────────────────────────────────────────────────────────────-┐   │
│  │  Route Selection Based on CompilerConfig:                             │   │
│  │                                                                       │   │
│  │  if compile_transpiler_to_python == True:                             │   │
│  │      → Transpiler Path                                                │   │
│  │  elif compile_tvm_to_python == True:                                  │   │
│  │      → TVM Path                                                       │   │
│  │  else:                                                                │   │
│  │      → Error: Must specify one path                                   │   │
│  └──────────────────────────────────────────────────────────────────────-┘   │
└──────────────────────────────┬──────────────────────────────────────────────-┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
    ┌───────────────────────┐     ┌───────────────────────┐
    │  TRANSPILER PATH      │     │      TVM PATH         │
    │                       │     │                       │
    │  ONNX Model           │     │  ONNX/PyTorch/        │
    │      ↓                │     │  PaddlePaddle/        │
    │  ONNXToForgeTranspiler│     │  TensorFlow/JAX       │
    │  • Validation         │     │      ↓                │
    │  • Shape inference    │     │  TVM Relay IR         │
    │  • Opset extraction   │     │      ↓                │
    │      ↓                │     │  TVM Compile Passes   │
    │  TIRGraph             │     │  • Graph optimization │
    │  • Framework-agnostic │     │  • Operation fusion   │
    │  • Nodes, params,     │     │      ↓                │
    │    constants          │     │  JSON Graphs          │
    │      ↓                │     │      ↓                │
    │  CodeGenerator        │     │  ForgeWriter          │
    │  • Code generation    │     │  • Code generation    │
    │  • Memory optimization│     │      ↓                │
    │      ↓                │     │                       │
    │  ForgeModule          │     │  ForgeModule          │
    └───────────────────────┘     └───────────────────────┘
                │                             │
                └──────────────┬──────────────┘
                               │
                               ▼
                ┌───────────────────────────────┐
                │      ForgeModule              │
                │  (Unified Output from Both)   │
                └──────────────┬────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              generate_initial_graph()                                        │
│  • Converts ForgeModule to Forge Graph                                       │
│  • Extracts operations, parameters, and topology                             │
│  • Creates initial computational graph                                       │
└──────────────────────────────┬──────────────────────────────────────────────-┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              Forge Graph Passes                                              │
│  • Post-initial graph passes (structure validation, transformations)         │
│  • ConstEval pass (constant folding and evaluation)                          │
│  • Pattern matcher (operation pattern recognition and optimization)          │
│  • Optimization passes (graph-level optimizations, operation fusion)         │
│  • Autograd pass (automatic differentiation, if training=True)               │
│  • Post-autograd passes (post-autograd optimizations)                        │
│  • Pre-lowering passes (final graph transformations)                         │
│  • Graph splitting (multi-device partitioning and device assignment)         │
└──────────────────────────────┬──────────────────────────────────────────────-┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              MLIR Compilation                                                │
│  • Lower Forge Graph to MLIR                                                 │
│  • MLIR optimization passes                                                  │
│  • Device-specific code generation                                           │
└──────────────────────────────┬──────────────────────────────────────────────-┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              Binary Generation                                               │
│  • Generate executable binary                                                │
│  • Package with metadata                                                     │
└──────────────────────────────┬────────────────────────────────────────────-──┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    CompiledModel                                             │
│  (Ready for deployment and execution)                                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Compiling MNIST Model: TVM vs Transpiler Path

This section shows how to compile the MNIST model using both TVM and transpiler paths. The MNIST test (`test_mnist_onnx.py`) uses pytest parametrize to test both compilation paths in a single test file, allowing easy comparison between TVM and transpiler paths.

### Combined Test Approach

The MNIST test uses `@pytest.mark.parametrize` to run the same test with both compilation paths:

```python
@pytest.mark.parametrize("use_transpiler", [False, True], ids=["tvm", "transpiler"])
def test_mnist(forge_tmp_path, use_transpiler):
    # ... test setup ...

    if use_transpiler:
        # Transpiler path configuration
        compiler_cfg = CompilerConfig(
            compile_transpiler_to_python=True,
            compile_tvm_to_python=False,
            transpiler_enable_debug=True,
        )
        verify_cfg = DeprecatedVerifyConfig(
            verify_transpiler_graph=True,
            verify_forge_codegen_vs_framework=True,
        )
    else:
        # TVM path configuration (default)
        compiler_cfg = CompilerConfig()
        verify_cfg = DeprecatedVerifyConfig(verify_forge_codegen_vs_framework=True)

    # Compile and verify...
```

This approach ensures both paths are tested with the same model and inputs, making it easy to compare results and verify that both compilation paths produce equivalent outputs.

### Compiling MNIST with Transpiler Path

The transpiler path configuration uses the following settings:

```python
import torch
import onnx
import forge
from forge.config import CompilerConfig
from forge.verify.config import DeprecatedVerifyConfig

# Load ONNX model
onnx_model = onnx.load("mnist.onnx")
framework_model = forge.OnnxModule("mnist", onnx_model)
inputs = [torch.randn(1, 1, 28, 28)]

# Compiler config for transpiler path
compiler_cfg = CompilerConfig(
    compile_transpiler_to_python=True,  # Enable transpiler path
    compile_tvm_to_python=False,         # Disable TVM path
    transpiler_enable_debug=True,       # Enable debug mode (ONNX Runtime comparison)
)

# Verify config
verify_cfg = DeprecatedVerifyConfig(
    verify_transpiler_graph=True,              # Verify TIRGraph outputs vs framework
    verify_forge_codegen_vs_framework=True,    # Verify ForgeModule outputs vs framework
)

# Compile using transpiler
compiled_model = forge.compile(
    framework_model,
    sample_inputs=inputs,
    module_name="mnist_transpiler",
    compiler_cfg=compiler_cfg,
    verify_cfg=verify_cfg,
)
```

**CompilerConfig Usage:**
- `compile_transpiler_to_python=True`: Routes compilation through transpiler path (ONNX → TIRGraph → ForgeModule)
- `compile_tvm_to_python=False`: Disables TVM path (required when using transpiler)
- `transpiler_enable_debug=True`: Enables debug mode for ONNX Runtime comparison and detailed debugging

**VerifyConfig Usage:**
- `verify_transpiler_graph=True`: Compares TIRGraph outputs with ONNX Runtime outputs after transpiler conversion
- `verify_forge_codegen_vs_framework=True`: Compares generated ForgeModule outputs with framework outputs

### Compiling MNIST with TVM Path

The TVM path configuration uses default settings:

```python
import torch
import onnx
import forge

# Load ONNX model
onnx_model = onnx.load("mnist.onnx")
framework_model = forge.OnnxModule("mnist", onnx_model)
inputs = [torch.randn(1, 1, 28, 28)]

# Compile using TVM path (default - no CompilerConfig needed)
# Note: forge.compile() accepts both onnx.ModelProto and forge.OnnxModule
compiled_model = forge.compile(
    onnx_model,  # Can also use framework_model (forge.OnnxModule)
    sample_inputs=inputs,
    module_name="mnist_tvm",
)
```

**Default Configuration:**
- `compile_tvm_to_python=True` (default): Routes compilation through TVM path (ONNX → TVM Relay IR → JSON Graphs → ForgeModule)
- `compile_transpiler_to_python=False` (default): Transpiler path is disabled
- No explicit `CompilerConfig` needed - TVM is the default path
- **Note**: `forge.compile()` accepts both `onnx.ModelProto` and `forge.OnnxModule` - both are automatically wrapped internally

### Summary

| Path | CompilerConfig | Key Settings | When to Use |
|------|---------------|--------------|-------------|
| **Transpiler** | `compile_transpiler_to_python=True`<br>`compile_tvm_to_python=False`<br>`transpiler_enable_debug=True` | Uses direct ONNX → TIRGraph → ForgeModule conversion | • ONNX models only<br>• Need faster compilation<br>• Want transparent conversion<br>• Need explicit opset handling |
| **TVM** | Default (no config needed)<br>`compile_tvm_to_python=True` (default) | Uses ONNX → TVM Relay IR → ForgeModule conversion | • Multiple frameworks (PyTorch, TensorFlow, etc.)<br>• Need advanced optimizations<br>• Model has unsupported operations |

---

## Testing

### Operation Tests

**Location**: `forge/test/transpiler/ops/`

**What is tested**: Individual ONNX operations are tested to verify correct conversion to TIR nodes. Each operation has a dedicated test file (e.g., `test_relu.py`, `test_conv.py`, `test_add.py`).

**Test coverage**: Tests verify operations across multiple opset versions, input shapes, data types, and edge cases. Tests compare TIRGraph outputs with ONNX Runtime outputs to ensure correctness.

**Supported operations**: Add, Sub, Mul, Div, Conv, Relu, Sigmoid, Tanh, Softmax, LogSoftmax, MaxPool, AvgPool, Gemm, Reshape, Transpose, Flatten, Squeeze, Unsqueeze, Concat, Pad, Clip, Dropout, and reduction operations.

### Model Tests

**Location**: `forge/test/transpiler/models/` and `forge/test/models/onnx/vision/mnist/`

**What is tested**: End-to-end conversion of complete models from PyTorch → ONNX → TIRGraph → ForgeModule. Tests verify the full conversion pipeline including code generation and compare outputs at multiple stages.

**MNIST Test**: The MNIST model test (`forge/test/models/onnx/vision/mnist/test_mnist_onnx.py`) uses pytest parametrize to test both TVM and transpiler compilation paths in a single test file. The test runs twice with `use_transpiler=False` (TVM path) and `use_transpiler=True` (transpiler path), ensuring both paths produce equivalent results:

```bash
# Run both paths
pytest forge/test/models/onnx/vision/mnist/test_mnist_onnx.py

# Run only TVM path
pytest forge/test/models/onnx/vision/mnist/test_mnist_onnx.py::test_mnist[tvm]

# Run only transpiler path
pytest forge/test/models/onnx/vision/mnist/test_mnist_onnx.py::test_mnist[transpiler]
```

This parametrized approach allows easy comparison between compilation paths and ensures both are tested with identical inputs and verification logic.
