# Forge Transpiler Architecture

## Motivation: Why We Need the Transpiler

Apache TVM is a comprehensive deep learning compiler stack that combines both runtime and compilation components. However, for Forge, this dual nature creates a fundamental mismatch: we already have our own runtime, so TVM's runtime components are unnecessary, while TVM's compilation path adds complexity through multiple intermediate representations and limits our control over the conversion process.

Additionally, TVM's compilation pipeline doesn't provide the transparency and debuggability needed to understand exactly how framework operations are converted to Forge operations, making it difficult to verify correctness and debug conversion issues. We need a lightweight, purpose-built transpiler that focuses solely on compilation—converting framework models to Forge modules—while providing direct control over the conversion pipeline, framework-specific optimizations, and explicit handling of framework version differences (such as ONNX opset versions).

## Overview

The Forge Transpiler is a direct, transparent, and debuggable compilation system that converts machine learning models from ONNX format (with planned support for PaddlePaddle, TensorFlow, and other frameworks) into executable Forge modules. Unlike the traditional TVM-based compilation path, the transpiler provides a streamlined conversion pipeline: Framework Model → TIRGraph (Transpiler Intermediate Representation) → Python Forge Module, eliminating unnecessary intermediate representations and reducing compilation overhead.

The transpiler architecture is organized into framework-specific frontends (currently ONNX) that convert framework models into a framework-agnostic TIRGraph—a computational graph representation that captures nodes, inputs, outputs, parameters, and constants. This TIRGraph is then processed by the TranspilerCodeGenerator to map TIR operations to Forge operations and generate executable Python Forge module code.

The system handles the complexity of model conversion through well-defined stages: model validation, shape inference, operation conversion using opset-aware converters, graph construction with proper topology and name sanitization, and finally code generation. Each stage is designed to be transparent and debuggable—the TIRGraph can be executed directly using PyTorch for validation, built-in debug mode compares outputs with ONNX Runtime, and the generated Python code is human-readable—while maintaining explicit opset-aware design that handles multiple ONNX opset versions through version-specific converter logic.

The transpiler is seamlessly integrated into the Forge compilation pipeline as an alternative path to TVM, allowing users to choose between the transpiler path or the TVM path, with both paths producing the same ForgeModule output that proceeds through Forge's graph passes, MLIR compilation, and binary generation. The system is built with extensibility in mind—new operations can be added by implementing converter classes, and the architecture supports future expansion to other frameworks beyond ONNX.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
   - [Component Responsibilities](#component-responsibilities)
   - [Shape Resolution: How Unknown Dimensions Are Resolved](#shape-resolution-how-unknown-dimensions-are-resolved)
3. [Transpiler Working - Detailed Walkthrough](#transpiler-working---detailed-walkthrough)
4. [Forge Compilation Pipeline](#forge-compilation-pipeline)
5. [Compiling Models: TVM vs Transpiler Path](#compiling-mnist-model-tvm-vs-transpiler-path)
6. [Testing](#testing)
   - [Operation Tests](#operation-tests)
   - [Model Tests: MNIST, ResNet-50, BERT, GPT-2](#model-tests)

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
├── frontends/onnx/               # ONNX-specific frontend
│   ├── engine.py                 # Main transpiler engine (ONNXToForgeTranspiler)
│   ├── converters/               # Operation converters (30+)
│   │   ├── activation.py         # Relu, Sigmoid, Tanh, Softmax, Dropout, etc.
│   │   ├── elementwise_binary.py # Add, Sub, Mul, Div, MatMul, Pow, Gemm
│   │   ├── elementwise_unary.py  # Abs, Neg, Sqrt, Erf, Log, etc.
│   │   ├── conv.py               # Conv1d/2d/3d
│   │   ├── pooling.py            # MaxPool, AvgPool, GlobalAvgPool
│   │   ├── reduction.py          # ReduceSum, ReduceMean, ReduceMax, ArgMax
│   │   ├── shape.py              # Reshape, Transpose, Flatten, Cast, Shape
│   │   ├── reshape.py            # Reshape
│   │   ├── concat.py             # Concat
│   │   ├── slice.py              # Slice
│   │   ├── split.py              # Split
│   │   ├── gather.py             # Gather, GatherElements
│   │   ├── expand.py             # Expand
│   │   ├── pad.py                # Pad
│   │   ├── squeeze.py            # Squeeze
│   │   ├── unsqueeze.py          # Unsqueeze
│   │   ├── clip.py               # Clip
│   │   ├── layernorm.py          # LayerNormalization
│   │   ├── condition.py          # Where
│   │   ├── constant.py           # Constant
│   │   ├── constantofshape.py    # ConstantOfShape
│   │   ├── trilu.py              # Trilu
│   │   ├── gemm.py               # Gemm
│   │   ├── base.py               # OnnxOpConverter base class
│   │   └── converter_result.py   # ConverterResult / ConstantResult types
│   ├── operations/               # ONNX-level shape metadata
│   │   └── op_shape_meta.py      # Shape evaluation metadata registry
│   ├── debug/                    # Debug utilities
│   │   └── validator.py          # ONNX Runtime comparison utilities
│   └── utils/                    # ONNX utilities
│       ├── naming.py             # Name sanitization and uniqueness enforcement
│       ├── attributes.py         # Attribute extraction and value parsing
│       ├── validation.py         # Constant input lookup & validation
│       ├── onnx_graph.py         # Graph manipulation helpers
│       ├── io_builder.py         # Input/output tensor extraction
│       ├── conversion_logger.py  # Structured conversion/execution logging
│       ├── constant_value_extractor.py  # Constant tensor value extraction
│       ├── shape_finder.py       # 4-step input shape resolution + 2-step output shape resolution
│       ├── subgraph_utils.py     # Subgraph traversal helpers
│       └── onnx_printer.py       # ONNX model pretty-printing
├── core/                         # Framework-agnostic core
│   ├── graph.py                  # TIRGraph implementation
│   ├── node.py                   # TIRNode base class
│   ├── types.py                  # TensorInfo and dtype utilities
│   └── shape_eval.py             # Shape evaluation metadata framework
├── operations/                   # TIR operation implementations
│   ├── arithmetic.py             # Add, Sub, Mul, Div, MatMul, Pow
│   ├── activation.py             # Relu, Sigmoid, Tanh, Softmax, etc.
│   ├── conv.py                   # Conv1d/2d/3d
│   ├── pooling.py                # MaxPool, AveragePool, GlobalAveragePool
│   ├── reduction.py              # ReduceSum, ReduceMean, ReduceMax, ArgMax
│   ├── shape.py                  # Reshape, Transpose, Squeeze, Unsqueeze, etc.
│   ├── shape_mixins.py           # Shape computation mixin helpers
│   ├── normalization.py          # LayerNorm
│   ├── indexing.py               # Gather, Slice, Split, Concat, Index
│   ├── comparison.py             # Equal, Greater, Less, etc.
│   └── other.py                  # Cast, Clip, Identity, Full, Trilu, etc.
├── utils/                        # Shared transpiler utilities
│   ├── exceptions.py             # Exception hierarchy (ConversionError, etc.)
│   ├── binary_ops.py             # Shape broadcasting utilities
│   └── graph_printer.py          # TIRGraph pretty-printing
└── codegen/                      # Code generation
    ├── transpiler_generator.py   # TranspilerCodeGenerator
    └── transpiler_to_forge.py    # Full pipeline: ONNX → ForgeModule
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
│  │  │  - Orchestrates the full ONNX -> TIRGraph conversion pipeline    │  │  │
│  │  │  - Model validation (ONNX schema checking)                       │  │  │
│  │  │  - Shape inference (fills missing tensor shapes)                 │  │  │
│  │  │  - Resolves symbolic/dynamic dims via ONNX Runtime once          │  │  │
│  │  │  - Opset extraction & converter map building                     │  │  │
│  │  │  - Parameter/constant distinction (heuristic-based)              │  │  │
│  │  │  - Name sanitization & uniqueness enforcement                    │  │  │
│  │  │  - Per Onnx node: resolves shapes, runs converter, updates graph │  │  │
│  │  │  - Folds constant ops into graph; no runtime node emitted        │  │  │
│  │  │  - Debug mode: caches ORT tensors for per-node comparison        │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Converters (OnnxOpConverter subclasses)                         │  │  │
│  │  │  - One converter class per ONNX op type (30+ converters)         │  │  │
│  │  │  - Opset-aware conversion via version-specific patterns          │  │  │
│  │  │  - Returns TIR nodes or constant results (constant folding)      │  │  │
│  │  │  - Attribute conversion from ONNX to PyTorch/Forge format        │  │  │
│  │  │  - Operation decomposition (Gemm->MatMul+Add, Trilu->Where)      │  │  │
│  │  │  - Input shape resolution before convert() (4-step strategy)     │  │  │
│  │  │  - Output shape resolution after convert() returns               │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Utils (utils/)                                                  │  │  │
│  │  │  - naming.py: Name sanitization & uniqueness enforcement         │  │  │
│  │  │  - attributes.py: Attribute extraction and value parsing         │  │  │
│  │  │  - validation.py: Constant input lookup & validation             │  │  │
│  │  │  - io_builder.py: Input/output TensorInfo dict building          │  │  │
│  │  │  - onnx_graph.py: Graph manipulation helpers                     │  │  │
│  │  │  - onnx_printer.py: Model pretty-print with shapes               │  │  │
│  │  │  - conversion_logger.py: Per-node structured debug logging       │  │  │
│  │  │  - subgraph_utils.py: Backward trace & subgraph execution        │  │  │
│  │  │  - constant_value_extractor.py: Compile-time const evaluation    │  │  │
│  │  │  - shape_finder.py: Unknown dim resolution (rules/fake exec)     │  │  │
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
│  │  │  - Computational graph (nodes in topological order)              │  │  │
│  │  │  - Tracks topology via producer and consumer mappings            │  │  │
│  │  │  - Stores params (trainable), constants, computed tensors        │  │  │
│  │  │  - Persists computed tensors to disk for runtime access          │  │  │
│  │  │  - Memory management via activation reference counting           │  │  │
│  │  │  - Direct execution with PyTorch for validation                  │  │  │
│  │  │  - Bidirectional name mapping (original <-> sanitized)           │  │  │
│  │  │  - Debug mode: compares outputs with ONNX Runtime                │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │  TIRNode (node.py)                                               │  │  │
│  │  │  - Base class for all TIR operations                             │  │  │
│  │  │  - Stores name, op_type, inputs, outputs, attrs                  │  │  │
│  │  │  - Executes operations for validation (PyTorch backend)          │  │  │
│  │  │  - Generates code metadata for Forge module generation           │  │  │
│  │  │  - Translates ONNX attributes to Forge format                    │  │  │
│  │  │  - Tracks Forge op name (e.g., "Conv2d", "Relu")                 │  │  │
│  │  │  - Records source ONNX node name for traceability                │  │  │
│  │  │  - Declares output shape dependency to guide the resolver        │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Shape Evaluation (shape_eval.py + op_shape_meta.py)             │  │  │
│  │  │  - Classifies each op by how its output shape is determined:     │  │  │
│  │  │      SHAPE_ONLY: follows from input shapes alone                 │  │  │
│  │  │      VALUE_OF_SHAPE_INPUT: depends on a shape-input value        │  │  │
│  │  │      VALUE_DEPENDENT: requires runtime data (exec skipped)       │  │  │
│  │  │  - Lets frontends register op metadata without coupling core     │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │  TensorInfo (types.py)                                           │  │  │
│  │  │  - Carries shape and dtype for a tensor through the pipeline     │  │  │
│  │  │  - Uses None for unknown dims so partial shapes can propagate    │  │  │
│  │  │  - Converts between ONNX element types and PyTorch dtypes        │  │  │
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
│  │  │  - All implement PyTorch-based execution for validation          │  │  │
│  │  │  - All generate code metadata for Forge module generation        │  │  │
│  │  │  - Arithmetic : AddNode, SubNode, MulNode, DivNode, MatMulNode   │  │  │
│  │  │  - Activation : ReluNode, SigmoidNode, TanhNode, SoftmaxNode     │  │  │
│  │  │  - Conv/Pool  : Conv1d/2d/3dNode, MaxPool/AvgPool1d/2d/3dNode    │  │  │
│  │  │  - Shape      : ReshapeNode, TransposeNode, SqueezeNode          │  │  │
│  │  │  - Reduction  : ReduceSumNode, ReduceMeanNode, ArgMaxNode        │  │  │
│  │  │  - Indexing   : GatherNode, SliceNode, ConcatNode, SplitNode     │  │  │
│  │  │  - Other      : LayerNormNode, CastNode, TriluNode, WhereNode    │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Shape Mixins (shape_mixins.py)                                  │  │  │
│  │  │  - Formula-based output shape computation per node type          │  │  │
│  │  │  - Compute output shape from input shapes and attributes         │  │  │
│  │  │  - No execution needed; reduces cost of shape resolution         │  │  │
│  │  │  - Cover elementwise, broadcast, matmul, conv, pooling,          │  │  │
│  │  │    reduction, reshape, transpose, and pad patterns               │  │  │
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
│  │  │  - Generates complete Python ForgeModule class from TIRGraph     │  │  │
│  │  │  - Writes import header and class definition                     │  │  │
│  │  │  - Registers params, constants, computed tensors as attributes   │  │  │
│  │  │  - forward() replays all ops in topological order                │  │  │
│  │  │  - Memory optimization: del after last use (ref counting)        │  │  │
│  │  │  - Generates weight-loading from ONNX initializers & .pt file   │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │  transpiler_to_forge (transpiler_to_forge.py)                    │  │  │
│  │  │  - Main entry point for ONNX -> ForgeModule conversion           │  │  │
│  │  │  - Builds TIRGraph via ONNXToForgeTranspiler                     │  │  │
│  │  │  - Generates Python source via TranspilerCodeGenerator           │  │  │
│  │  │  - Writes generated module to file system                        │  │  │
│  │  │  - Dynamically imports generated module at runtime               │  │  │
│  │  │  - Loads weights from ONNX initializers & computed tensors       │  │  │
│  │  │  - Verifies outputs against framework models when enabled        │  │  │
│  │  │  - Returns generated ForgeModule and sample inputs               │  │  │
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

- **Engine** (`engine.py`): The `ONNXToForgeTranspiler` class orchestrates the entire conversion process through a multi-stage pipeline. It validates the ONNX model structure and schema, runs shape inference to determine tensor shapes, optionally resolves dynamic (symbolic) dimensions by executing the model once via ONNX Runtime (`resolve_dynamic_shapes=True`), extracts the opset version from model metadata, builds opset-specific converter maps, processes initializers (distinguishing trainable parameters from constants), converts each ONNX node using appropriate converters, and constructs the final TIRGraph with proper topology and name mappings. Supports debug mode where ONNX Runtime is run once in a single pass to cache all intermediate tensors, which are then compared against TIR outputs per node. Supports parameter freezing options.

- **Converters** (`converters/`): Each ONNX operation type has its own dedicated converter class (30+ total). A converter's job is to take an ONNX node and produce one or more TIR nodes. They handle differences between ONNX opset versions transparently. When an ONNX op has no direct equivalent in Forge (e.g. `Gemm`, `GlobalAveragePool`, `Trilu`), the converter decomposes it into a sequence of simpler supported ops. When the converter's output is a compile-time constant (e.g. a `Constant` node), it returns a constant result instead of a TIR node — the value is stored directly in the graph without generating any runtime computation. Supported op categories: arithmetic (`Add`, `Sub`, `Mul`, `Div`, `MatMul`, `Gemm`, `Pow`), activation (`Relu`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax`, `LeakyRelu`, `Dropout`, `Sqrt`, `Erf`), convolution, pooling, shape ops (`Reshape`, `Transpose`, `Squeeze`, `Unsqueeze`, `Flatten`), reduction (`ReduceSum`, `ReduceMean`, `ReduceMax`, `ArgMax`), indexing (`Gather`, `Slice`, `Split`, `Concat`, `Expand`), normalization (`LayerNormalization`), comparison and logical ops, constants (`Constant`, `ConstantOfShape`), and utilities (`Cast`, `Clip`, `Identity`, `Pad`, `Where`, `Trilu`, `Shape`).

- **Utils** (`utils/`): Helper modules each focused on one responsibility:
  - `naming.py`: Converts ONNX tensor names into valid Python identifiers, ensuring uniqueness across the generated module
  - `attributes.py`: Parses ONNX attribute formats (int, float, tensor, list) into PyTorch-friendly Python values
  - `validation.py`: Checks whether a converter input is a known compile-time constant by searching through params, constants, and computed constants in the TIR graph; raises a distinct error type so the error origin is always unambiguous
  - `constant_value_extractor.py`: Evaluates compile-time constant values by tracing backward through the TIR graph and executing the constant subgraph; used by converters like `Gather` and `Reshape` that need the actual tensor value at conversion time
  - `subgraph_utils.py`: Shared graph-tracing utilities — backward BFS trace to classify ancestor tensors as constants or runtime inputs, minimal subgraph construction and execution, topological sorting, and node copying; used by both the shape resolver and the constant value extractor
  - `shape_finder.py`: Resolves unknown tensor dimensions using a four-step strategy before each converter runs, and a two-step strategy after it returns (see Shape Resolution section below)
  - `conversion_logger.py`: Structured per-node logging during conversion and TIR graph execution, including node headers, input/output shape summaries, and debug comparison blocks
  - `debug/validator.py`: Runs ONNX Runtime once in a single pass to cache all intermediate tensors, then compares them against TIR node outputs for numerical validation

#### Shape Resolution: How Unknown Dimensions Are Resolved

ONNX models — especially those exported with dynamic axes — frequently contain symbolic or `None` dimensions. The transpiler resolves all unknown dimensions to concrete integers before generating TIR nodes. Resolution happens at **two points** surrounding every converter call.

**Point 1 — Before the converter: resolve input tensor shapes**

Before each op-specific converter runs, the resolver inspects the incoming input shapes and, for any tensor whose shape contains an unknown dimension, runs the following four-step pipeline:

```
Step 0 — Graph-proto recovery
    Some ONNX shape inference passes convert zero-sized dimensions (0) into
    unknown (None). This step reads the original model declarations to recover
    those concrete zero values before attempting the backward trace.

Step 1 — Shape rules (no tensor execution)
    The resolver walks the TIR graph backward from the unknown tensor to
    collect all ancestor nodes. For each ancestor (in topological order), it
    applies a pure Python shape formula based on the operation family.
    Shape maps are seeded from all known constants, params, and already-resolved
    tensors, then propagated forward through the trace.
    Fast and deterministic — no PyTorch execution involved.

Step 2 — Constant subgraph execution
    If all ancestors are constants or params (no runtime model inputs),
    a minimal subgraph is built from those constants and executed with PyTorch.
    The shape of the resulting tensor is the answer.
    Skipped when any ancestor is a genuine runtime model input.

Step 3 — Fake-input execution
    Same as Step 2 but runtime model inputs are replaced by deterministic dummy
    tensors, with symbolic dimensions replaced by 1. Skipped when any op in the
    trace has VALUE_DEPENDENT shape metadata, because fake input values would
    produce incorrect shapes for ops like Reshape (whose output shape depends on
    the actual values of its second input, not just its shape).
```

If all steps fail, an error is raised with a diagnostic showing the tensor name, current shape, trace path, and whether any runtime model input was involved.

**Point 2 — After the converter: resolve output tensor shapes**

Immediately after a converter returns its TIR nodes, with input shapes now fully concrete, the resolver checks whether any output shapes are still unknown (ONNX shape inference may have left them symbolic for dynamic-axis ops like `Where`, `Equal`, `MatMul`, or `Concat`). For each TIR node with unknown outputs:

```
Step 1 — Shape rules
    A pure Python shape formula is applied using the concrete input shapes and
    any constant/param values already in the TIRGraph. No tensor execution needed.

Step 2 — Fake-input subgraph execution
    A minimal subgraph is built with real constant tensors and synthetic
    activation tensors constructed from the known concrete shapes. The node is
    executed and the output shapes are read off the produced tensors.
    Skipped for VALUE_DEPENDENT nodes (e.g. Reshape).
```

Resolved shapes are written back into the TIR nodes and output tensors so the engine can propagate them to all downstream converter calls.

**How concrete shapes propagate across nodes**

After each ONNX node is converted, the engine records every resolved output shape. Subsequent converter calls therefore receive concrete shapes and almost never need to trigger the backward trace at all. The trace is only invoked when a downstream node still sees a symbolic dimension — either because ONNX shape inference could not infer it, or because the model uses truly dynamic axes.

**Shape dependency classification**

Each TIR node carries a tag that tells the resolver which resolution strategy is safe to use:

| Category | Meaning | Fake-exec safe? |
|---|---|---|
| `SHAPE_ONLY` | Output shape depends only on input shapes (e.g. Relu, Add, Transpose) | Yes |
| `VALUE_OF_SHAPE_INPUT` | Output shape depends on the *value* of one specific input tensor (e.g. Reshape, Unsqueeze) | Yes — that input is typically a constant |
| `VALUE_DEPENDENT` | Output shape depends on runtime input values — conservative default | No |

All 30+ supported op types are registered with their category at startup. Unregistered ops default to `VALUE_DEPENDENT` (safest assumption), ensuring the resolver never uses fake inputs where they would produce an incorrect shape.

#### Core Layer (`core/`)

The core layer provides framework-agnostic abstractions that work across all frontends:

- **TIRGraph** (`graph.py`): Represents the computational graph in a framework-agnostic way. It maintains nodes in execution order, topology maps (`producer_map` and `consumer_map`) tracking tensor dependencies, three distinct tensor stores — `params` (trainable weights from ONNX initializers), `constants` (non-trainable values from ONNX initializers), and `computed_constants` (tensors produced by ONNX ops such as `Constant`, `ConstantOfShape`, and auxiliary scalars created during conversion that are NOT in the ONNX initializer list) — and bidirectional name mappings between original frontend names and sanitized names. The graph can be executed directly using PyTorch via the `run()` method (inputs are optional for subgraphs whose all inputs are already constants), supports topological sorting using Kahn's algorithm, computes activation dependencies for memory management, and includes an optimized debug mode that runs ONNX Runtime once (single-pass) to cache all intermediate tensors before per-node comparison. A `log_execution` flag suppresses verbose logs for internal utility sub-graphs. The `initializers` property combines `params + constants + computed_constants` for uniform access when all tensor stores are needed together.

- **Exceptions** (`utils/exceptions.py`): Defines the exception hierarchy used throughout the transpiler — a base error type and distinct subtypes for op conversion failures, structural validation failures, debug comparison failures, unsupported operations, and ONNX model validation failures. Lives in `utils/` (shared across all layers) and is re-exported from `core/__init__.py` for convenient import.

- **TIRNode** (`node.py`): Base class for every operation in the TIRGraph. It stores the operation's name, type, inputs, outputs, and attributes. It can execute itself with PyTorch for validation and emit the metadata needed to generate a Forge API call in the output Python file. Attribute conversion from PyTorch format to Forge's format is done lazily via a property on first access, ensuring any subclass override fires only after the subclass is fully constructed. Each node also carries a shape dependency tag (populated from the registry at construction) that the shape resolver uses to decide which resolution strategies are safe to apply.

- **Shape Evaluation Metadata** (`core/shape_eval.py`): Defines the three shape dependency categories (`SHAPE_ONLY`, `VALUE_OF_SHAPE_INPUT`, `VALUE_DEPENDENT`) and a pluggable registry that lets frontends register per-op-type mappings. The core layer has no knowledge of ONNX — the ONNX frontend registers its own mappings at startup. Unregistered ops default to `VALUE_DEPENDENT` (safest conservative assumption).

- **ONNX Op Shape Metadata** (`frontends/onnx/operations/op_shape_meta.py`): Registers the shape dependency category for all 30+ ONNX op types supported by the frontend. The shape resolver reads these registrations to select the cheapest resolution strategy that is safe for each op.

- **Types** (`types.py`): A lightweight container that carries a tensor's shape and data type through the pipeline, with support for partially-known shapes containing `None` dimensions, and utilities for converting between ONNX and PyTorch data type representations.

#### Operations Layer (`operations/`)

Every supported operation is implemented as a concrete `TIRNode` subclass with a PyTorch implementation for validation and code-generation metadata for the output file. Operations are grouped by family:

- **Arithmetic** — Add, Sub, Mul, Div, MatMul, Pow
- **Convolution** — Conv1d, Conv2d, Conv3d (selected automatically based on input rank)
- **Activation** — Relu, Sigmoid, Tanh, Softmax, LogSoftmax, LeakyRelu, Dropout, Erf, Sqrt
- **Pooling** — MaxPool and AveragePool for 1d/2d/3d, GlobalAveragePool
- **Shape** — Reshape, Transpose, Squeeze, Unsqueeze
- **Reduction** — ReduceSum, ReduceMean, ReduceMax, ArgMax
- **Indexing** — Gather, Slice, Split, Concat, Index
- **Normalization** — LayerNorm
- **Comparison** — Equal, Greater, Less, GreaterOrEqual, LessOrEqual
- **Logical** — LogicalAnd, LogicalNot
- **Other** — Cast, Clip, Identity, Full (constant tensor), Trilu, Where

**Shape Mixins** (`operations/shape_mixins.py`): Each operation class also inherits a shape mixin that provides a pure-Python formula for computing the output shape without executing the operation. This allows the shape resolver to determine output shapes cheaply before (or instead of) running PyTorch. Mixins cover all operation families — elementwise, broadcast, matmul, convolution, pooling, reduction, reshape, transpose, pad, concat, and others — so the resolver always has a formula-based path available as its first and cheapest option.

#### Codegen Layer (`codegen/`)

The codegen layer transforms a `TIRGraph` into a runnable Python file that implements a `ForgeModule`:

- **TranspilerCodeGenerator** (`transpiler_generator.py`): Generates the complete `ForgeModule` Python class from the TIRGraph. The generated class `__init__` registers all trainable parameters, ONNX-initializer constants, and computed constants (tensors that were produced during transpilation, such as `ConstantOfShape` outputs) as module attributes with proper device-aware data format handling. The `forward` method replays every operation in topological order as Forge API calls. The generator also applies a reference-counting algorithm so that intermediate activations are deleted as soon as they are no longer needed, keeping peak memory usage low. A parameter-loading method is generated that loads weights from the ONNX file and, if a companion `.pt` file was saved for computed constants, loads those too.

- **transpiler_to_forge** (`transpiler_to_forge.py`): The single entry point that ties the whole pipeline together. It calls the engine to build the TIRGraph (passing real sample inputs so dynamic shapes can be resolved), saves any computed constants to a `.pt` file on disk, calls the code generator to produce the Python source, writes that source to a file, imports it at runtime, instantiates the `ForgeModule`, loads all weights, and optionally runs a numerical comparison against the original ONNX model to verify correctness. Returns the generated module and sample inputs to the Forge compilation pipeline.

---

## Transpiler Working - Detailed Walkthrough

The transpiler converts an ONNX `ModelProto` into a runnable Python `ForgeModule` through five sequential stages. All five stages are orchestrated by `transpiler_to_forge.py`; stages 1–4 run inside `ONNXToForgeTranspiler.transpile()`, and stage 5 runs in `transpiler_to_forge()`.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ONNX Model                                     │
│  (ModelProto: nodes, initializers, inputs, outputs, opset metadata)         │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 1 — Validate & Prepare                                               │
│  • Validate model against ONNX spec (optional, raises on failure)           │
│  • Extract opset version and build the op-type → converter map              │
│  • Run ONNX shape inference to fill in missing tensor shapes                │
│  • Remove initializers from the graph input list                            │
│  • Optionally run model once via ONNX Runtime to resolve dynamic dims       │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 2 — Process Initializers                                             │
│  • Classify each initializer as param (trainable) or constant               │
│    using name/shape/dtype heuristics, or freeze_params flag                 │
│  • Convert ONNX tensors to PyTorch and store in TIRGraph                    │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 3 — Convert ONNX Nodes → TIR Nodes                                  │
│  Pre-scan: reject all unsupported op types upfront (fail-fast)              │
│  For each node (in topological order):                                      │
│    1. Build TensorInfo dicts for inputs and outputs                         │
│    2. Extract and normalize ONNX attributes                                 │
│    3. Resolve unknown input shapes (4-step strategy)                        │
│    4. Call the opset-bound converter                                        │
│    5. Resolve unknown output shapes (2-step strategy)                       │
│    6. Store: TIR nodes → graph, ConstantResult → computed_constants         │
│    7. Propagate concrete output shapes to value_info_map                    │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 4 — Finalize TIRGraph                                                │
│  • Confirm output name → sanitized name mappings                            │
│  • Compute activation dependency map (drives memory management)             │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TIRGraph (complete)                                      │
│  • TIR nodes in topological order                                           │
│  • params, constants, computed_constants                                    │
│  • producer_map / consumer_map topology                                     │
│  • Bidirectional original ↔ sanitized name maps                             │
│  • Activation dependency map                                                │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 5 — Generate & Instantiate ForgeModule                               │
│  • Save computed_constants to {model}_constants.pt (before codegen)         │
│  • Generate ForgeModule Python class from TIRGraph                          │
│    - __init__: registers params, constants, computed constants              │
│    - forward(): ops in topological order as Forge API calls                 │
│    - Memory optimization: del after last use (reference counting)           │
│    - process_framework_parameters(): loads weights + computed constants     │
│  • Write source to generated_modules/{model_name}.py                       │
│  • Import module at runtime via importlib                                   │
│  • Instantiate ForgeModule and load all weights (2-phase loading)           │
│  • Optionally verify outputs against original ONNX model                   │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ForgeModule (ready for Forge compilation)                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1 — Validate & Prepare

The goal of this stage is to give every subsequent stage a clean, fully-shaped model with no ambiguity about opset version, tensor shapes, or graph structure.

**Validation** (optional, `validate_model=True`): `onnx.checker.check_model()` verifies the model conforms to the ONNX specification. Structural problems, schema violations, missing required fields, and type mismatches are caught here before any conversion work begins. Failure raises `ONNXModelValidationError` immediately.

**Opset extraction**: The primary opset version is read from the model's `opset_import` list. This number determines which conversion logic applies to every op. If absent, the transpiler defaults to opset 1. Different opset versions can move attributes to tensor inputs, rename fields, or change operation semantics, so this must be determined first.

**Converter map building**: A dictionary from ONNX op-type string to converter function is built. Each converter is retrieved via `get_converter(opset)`, which binds the opset version at lookup time. The opset does not need to be passed again at call time.

**Shape inference**: `onnx.shape_inference.infer_shapes()` propagates shapes and dtypes from model inputs forward through all nodes, filling in whatever can be determined statically. Remaining dynamic dimensions (symbolic strings like `"__unk__"`, or `None`) stay as-is and are resolved later by the inline shape resolver. Failure is fatal — the transpiler cannot proceed without shape information.

**Remove initializers from inputs**: ONNX models sometimes list weight tensors in the graph input list. This step removes them, leaving only the true runtime inputs that the caller must supply.

**Dynamic shape resolution** (optional, `resolve_dynamic_shapes=True`): When sample inputs are provided, the model is run once via ONNX Runtime with those concrete inputs. The actual integer value of every symbolic or unknown dimension (across all inputs, outputs, and intermediate tensors) is collected and written back into the model proto in-place. The patched model is used for the rest of the conversion so every converter receives fully concrete shapes without triggering backward traces. This is essential for models such as GPT-2 and BERT that are exported with dynamic sequence dimensions.

---

### Stage 2 — Process Initializers

ONNX initializers are the pre-computed tensors stored in the model file — weights, biases, embedding tables, positional encodings, and similar values. Each is converted from ONNX format to a PyTorch tensor and classified as either a **parameter** (trainable) or a **constant** (non-trainable).

**Classification heuristics** (when `freeze_params=False`):
- Classified as a constant if the name contains "constant" (case-insensitive), the tensor is a scalar, or it has an integer/boolean dtype and is not a weight or bias
- Everything else is classified as a parameter

When `freeze_params=True`, all initializers are treated as constants regardless of name or dtype.

Parameters go into `tir_graph.params`; constants go into `tir_graph.constants`. The code generator handles them differently: parameters are registered as trainable module attributes, constants as fixed buffers.

---

### Stage 3 — Convert ONNX Nodes → TIR Nodes

This is the core of the transpiler. Every ONNX node is converted into one or more TIR nodes, or into a constant value if its output can be determined at compile time.

**Pre-scan**: Before the conversion loop starts, all nodes are scanned against the converter map. Any unsupported op types are collected and reported together in a single `UnsupportedOperationError`. This prevents partial conversion failures midway through large models.

**Per-node conversion loop** (nodes processed in ONNX graph order, which is topological):

1. **Build TensorInfo dicts** — For each node input, the shape and dtype are looked up from `value_info_map` (populated by shape inference and updated after each previous node). If not found there, the value is taken directly from params or constants. For each node output, shape/dtype come from the shape inference results.

2. **Extract attributes** — ONNX attributes are parsed from the node proto and converted to Python values. Attribute names are normalised to PyTorch conventions (e.g. `dilations` → `dilation`). Defaults are applied for any missing optional attribute.

3. **Resolve unknown input shapes** — Before the converter is called, any input tensor whose shape still contains a `None`, a symbolic string, or a negative integer is resolved by the inline shape resolver using a four-step strategy (see [Shape Resolution](#shape-resolution-how-unknown-dimensions-are-resolved)). Converters always receive fully concrete input shapes.

4. **Call the converter** — The opset-bound converter function is called with the node proto, TensorInfo dicts, attributes, the graph proto, and the TIRGraph. The converter may decompose the ONNX op into multiple TIR nodes (e.g. `Gemm` → `MatMul` + `Add`, `GlobalAveragePool` → chain of `ReduceMean`, `Trilu` → mask + `Where`).

5. **Resolve unknown output shapes** — After the converter returns, any output tensor whose shape is still unknown is resolved using a two-step strategy (shape rules, then fake-input execution). `VALUE_DEPENDENT` nodes (e.g. `Reshape`) skip fake execution.

6. **Store the result**:
   - *TIR nodes*: Each node gets a sanitized output name (e.g. `conv2d_0`, `relu_1`) generated from the op type and a counter. Bidirectional name mappings are updated. Each node is added to the TIRGraph, which automatically updates the `producer_map` and `consumer_map`.
   - *ConstantResult*: The tensor value is stored in `tir_graph.computed_constants` (distinct from `tir_graph.constants` which holds ONNX initializers). Scalar 0-d tensors are reshaped to 1-d before storage. No TIR node is created.

7. **Propagate shapes** — Resolved output shapes are written back into `value_info_map` so the next node in the loop reads concrete shapes directly, without triggering another backward trace.

**Opset differences handled per-converter**:
- `Reshape`: shape as attribute (old) vs. input tensor (opset ≥ 5)
- `Squeeze`/`Unsqueeze`: axes as attribute (old) vs. input tensor (opset ≥ 13)
- `ReduceSum`: axes as attribute (old) vs. optional input tensor (opset ≥ 18)

---

### Stage 4 — Finalize TIRGraph

The TIRGraph has been built incrementally throughout stages 1–3. It was created at the start of `transpile()`, input names were sanitized before the conversion loop, and topology maps were updated node-by-node during stage 3. Two finalization steps complete it:

**Output name resolution**: The final list of graph output names is confirmed. Their sanitized identifiers were pre-registered before the conversion loop, but this step sets the definitive `tir_graph.outputs` list, handling edge cases where an output comes directly from a constant or parameter.

**Activation dependency computation**: For each TIR node, the graph records which of its outputs are consumed by later nodes. This map tells the code generator exactly when each intermediate activation can be safely deleted after its last use, keeping peak memory low. The same map is used during direct PyTorch execution of the graph for garbage collection between nodes.

---

### Stage 5 — Generate & Instantiate ForgeModule

This stage (in `transpiler_to_forge.py`) takes the complete TIRGraph and produces a running `ForgeModule` instance with all weights loaded.

**Save computed constants** (before code generation): Any tensors in `tir_graph.computed_constants` are de-duplicated by sanitized name and saved to `generated_modules/{module_name}_constants.pt`. The file path is baked into the generated `process_framework_parameters()` so constants are loaded automatically at runtime.

**Code generation**: The `TranspilerCodeGenerator` produces a complete Python `ForgeModule` class:
- `__init__` registers all params, constants, and computed constants as module attributes
- `forward()` replays every TIR node in topological order as Forge API calls
- A reference-counting pass inserts `del` statements for intermediate activations immediately after their last use
- `process_framework_parameters()` loads ONNX initializer weights (phase 1) and, if present, computed constants from the `.pt` file (phase 2)

**Write & import**: The source is written to `generated_modules/{module_name}.py` and imported at runtime via `importlib`. No separate process or restart is needed.

**Instantiate & load weights**: The `ForgeModule` class is instantiated. Weights are then loaded in two phases: ONNX initializers first (mapped from original to sanitized names), then computed constants from the `.pt` file.

**Verify** (optional): The module is run with the sample inputs and its outputs are compared against the original ONNX model within numerical tolerance.

**Return**: The fully initialised `ForgeModule` and the converted sample inputs are returned to the Forge compilation pipeline for graph optimisation, MLIR lowering, and binary code generation.

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

## Compiling Models: TVM vs Transpiler Path

This section shows how to compile models using both TVM and transpiler paths, using MNIST as the canonical example. The system now supports MNIST, ResNet-50, BERT, and GPT-2. The MNIST test (`test_mnist_onnx.py`) uses pytest parametrize to test both compilation paths in a single test file, allowing easy comparison between TVM and transpiler paths.

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
- `transpiler_resolve_dynamic_shapes=True`: Runs the model once via ONNX Runtime with actual inputs to resolve symbolic/dynamic dimensions (e.g., sequence length in BERT/GPT-2) before conversion. Requires ONNX Runtime and actual inputs to be passed to `forge.compile()`.

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
| **Transpiler** | `compile_transpiler_to_python=True`<br>`compile_tvm_to_python=False`<br>`transpiler_enable_debug=True`<br>`transpiler_resolve_dynamic_shapes=True` | Direct ONNX → TIRGraph → ForgeModule | • ONNX models only<br>• Need faster compilation<br>• Want transparent conversion<br>• Models with dynamic shapes (BERT, GPT-2) |
| **TVM** | Default (no config needed)<br>`compile_tvm_to_python=True` (default) | ONNX → TVM Relay IR → ForgeModule | • Multiple frameworks (PyTorch, TensorFlow, etc.)<br>• Need advanced optimizations<br>• Model has unsupported operations |

---

## Testing

### Operation Tests (`forge/test/transpiler/ops/`)

Each ONNX operation has a dedicated test file. Tests compare TIRGraph outputs against ONNX Runtime outputs across opset versions, input shapes, dtypes, and edge cases.

| Test File | Operations Covered |
|-----------|-------------------|
| `test_add.py` | Add |
| `test_arithmetic.py` | Sub, Mul, Div |
| `test_matmul.py` | MatMul |
| `test_pow.py` | Pow |
| `test_erf.py` | Erf |
| `test_sqrt.py` | Sqrt |
| `test_comparison.py` | Equal, Greater, Less, GreaterOrEqual, LessOrEqual |
| `test_logical.py` | LogicalAnd, LogicalNot |
| `test_reduction.py` | ReduceSum, ReduceMean, ReduceMax |
| `test_argmax.py` | ArgMax |
| `test_globalavgpool.py` | GlobalAveragePool |
| `test_reshape.py` | Reshape |
| `test_concat.py` | Concat |
| `test_slice.py` | Slice |
| `test_split.py` | Split |
| `test_gather.py` | Gather, GatherElements |
| `test_expand.py` | Expand |
| `test_unsqueeze.py` | Unsqueeze |
| `test_shape.py` | Shape |
| `test_constantofshape.py` | ConstantOfShape |
| `test_layernorm.py` | LayerNormalization |
| `test_trilu.py` | Trilu |
| `test_where.py` | Where |

### Model Tests (`forge/test/transpiler/models/`, `forge/test/models/onnx/`)

End-to-end tests covering the full pipeline: PyTorch → ONNX → TIRGraph → ForgeModule, including code generation, computed constant persistence, and numerical output comparison. Shared helpers in `model_test_utils.py` provide consistent output comparison across all tests.

```bash
# MNIST
pytest forge/test/models/onnx/vision/mnist/test_mnist_onnx.py
pytest forge/test/transpiler/models/test_mnist.py

# ResNet-50
pytest forge/test/models/onnx/vision/resnet/test_resnet.py
pytest forge/test/transpiler/models/test_resnet50.py

# BERT
pytest forge/test/models/onnx/text/bert/test_bert.py
pytest forge/test/transpiler/models/test_bert.py

# GPT-2
pytest forge/test/models/onnx/text/gpt2/test_gpt2_onnx.py
pytest forge/test/transpiler/models/test_gpt2.py
```
