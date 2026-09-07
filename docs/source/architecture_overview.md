# Architecture Overview

TT-Forge is a comprehensive compiler designed to facilitate the development and optimization of machine learning models. It encompasses various components, each serving a specific purpose in the compiling and running machine learning pipelines. This document provides an overview of the key components with focus on [TT-Forge-ONNX](https://github.com/tenstorrent/tt-forge-onnx).

## Table of contents
- [TT-Forge Overview](#tt-forge-overview)
- [TT-TVM Overview](#tt-tvm-overview)
   - [TVM IR](#tvm-ir)
   - [TVM Compile](#tvm-compile)
     - [Relay Compile Passes](#relay-compile-passes)
     - [Forge Compile Passes](#forge-compile-passes)
   - [Partition Graph](#partition-graph)
   - [Construct Inputs, Constants and Ops](#construct-inputs-constants-and-ops)
   - [Generate Forge-ONNX Module](#generate-forge-onnx-module)
   - [Standalone Forge-ONNX Module](#standalone-forge-onnx-module)
- [TT-Forge-ONNX Overview](#tt-forge-onnx-overview)
   - [Initialize Compile](#initialize-compile)
   - [Generate Initial Graph (TT-TVM)](#generate-initial-graph-tt-tvm)
   - [Post Initial Graph passes](#post-initial-graph-passes)
   - [Consteval](#consteval)
   - [Autograd](#autograd)
   - [Post Autograd](#post-autograd)
   - [Pre Lowering](#pre-lowering)
   - [Graph Split](#graph-split)
   - [Compiler TTIR](#compiler-ttir)
   - [Output Binary](#output-binary)

## TT-Forge Overview

![TT-Forge Overview](imgs/compiler_arch/tt-forge_overview.svg "TT-Forge Overview")

## TT-TVM Overview

![TT-TVM Overview](imgs/compiler_arch/tt-tvm_overview.svg "TT-TVM Overview")

### TVM IR

Coming soon!

### TVM Compile

Coming soon!

#### Relay Compile Passes

Coming soon!

#### Forge Compile Passes

Coming soon!

### Partition Graph

Coming soon!

### Construct Inputs, Constants and Ops

Coming soon!

### Generate Forge-ONNX Module

Coming soon!

### Standalone Forge-ONNX Module

Coming soon!

## TT-Forge-ONNX Overview

![TT-Forge-ONNX Overview](imgs/compiler_arch/forge-onnx_overview.drawio.svg "TT-Forge-ONNX Overview")

The diagram traces a single ONNX `Add` from the frontend to the FPU, pass by pass, and
marks where a Quasar target diverges from Wormhole. It is generated — run
`python scripts/gen_pipeline_diagram.py` to rebuild it after a tt-mlir pin bump; the
tt-mlir pass list is read out of the real pipeline rather than transcribed. The file is
also a diagrams.net document, so it opens editable at
[app.diagrams.net](https://app.diagrams.net).

For a runnable narration of the same path, with the flatbuffer inspected at each stage,
see `python scripts/add_op_walkthrough.py`. For what the Quasar annotations mean, see
[Quasar](dev_notes/quasar.md).


### Initialize Compile

Coming soon!

### Generate Initial Graph (TT-TVM)

Coming soon!

### Post Initial Graph passes

Coming soon!

### Consteval

Coming soon!

### Autograd

Coming soon!

### Post Autograd

Coming soon!

### Pre Lowering

Coming soon!

### Graph Split

Coming soon!

### Compiler TTIR

Coming soon!

### Output Binary

Coming soon!
