# Introduction

The **TT-Forge-ONNX** is a graph compiler designed to optimize and transform computational graphs for deep learning models on single-chip systems, enhancing their performance and efficiency.

Built on top of the [TT-MLIR](https://docs.tenstorrent.com/tt-mlir/) backend, **TT-Forge-ONNX** is an integral component of the [TT-Forge](https://docs.tenstorrent.com/tt-forge/) project, which provides a comprehensive suite of tools for optimizing and deploying deep learning models on [Tenstorrent](https://tenstorrent.com/) hardware.

The main project goals are:
- Provide abstraction of ONNX, TensorFlow, and PaddlePaddle frontend frameworks
- Compile many kinds of model architectures without custom modification and with great performance (for example, Transformers, CNNs, etc.)
- Abstract all Tenstorrent device architectures (for example, Wormhole, Blackhole, etc.)
