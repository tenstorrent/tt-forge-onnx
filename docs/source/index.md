# TT-Forge-ONNX Documentation

TT-Forge-ONNX is a graph compiler designed to optimize and transform
computational graphs for deep learning models on Tenstorrent AI hardware. It is
an integral component of the [TT-Forge](https://docs.tenstorrent.com/tt-forge/)
project, built on top of the [TT-MLIR](https://docs.tenstorrent.com/tt-mlir/)
backend.

```{toctree}
:caption: Introduction
:maxdepth: 2

introduction
getting_started
getting_started_docker
getting_started_build_from_source
architecture_overview
```

```{toctree}
:caption: Project Setup
:maxdepth: 2

test
pytest
perf
tools
```

```{toctree}
:caption: Operations Reference
:maxdepth: 2

operations
```

```{toctree}
:caption: Dev Notes
:maxdepth: 2

dev_notes/standalone_ttir_run
dev_notes/verification
docstring_standard
```
