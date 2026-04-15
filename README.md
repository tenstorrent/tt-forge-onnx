[![Tests][tests badge]][tests]
[![Codecov][codecov badge]][codecov]
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tenstorrent/tt-forge-onnx)

<div align="center">

<h1>

[Hardware](https://tenstorrent.com/cards/) | [Documentation](https://docs.tenstorrent.com/tt-forge-onnx/) | [Discord](https://discord.gg/tenstorrent) | [Join Us](https://boards.greenhouse.io/tenstorrent?gh_src=22e462047us) | [Bounty $](https://github.com/tenstorrent/tt-forge-onnx/issues?q=is%3Aissue%20state%3Aopen%20label%3Abounty)

</h1>

<img src="./docs/src/imgs/tt_refresh_forge-onnx_w_logo_purple.png" alt="forge onnx" height="230"/>

<br>

**TT-Forge-ONNX** is a graph compiler for running **ONNX**, **TensorFlow**, and **PaddlePaddle** models on Tenstorrent hardware, optimizing computational graphs for performance and efficiency.

> **Part of the [TT-Forge](https://github.com/tenstorrent/tt-forge) AI compiler ecosystem.**

</div>

<br>

-----
# Run a Model

Install TT-Forge-ONNX and run an ONNX model on Tenstorrent hardware:

```bash
# Install uv if you don't have it yet
curl -LsSf https://astral.sh/uv/install.sh | sh

uv pip install tt_forge_onnx --extra-index-url https://pypi.eng.aws.tenstorrent.com/
uv pip install tt_tvm --extra-index-url https://pypi.eng.aws.tenstorrent.com/
```

```python
import torch, onnx, forge

# Load any ONNX model
onnx_model = onnx.load("resnet50.onnx")
input_tensor = torch.randn(1, 3, 224, 224)

# Compile and run on Tenstorrent
compiled_model = forge.compile(onnx_model, [input_tensor])
output = compiled_model(input_tensor)

predicted_class = output[0].argmax(dim=-1).item()
print(f"Predicted ImageNet class: {predicted_class}")
```

Any `.onnx` file works — export from PyTorch, TensorFlow, PaddlePaddle, or grab one from the [ONNX Model Zoo](https://github.com/onnx/models). See the full [Getting Started Guide](docs/src/getting_started.md) for Docker and build-from-source options.

-----
# Quick Links
- [Getting Started / How to Run a Model](docs/src/getting_started.md)
- [Build from Source](docs/src/getting_started_build_from_source.md) — For development work
- [Demos](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-forge-onnx) — Ready-to-run models
- [Supported Operations](https://docs.tenstorrent.com/tt-forge-onnx/operations.html) — 70+ supported ops

-----
# What is this Repo?

TT-Forge-ONNX is a TVM-based frontend within the TT-Forge ecosystem. It compiles models from ONNX, TensorFlow, and PaddlePaddle for Tenstorrent hardware (Wormhole, Blackhole). It also supports PyTorch, though [TT-XLA](https://github.com/tenstorrent/tt-xla) is recommended for PyTorch and JAX models. TT-Forge-ONNX is for **single-chip configurations only**.

| Frontend | Use For | Chip Support |
|----------|---------|-------------|
| **[TT-XLA](https://github.com/tenstorrent/tt-xla)** | PyTorch, JAX | Single & Multi-chip |
| **TT-Forge-ONNX** (this repo) | ONNX, TensorFlow, PaddlePaddle | Single-chip |

-----
# Related Tenstorrent Projects
- [TT-Forge](https://github.com/tenstorrent/tt-forge) — Central hub for the TT-Forge compiler project (demos, benchmarks, releases)
- [TT-XLA](https://github.com/tenstorrent/tt-xla) — Primary frontend for PyTorch and JAX (single and multi-chip)
- [TT-MLIR](https://github.com/tenstorrent/tt-mlir) — Core MLIR-based compiler framework for Tenstorrent hardware
- [TT-Metal](https://github.com/tenstorrent/tt-metal) — Low-level programming model and kernel development for Tenstorrent hardware


# Tenstorrent Bounty Program Terms and Conditions
This repo is a part of Tenstorrent’s bounty program. If you are interested in helping to improve tt-forge-onnx, please make sure to read the [Tenstorrent Bounty Program Terms and Conditions](https://docs.tenstorrent.com/bounty_terms.html) before heading to the issues tab. Look for the issues that are tagged with both “bounty” and difficulty level!
- - -

[codecov]: https://codecov.io/gh/tenstorrent/tt-forge-onnx
[tests]: https://github.com/tenstorrent/tt-forge-onnx/actions/workflows/on-pr.yml?query=branch%3Amain
[deepwiki]: https://deepwiki.com/tenstorrent/tt-forge-onnx
[codecov badge]: https://codecov.io/gh/tenstorrent/tt-forge-onnx/graph/badge.svg
[tests badge]: https://github.com/tenstorrent/tt-forge-onnx/actions/workflows/on-pr.yml/badge.svg?query=branch%3Amain
