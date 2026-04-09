# Getting Started
This document walks you through how to set up TT-Forge-ONNX. This is the main Getting Started page. There are two additional Getting Started pages depending on what you want to do. They are all described here, with links provided to each.

> **NOTE:** TT-Forge-ONNX is a framework agnostic frontend  that can convert any model to a generic
> Intermediate Representation (IR) that can then be converted to a Tenstorrent specific IR for use with
> Tenstorrent hardware. TT-Forge-ONNX is for use with single-chip systems only.

The following topics are covered:

* [Setup Options](#setup-options)
* [Configuring Hardware](#configuring-hardware)
* [Installing a Wheel and Running an Example](#installing-a-wheel-and-running-an-example)
* [Other Setup Options](#other-set-up-options)
    * [Using a Docker Container to Run an Example](getting_started_docker.md)
    * [Building From Source](getting_started_build_from_source.md)
* [Where to Go Next](#where-to-go-next)

> **NOTE:** If you encounter issues, please request assistance on the
>[TT-Forge-ONNX Issues](https://github.com/tenstorrent/tt-forge-onnx/issues) page.

## Setup Options
TT-Forge-ONNX can be used to run models from any framework. Because TT-Forge-ONNX is open source, you can also develop and add features to it. Setup instructions differ based on the task. You have the following options, listed in order of difficulty:
* [Installing a Wheel and Running an Example](#installing-a-wheel-and-running-an-example) - You should choose this option if you want to run models.
* [Using a Docker Container to Run an Example](getting_started_docker.md) - Choose this option if you want to keep the environment for running models separate from your existing environment.
* [Building from Source](getting_started_build_from_source.md) - This option is best if you want to develop TT-Forge-ONNX further. It's a more complex process you are unlikely to need if you want to stick with running a model.

## Configuring Hardware
Before setup can happen, you must configure your hardware. You can skip this section if you already completed the configuration steps. Otherwise, this section of the walkthrough shows you how to do a quick setup using TT-Installer.

1. Configure your hardware with TT-Installer using the [Quick Installation section here.](https://docs.tenstorrent.com/getting-started/README.html#quick-installation)

2. Reboot your machine.

3. Make sure **hugepages** is enabled:

```bash
sudo systemctl enable --now 'dev-hugepages\x2d1G.mount'
sudo systemctl enable --now tenstorrent-hugepages.service
```

4. Please ensure that after you run the TT-Installer script, after you complete reboot and set up hugepages, you activate the virtual environment it sets up - ```source ~/.tenstorrent-venv/bin/activate```.

5. After your environment is running, to check that everything is configured, type the following:

```bash
tt-smi
```

You should see the Tenstorrent System Management Interface. It allows you to view real-time stats, diagnostics, and health info about your Tenstorrent device.

![TT-SMI](./imgs/tt_smi.png)

## Installing a Wheel and Running an Example

This section walks you through downloading and installing a wheel. You can install the wheel wherever you would like.

1. Make sure you have Python 3.12 installed and you are in an active Python 3.12 virtual environment. This walkthrough uses the same environment you activated to look at TT-SMI in the [Configuring Hardware](#configuring-hardware) section.

2. For this walkthrough, TT-Forge-ONNX is used. You need to install the **tt_forge_onnx** and **tt_tvm** wheels:

```bash
pip install tt_forge_onnx --extra-index-url https://pypi.eng.aws.tenstorrent.com/
pip install tt_tvm --extra-index-url https://pypi.eng.aws.tenstorrent.com/
```

3. To test that everything is running correctly, try an example model. You can use nano or another text editor to paste this code into a file named **forge_example.py** and then run it from the terminal. You should still have your virtual environment running after installing the wheel when running this example:

```python
import numpy as np
import onnx
import onnx.helper as helper
import forge

# Create a minimal ONNX model (elementwise add of two tensors)
X = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4])
Y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 4])
Z = helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [1, 4])

add_node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"])
graph = helper.make_graph([add_node], "add_graph", [X, Y], [Z])
onnx_model = helper.make_model(graph)
onnx.checker.check_model(onnx_model)

# Compile and run on Tenstorrent hardware
x = np.random.rand(1, 4).astype(np.float32)
y = np.random.rand(1, 4).astype(np.float32)
compiled_model = forge.compile(onnx_model, sample_inputs=[x, y])

output = compiled_model(x, y)
print("Output:", output)
```

4. You have now set up the latest wheel for TT-Forge-ONNX, and can run any models you want inside your virtual environment.

## Other Set up Options
If you want to keep your environment completely separate in a Docker container, or you want to develop TT-Forge-ONNX further, this section links you to the pages with those options:

* [Setting up a Docker Container](getting_started_docker.md) - keep everything for running models in a container
* [Building from Source](getting_started_build_from_source.md) - set up so you can develop TT-Forge-ONNX

## Where to Go Next

Now that you have set up TT-Forge-ONNX, you can compile and run other demos or your own code. See the [TT-Forge-ONNX folder in the TT-Forge repo](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-forge-onnx) for more demo options.
