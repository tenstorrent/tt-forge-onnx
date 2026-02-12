# Getting Started with Building from Source

This document describes how to build the TT-Forge-ONNX project on your local machine. You must build from source if you want to develop for TT-Forge-ONNX. If you only want to run models, please choose one of the following sets of instructions instead:
* [Installing a Wheel and Running an Example](getting_started.md) - You should choose this option if you want to run models.
* [Using a Docker Container to Run an Example](getting_started_docker.md) - Choose this option if you want to keep the environment for running models separate from your existing environment.

> **NOTE:** TT-Forge-ONNX is a framework agnostic frontend  that can convert any model to a generic
> Intermediate Representation (IR) that can then be converted to a Tenstorrent specific IR for use with
> Tenstorrent hardware. TT-Forge-ONNX is for use with single-chip systems only.

The topics covered in this document are:
* [Configuring Your Hardware](#configuring-your-hardware)
* [Prerequisites](#prerequisites)
* [Building the Environment](#building-the-environment)
* [Building the Docs](#building-the-docs)
* [Build Cleanup](#build-cleanup)
* [Useful Build Environment Variables](#useful-build-environment-variables)

> **NOTE:** If you encounter issues, please request assistance on the
>[TT-Forge-ONNX Issues](https://github.com/tenstorrent/tt-forge-onnx/issues) page.

## Configuring Your Hardware
If you already configured your hardware, you can skip this section. Otherwise do the following:

1. Configure your hardware with TT-Installer using the [Quick Installation section here.](https://docs.tenstorrent.com/getting-started/README.html#quick-installation)

2. Reboot your machine.

3. Please ensure that after you run this script, after you complete reboot, you activate the virtual environment it sets up - ```source ~/.tenstorrent-venv/bin/activate```.

4. After your environment is running, to check that everything is configured, type the following:

```bash
tt-smi
```

You should see the Tenstorrent System Management Interface. It allows you to view real-time stats, diagnostics, and health info about your Tenstorrent device.

## Prerequisites
The prerequisites for building TT-Forge-ONNX from source are:

* Clang 17
* Ninja
* CMake (latest)
* Python 3.11

On Ubuntu 22.04 systems, you can install these dependencies using the following commands:

```bash
# Update package list
sudo apt update -y
sudo apt upgrade -y
```

### Installing Clang
To install Clang if you do not have it already, use the following command:

```bash
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
sudo ./llvm.sh 17
sudo apt install -y libc++-17-dev libc++abi-17-dev
sudo ln -s /usr/bin/clang-17 /usr/bin/clang
sudo ln -s /usr/bin/clang++-17 /usr/bin/clang++
```

You can check the version afterwards with these commands:

```bash
clang --version
clang++ --version
```

If you already have Clang installed and need to choose the appropriate version, you can use these commands:

```bash
sudo update-alternatives --install /usr/bin/clang
clang /usr/bin/clang-17 100
sudo update-alternatives --install /usr/bin/clang++
clang++ /usr/bin/clang++-17 100
```

### Installing Ninja
Install Ninja with the following command:

```bash
sudo apt install ninja-build
```

### Checking Python Version
Make sure you have Python 3.11 installed:

```bash
python3 --version
```

If you do not have Python 3.11 installed:

```bash
sudo apt install python3.11
```

### Installing CMake
Install CMake and check the version with the following commands:

```bash
pip install cmake
```

Check that it installed:

```bash
cmake --version
```

### Installing Additional Dependencies
This section goes over additional required dependencies. You may wish to check if you already have them installed before running installation steps for each item. Run the following commands:

1. Install the required development packages:

```bash
sudo apt install -y \
    g++ \
    libstdc++-12-dev \
    libgmock-dev \
    libnuma-dev \
    libhwloc-dev \
    doxygen \
    libboost-container-dev
```

2. Download and install the MPI implementation:

```bash
wget -q https://github.com/dmakoviichuk-tt/mpi-ulfm/releases/download/v5.0.7-ulfm/openmpi-ulfm_5.0.7-1_amd64.deb -O /tmp/openmpi-ulfm.deb && \
sudo apt install -y /tmp/openmpi-ulfm.deb
```

3. Export environment variables:

```bash
export PATH=/opt/openmpi-v5.0.7-ulfm/bin:$PATH
export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
```

## Building the Environment
This is a one off step to build the toolchain and create a virtual environment for TT-Forge-ONNX. Generally, you need to run this step only once, unless you want to update the toolchain. Since TT-Forge-ONNX uses TT-MLIR, this step also builds the TT-MLIR environment (toolchain).

1. First, it's required to create toolchain directories. The proposed example creates directories using the default paths. You can change the paths if you want to use different locations (see the [Useful Build Environment Variables](#useful-build-environment-variables) section below).

```bash
# FFE related toolchain (default path)
sudo mkdir -p /opt/ttforge-toolchain
sudo chown -R $USER /opt/ttforge-toolchain

# MLIR related toolchain (default path)
sudo mkdir -p /opt/ttmlir-toolchain
sudo chown -R $USER /opt/ttmlir-toolchain
```

2. Clone the TT-Forge-ONNX repo:

```bash
git clone https://github.com/tenstorrent/tt-forge-onnx.git
```

3. Navigate into the TT-Forge-ONNX repo.

4. Initialize required env variables:

```bash
source env/activate
```

> **NOTE:** You will not see a virtual environment start from this command. That is expected behavior.

5. Initialize and update submodules:

```bash
sudo git submodule update --init --recursive -f
```

6. Build the environment:

```bash
cmake -B env/build env
cmake --build env/build
```

> **Expert Tip:** If you already have the TT-MLIR toolchain built, you can use the `TTFORGE_SKIP_BUILD_TTMLIR_ENV` option to skip rebuilding the TT-MLIR environment (toolchain) to save time. Like so:
> ```bash
> cmake -B env/build env -DTTFORGE_SKIP_BUILD_TTMLIR_ENV=ON
> cmake --build env/build
> ```
>
> **NOTE:** Special care should be taken to ensure that the already built TT-MLIR environment (toolchain) version is compatible with the one TT-Forge-ONNX is using.

7. Activate the virtual environment for TT-Forge-ONNX. (This time when you run the command, you should see a running virtual environment):

```bash
source env/activate
```

8. Build the TT-Forge-ONNX environment:

```bash
cmake -G Ninja -B build -DCMAKE_CXX_COMPILER=clang++-17 -DCMAKE_C_COMPILER=clang-17
cmake --build build
```

> **NOTE:** Tenstorrent's official compiler is Clang 17.
>
> If you want to try other compilers, while they are not tested, you can do so by changing the `-DCMAKE_CXX_COMPILER` and `-DCMAKE_C_COMPILER` options.

You can pass additional options to the `cmake` command to customize the build. For example, to build everything in debug mode, you can run:
```sh
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++-17 -DCMAKE_C_COMPILER=clang-17
cmake --build build
```

> List of commonly used options:
> - `-DCMAKE_BUILD_TYPE=Debug|Release`      - Build type (Debug, Release)
> - `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache`  - Use [`ccache`](https://ccache.dev/) to speed up re-builds
> - `-DTTMLIR_RUNTIME_DEBUG=ON|OFF`         - Build runtime debug tools (more logging, debug environment flags)

### Incremental Building
If you have made changes to the C++ sources (of the TT-Forge-ONNX compiler, TT-MLIR or TT-Metal), you might want to do an incremental build to save time. This can be done by running the following command:
```bash
# If you are not already inside the virtual environment, activate it
source env/activate

cmake --build build -- install_ttforge
```

This will build TT-Forge-ONNX C++ sources and the dependencies (TT-MLIR, TT-Metal) and install them in the virtual environment.

## Building the Docs

To build documentation, mdBook is required, see the installation guide [here](./tools.md#mdbook).

After installing mdBook, run the following commands to build and serve the documentation:

```bash
source env/activate
cmake --build build -- docs

# Serve the documentation
mdbook serve build/docs
```

> **Note:** `mdbook serve` will by default create a local server at `http://localhost:3000`.

> **Note:** For a custom port, specify the `-p` attribute. <br><br> E.g. `mdbook serve build/docs -p 5005`, and visit `http://localhost:5005`.

## Build Cleanup

To ensure a clean build environment, follow these steps to remove existing build artifacts:

1. Remove TT-Forge-ONNX build artifacts:
    ```bash
    rm -rf build
    ```
    > **NOTE:** This command removes the `build` directory and all its contents, effectively cleaning up the build artifacts specific to tt-forge-onnx.

2. Clean all TT-Forge-ONNX build artifacts:
     ```bash
     ./clean_build.sh
     ```
   > **NOTE:** This script executes a comprehensive cleanup, removing all build artifacts across the entire Forge project, ensuring a clean slate for subsequent builds.

   > **NOTE:** The `clean_build.sh` script will not clean toolchain (LLVM) build artifacts and dependencies.

3. Clean everything (including the environment):
    ```bash
    ./clean_build.sh
    rm -rf env/build third_party/tt-mlir/env/build
    ```
    > **NOTE:** This should rarely be needed, as it removes the entire build and environment (consequently entire toolchain will need to be rebuilt).

## Useful Build Environment Variables
This section goes over some useful environment variables for use with the [Building the Environment](#building-the-environment) section.

* `TTMLIR_TOOLCHAIN_DIR` - Specifies the directory where TTMLIR dependencies will be installed. Defaults to `/opt/ttmlir-toolchain` if not defined.
* `TTMLIR_VENV_DIR` - Specifies the virtual environment directory for TTMLIR. Defaults to `/opt/ttmlir-toolchain/venv` if not defined.
* `TTFORGE_TOOLCHAIN_DIR` - Specifies the directory where tt-forge dependencies will be installed. Defaults to `/opt/ttforge-toolchain` if not defined.
* `TTFORGE_VENV_DIR` - Specifies the virtual environment directory for tt-forge. Defaults to `/opt/ttforge-toolchain/venv` if not defined.
* `TTFORGE_PYTHON_VERSION` - Specifies the Python version to use. Defaults to `python3.11` if not defined.
