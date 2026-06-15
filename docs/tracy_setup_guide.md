# Tracy Profiler Setup Guide — tt-forge-onnx

This guide walks you through building **tt-forge-onnx from source** with the
Tracy profiler enabled, and running it against the **ResNet50 ONNX benchmark**
to generate a per-op performance report.

> **What is Tracy?**
> Tracy is an integrated performance profiling system built into tt-metal. It
> captures host-side Python/C++ timing, device-side kernel execution cycles, and
> op-to-op dispatch latency — all in a single run — and produces a CSV report
> you can inspect.

---

## Prerequisites at a Glance

| Tool | Required version |
|---|---|
| OS | Ubuntu 22.04 / 24.04 |
| Clang | 17 |
| Python | 3.12 |
| CMake | ≥ 3.20 |
| Ninja | any |
| uv | any |

---

## Step 1 — Install System Dependencies

Update your package list first, then install each dependency group below.

```bash
sudo apt update -y && sudo apt upgrade -y
```

### 1.1 Clang 17

Tenstorrent's official compiler is Clang 17. Install it and create the default
symlinks so the build system can find it:

```bash
wget https://apt.llvm.org/llvm.sh && chmod u+x llvm.sh && sudo ./llvm.sh 17
sudo apt-get install -y libc++-17-dev libc++abi-17-dev
sudo ln -sf /usr/bin/clang-17    /usr/bin/clang
sudo ln -sf /usr/bin/clang++-17  /usr/bin/clang++
sudo ln -sf /usr/bin/FileCheck-17 /usr/bin/FileCheck
```

Verify:
```bash
clang --version    # expected: clang version 17.x.x
clang++ --version  # expected: clang version 17.x.x
```

### 1.2 Build Tools and Dev Libraries

```bash
sudo apt install -y ninja-build g++ libstdc++-14-dev \
    libgmock-dev libnuma-dev libhwloc-dev doxygen libboost-container-dev
```

### 1.3 Python 3.12

```bash
sudo apt install -y python3.12
```

Verify:
```bash
python3 --version  # expected: Python 3.12.x
```

### 1.4 uv — Python Package Manager

`uv` is used to install CMake and other Python build tools:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # reload PATH to pick up uv
```

Verify:
```bash
uv --version
```

### 1.5 CMake

```bash
uv pip install cmake
```

Verify:
```bash
cmake --version   # expected: cmake 3.20 or later
```

---

## Step 2 — Clone and Checkout the Tracy Branch

### 2.1 Create toolchain directories

These are required by the build system (one-time setup):

```bash
sudo mkdir -p /opt/ttforge-toolchain && sudo chown -R $USER /opt/ttforge-toolchain
sudo mkdir -p /opt/ttmlir-toolchain  && sudo chown -R $USER /opt/ttmlir-toolchain
```

### 2.2 Clone the repository

```bash
git clone https://github.com/tenstorrent/tt-forge-onnx.git
cd tt-forge-onnx
```

### 2.3 Switch to the Tracy integration branch

```bash
git fetch origin
git checkout pchandrasekaran/tracy_profiler_integration
```

### 2.4 Initialise submodules

```bash
git submodule update --init --recursive -f
```

---

## Step 3 — Build the Toolchain

> **One-time step.** This builds the LLVM/MLIR toolchain and virtual
> environment.

```bash
cmake -B env/build env
cmake --build env/build
```

---

## Step 4 — Activate the Virtual Environment

```bash
source env/activate
```

> **Important:** Run `source env/activate` at the start of every new shell
> session before using any project commands.

---

## Step 5 — Build tt-forge-onnx with Tracy Enabled

The `TT_RUNTIME_ENABLE_PERF_TRACE=ON` flag is already set in
`third_party/CMakeLists.txt` on this branch. This instructs tt-metal to compile
Tracy instrumentation zones into the binary, so no extra flag is needed.

```bash
cmake -G Ninja -B build \
    -DCMAKE_C_COMPILER=clang-17 \
    -DCMAKE_CXX_COMPILER=clang++-17 \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

cmake --build build
```

---

## Step 6 — Install Python Dependency

```bash
pip install graphviz
```

---

## Step 7 — Run the Tracy Profiler on ResNet50

The `scripts/tracy_run.sh` wrapper handles all environment setup, copies the
patched post-processors into the tt-metal install tree, starts the Tracy capture
process, runs your test, and post-processes the results into a CSV report.

```bash
bash scripts/tracy_run.sh \
    -o resnet50_tracy_profile \
    -n resnet50 \
    -p \
    --no-device-trace \
    --dispatch-cores \
    --op-count 3000 \
    -- pytest forge/test/models/onnx/vision/resnet/test_resnet.py \
        -k 'ResNet50' -vss
```

### What each flag does

| Flag | Description |
|---|---|
| `-o resnet50_tracy_profile` | Directory where all output files are written (relative to repo root) |
| `-n resnet50` | Appended to the CSV report filename for easy identification |
| `-p` | Partial profile — only capture zones explicitly marked by Tracy, reducing overhead |
| `--no-device-trace` | Use per-op profiling mode instead of Metal Trace mode |
| `--dispatch-cores` | Enable dispatch-core profiling (`TT_METAL_DEVICE_PROFILER_DISPATCH=1`) |
| `--op-count 3000` | Maximum number of ops the on-device profiler buffer can hold before overflow |

---

## Step 8 — Inspect the Output

After the run completes, the output directory contains:

```
resnet50_tracy_profile/
│
├── .logs/                                         ← raw captured data
│   ├── tracy_profile_log_host.tracy               ← open in Tracy GUI
│   ├── profile_log_device.csv                     ← raw device timing (cycles)
│   ├── tracy_ops_data.csv                         ← host-side op metadata
│   └── tracy_ops_times.csv                        ← host-side op durations
│
└── reports/resnet50/<YYYY_MM_DD_HH_MM_SS>/        ← post-processed results
    └── ops_perf_results_resnet50_<date>.csv        ← main report (279 ops, 92 columns)
```

---

## Step 9 — Populated CSV Columns

The main CSV report contains one row per dispatched op with the following data:

| Column Group | What it tells you |
|---|---|
| **Identity** | OP CODE, OP TYPE, GLOBAL CALL COUNT, DEVICE ID, ATTRIBUTES, MATH FIDELITY |
| **Host timing** | How long Python/C++ spent dispatching each op (HOST DURATION [ns]) |
| **Device timing** | How long the op actually ran on-chip — firmware duration, kernel duration, per-RISC breakdown (BRISC/NCRISC/TRISC0/1/2), and per-core min/max/avg |
| **Dispatch latency** | Gap between the end of one op and the start of the next (OP TO OP LATENCY [ns]) |
| **Tensor I/O** | Input/output tensor shapes, layouts, data types, and memory locations |
| **Kernel info** | Kernel source paths, hashes, program cache hit/miss, compiled kernel sizes |
| **Performance model** | Ideal, compute-bound, and bandwidth-bound time estimates; PM FPU UTIL (%) |
