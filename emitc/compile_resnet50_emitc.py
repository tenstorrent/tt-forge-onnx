#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""ResNet-50: ONNX → Forge → EmitC → .so, with saved inputs and golden outputs."""

from __future__ import annotations

import os
import struct
import sys
import pathlib
from typing import Optional

import onnx
import torch

import forge
from third_party.tt_forge_models.resnet.pytorch import ModelLoader, ModelVariant

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent

ONNX_PATH   = SCRIPT_DIR / "onnx_files"  / "resnet50.onnx"
OUTPUT_CPP  = SCRIPT_DIR / "cpp_files"   / "resnet50_forward.cpp"
OUTPUT_SO   = SCRIPT_DIR / "so_files"    / "resnet50_forward.so"
INPUTS_DIR  = SCRIPT_DIR / "inputs"
GOLDEN_PATH = INPUTS_DIR / "golden_outputs.bin"

MODULE_NAME = "resnet50_onnx"
VARIANT     = ModelVariant.RESNET_50

# ---------------------------------------------------------------------------
# TTNN binary tensor-list format (little-endian)
#   Header  : magic(4B "TTNN") + version(u32) + num_tensors(u32)
#   Per tensor: ndim(u32) + shape(i64*ndim) + dtype_code(u32)
#             + data_size(u64) + raw bytes
# ---------------------------------------------------------------------------
_DTYPE_TO_CODE: dict[torch.dtype, int] = {
    torch.float32:  0,
    torch.float16:  1,
    torch.bfloat16: 2,
    torch.int32:    3,
    torch.int64:    4,
    torch.int8:     5,
    torch.uint8:    6,
}


def _save_tensor_list(tensors: list[torch.Tensor], path: pathlib.Path) -> None:
    with open(str(path), "wb") as f:
        f.write(b"TTNN")
        f.write(struct.pack("<II", 1, len(tensors)))
        for t in tensors:
            tc = t.detach().cpu().contiguous()
            shape = list(tc.shape)
            dtype_code = _DTYPE_TO_CODE.get(tc.dtype, 0)
            data = tc.numpy().tobytes()
            f.write(struct.pack("<I", len(shape)))
            if shape:
                f.write(struct.pack(f"<{len(shape)}q", *shape))
            f.write(struct.pack("<IQ", dtype_code, len(data)))
            f.write(data)


# ===========================================================================
# Step 1: Load ResNet-50
# ===========================================================================
def load_model(variant: ModelVariant = VARIANT) -> torch.nn.Module:
    print(f"[1/10] Loading model ({variant.value}) ...")
    loader = ModelLoader(variant=variant)
    model  = loader.load_model()
    model.eval()
    return model


# ===========================================================================
# Step 2: Prepare sample inputs
# ===========================================================================
def load_inputs(batch_size: int = 1) -> torch.Tensor:
    print(f"[2/10] Preparing inputs (batch={batch_size}) ...")
    return torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)


# ===========================================================================
# Step 3: Export to ONNX
# ===========================================================================
def export_to_onnx(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    onnx_path: pathlib.Path = ONNX_PATH,
    opset_version: int = 17,
) -> onnx.ModelProto:
    print(f"[3/10] Exporting to ONNX (opset {opset_version}) → {onnx_path} ...")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            str(onnx_path),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
        )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"       ONNX model validated ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return onnx_model


# ===========================================================================
# Step 4: Compile with Forge
# ===========================================================================
def compile_model(
    onnx_model: onnx.ModelProto,
    sample_input: torch.Tensor,
) -> forge.CompiledModel:
    print("[4/10] Compiling with Forge ...")
    compiled = forge.compile(onnx_model, sample_inputs=[sample_input])
    print("       Forge compilation complete.")
    return compiled


# ===========================================================================
# Step 5: Run forward pass on device
# ===========================================================================
def run_model(compiled: forge.CompiledModel, inputs: torch.Tensor) -> list:
    print("[5/10] Running forward pass on device ...")
    outputs = compiled(inputs)
    print(f"       {len(outputs)} output(s).")
    return outputs


# ===========================================================================
# Step 6: Save golden outputs
# ===========================================================================
def save_outputs(
    outputs: list,
    golden_path: pathlib.Path = GOLDEN_PATH,
) -> None:
    print(f"[6/10] Saving golden outputs → {golden_path} ...")
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = []
    for o in outputs:
        if hasattr(o, "to_torch"):
            tensors.append(o.to_torch())
        else:
            tensors.append(o.detach().cpu().contiguous())
    _save_tensor_list(tensors, golden_path)
    print(f"       {len(tensors)} tensor(s) saved.")


# ===========================================================================
# Step 7: Convert compiled model to C++ via EmitC
# ===========================================================================
def convert_to_cpp(compiled: forge.CompiledModel) -> str:
    print("[7/10] Converting to C++ via EmitC pipeline ...")
    from forge._C import run_mlir_compiler_to_cpp
    cpp_source = run_mlir_compiler_to_cpp(compiled.forge_graph_module)
    print("       C++ source generated.")
    return cpp_source


# ===========================================================================
# Step 8: Save C++ source to disk
# ===========================================================================
def save_cpp(cpp_source: str, output_path: pathlib.Path = OUTPUT_CPP) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cpp_source, encoding="utf-8")
    print(f"[8/10] Saved C++ → {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


# ===========================================================================
# Step 9: Save activation and persistent inputs
# ===========================================================================
def save_inputs(
    compiled: forge.CompiledModel,
    inputs_dir: pathlib.Path = INPUTS_DIR,
) -> None:
    print(f"[9/10] Saving inputs → {inputs_dir} ...")
    inputs_dir.mkdir(parents=True, exist_ok=True)

    from forge._C.runtime import ProgramType, testutils as rt_testutils

    act_tensors: list[torch.Tensor] = [ct.to_torch() for ct in compiled.inputs]
    act_path = inputs_dir / "activation_inputs.bin"
    _save_tensor_list(act_tensors, act_path)
    print(f"       activation_inputs.bin  : {len(act_tensors)} tensor(s)")

    consts_params = rt_testutils.get_persistent_inputs(
        ProgramType.Forward, compiled.runtime_model_state
    )
    pers_tensors: list[torch.Tensor] = [ct.to_torch() for ct in consts_params]
    pers_path = inputs_dir / "persistent_inputs.bin"
    _save_tensor_list(pers_tensors, pers_path)
    print(f"       persistent_inputs.bin  : {len(pers_tensors)} tensor(s)")


# ===========================================================================
# Step 10: Compile .cpp → .so
# ===========================================================================
def resolve_metal_paths() -> tuple[str, str, str]:
    metal_runtime_root = os.environ.get("TT_METAL_RUNTIME_ROOT")
    forge_home         = os.environ.get("FORGE_HOME")

    if not metal_runtime_root:
        raise RuntimeError("TT_METAL_RUNTIME_ROOT is not set.")
    if not forge_home:
        raise RuntimeError("FORGE_HOME is not set.")

    metal_src_dir = metal_runtime_root

    if os.environ.get("FORGE_IN_WHEEL"):
        metal_lib_dir  = str(pathlib.Path(forge_home) / "forge" / "lib")
        standalone_dir = str(pathlib.Path(forge_home) / "forge" / "tools" / "ttnn-standalone")
    elif os.environ.get("FORGE_IN_SOURCE"):
        # TT_METAL_RUNTIME_ROOT = <install>/tt-metal  → lib dir is one level up: <install>/lib
        metal_lib_dir  = str(pathlib.Path(metal_runtime_root).parent / "lib")
        standalone_dir = str(
            pathlib.Path(forge_home) / "third_party" / "tt-mlir" / "tools" / "ttnn-standalone"
        )
    else:
        raise RuntimeError("Neither FORGE_IN_WHEEL nor FORGE_IN_SOURCE is set.")

    return metal_src_dir, metal_lib_dir, standalone_dir


def compile_cpp_to_so(
    cpp_path: pathlib.Path,
    output_dir: Optional[pathlib.Path] = None,
) -> pathlib.Path:
    print("[10/10] Compiling C++ → shared object ...")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    metal_src_dir, metal_lib_dir, standalone_dir = resolve_metal_paths()

    if standalone_dir not in sys.path:
        sys.path.insert(0, standalone_dir)

    from emitc_compiler import compile_emitc_to_so

    so_path_str = compile_emitc_to_so(
        cpp_file_path=str(cpp_path),
        output_dir=str(output_dir) if output_dir else str(cpp_path.parent),
        build_type="Release",
        incremental=False,
        metal_src_dir=metal_src_dir,
        metal_lib_dir=metal_lib_dir,
        verbose=True,
    )

    so_path = pathlib.Path(so_path_str)
    print(f"       Shared object → {so_path} ({so_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return so_path


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    print("=" * 60)
    print("  ResNet-50  ONNX → Forge → EmitC → .so")
    print("=" * 60)

    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    model   = load_model()
    inputs  = load_inputs(batch_size=1)
    onnx_m  = export_to_onnx(model, inputs, ONNX_PATH)
    compiled = compile_model(onnx_m, inputs)
    outputs  = run_model(compiled, inputs)

    save_outputs(outputs, GOLDEN_PATH)

    cpp_source = convert_to_cpp(compiled)
    save_cpp(cpp_source, OUTPUT_CPP)
    save_inputs(compiled, INPUTS_DIR)
    so_path = compile_cpp_to_so(OUTPUT_CPP, SCRIPT_DIR / "so_files")

    print()
    print("Done!")
    print(f"  ONNX model        : {ONNX_PATH}")
    print(f"  C++ source        : {OUTPUT_CPP}")
    print(f"  Shared object     : {so_path}")
    print(f"  Golden outputs    : {GOLDEN_PATH}")
    print(f"  Activation inputs : {INPUTS_DIR / 'activation_inputs.bin'}")
    print(f"  Persistent inputs : {INPUTS_DIR / 'persistent_inputs.bin'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
