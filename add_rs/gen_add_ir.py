#!/usr/bin/env python3
"""Compile a one-node ONNX Add for each arch and save the TTIR and TTNN IR.

No device: set_target_arch names the target, so emit_mlir skips stamping the live
descriptor and tt-mlir's TTCoreRegisterDevice builds a mock one instead.

    python add_rs/gen_add_ir.py
"""
import json, os, shutil, sys

import torch
from onnx import TensorProto, helper

import forge
from forge.config import CompilerConfig, MLIRConfig

OUT = os.path.dirname(os.path.abspath(__file__))
SHAPE = [2, 32, 32]
ARCHES = {
    "wormhole_b0": forge._C.Arch.WORMHOLE_B0,
    "quasar": forge._C.Arch.QUASAR,
}


def build_add():
    node = helper.make_node("Add", inputs=["input_A", "input_B"], outputs=["output"])
    graph = helper.make_graph(
        nodes=[node],
        name="AddGraph",
        inputs=[
            helper.make_tensor_value_info("input_A", TensorProto.FLOAT, SHAPE),
            helper.make_tensor_value_info("input_B", TensorProto.FLOAT, SHAPE),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
    )
    return helper.make_model(
        graph, producer_name="add_rs", opset_imports=[helper.make_operatorsetid("", 21)]
    )


for name, arch in ARCHES.items():
    module = f"add_rs_{name}"
    cfg = CompilerConfig(mlir_config=MLIRConfig().set_target_arch(arch))
    compiled = forge.compile(
        build_add(),
        [torch.rand(SHAPE), torch.rand(SHAPE)],
        module_name=module,
        compiler_cfg=cfg,
    )

    binary = compiled.compiled_binary
    blob = json.loads(binary.as_json())

    # The TTNN module is embedded in the flatbuffer, so it needs no reportify.
    ttnn_src = blob.get("mlir", {}).get("source", "")
    with open(os.path.join(OUT, f"ttnn_{name}.mlir"), "w") as fh:
        fh.write(ttnn_src)

    # TTIR comes from reportify's dump.
    rep = os.path.expanduser(f"~/testify/ll-sw/{module}/mlir_reports/ttir.mlir")
    if os.path.exists(rep):
        shutil.copy(rep, os.path.join(OUT, f"ttir_{name}.mlir"))

    binary.store(os.path.join(OUT, f"add_{name}.ttnn"))
    with open(os.path.join(OUT, f"system_desc_{name}.json"), "w") as fh:
        json.dump(blob.get("system_desc", {}), fh, indent=2)

    ops = blob.get("programs", [{}])[0].get("operations", [])
    print(
        f"{name:12} ops={len(ops):2}  ttnn={len(ttnn_src.splitlines()):3} lines  "
        f"mlir.name={blob.get('mlir',{}).get('name')}"
    )
print("done")
