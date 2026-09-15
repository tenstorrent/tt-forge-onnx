#!/usr/bin/env python3
"""Compile the same Add against the LIVE craq-sim device and dump its descriptor.

No target_arch is named, so emit_mlir stamps the module with the attached device's
real descriptor -- which is what we want to diff against the mock. Compile only; the
graph is never executed.
"""
import json, os
import torch
from onnx import TensorProto, helper
import forge
from forge.config import CompilerConfig

OUT = os.path.dirname(os.path.abspath(__file__))
SHAPE = [2, 32, 32]

from forge._C.runtime.experimental import TTSystem
d = TTSystem.get_system().devices[0]
print(f"device arch as forge sees it: {d.arch}", flush=True)

node = helper.make_node("Add", inputs=["input_A", "input_B"], outputs=["output"])
graph = helper.make_graph(
    nodes=[node], name="AddGraph",
    inputs=[helper.make_tensor_value_info("input_A", TensorProto.FLOAT, SHAPE),
            helper.make_tensor_value_info("input_B", TensorProto.FLOAT, SHAPE)],
    outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
)
model = helper.make_model(graph, producer_name="add_rs_live",
                          opset_imports=[helper.make_operatorsetid("", 21)])

compiled = forge.compile(model, [torch.rand(SHAPE), torch.rand(SHAPE)],
                         module_name="add_rs_live", compiler_cfg=CompilerConfig())
blob = json.loads(compiled.compiled_binary.as_json())
with open(os.path.join(OUT, "system_desc_quasar_LIVE.json"), "w") as fh:
    json.dump(blob.get("system_desc", {}), fh, indent=2)
src = blob.get("mlir", {}).get("source", "")
with open(os.path.join(OUT, "ttnn_quasar_LIVE.mlir"), "w") as fh:
    fh.write(src)
print("wrote system_desc_quasar_LIVE.json and ttnn_quasar_LIVE.mlir", flush=True)
