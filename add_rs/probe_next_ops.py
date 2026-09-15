"""Compile-only probe: which ttnn ops does forge emit for these single-op models?

No execution -- we only need to know which runtime dispatch sites a candidate op
would actually reach, before writing any dispatch code for it.
"""
import json
import numpy as np
import torch
from onnx import TensorProto, helper, numpy_helper
import forge
from forge.config import CompilerConfig
from forge._C.runtime.experimental import TTSystem

d = TTSystem.get_system().devices
print("arch:", d[0].arch if d else "no device", flush=True)

def model(nodes, inputs, outputs, inits=()):
    g = helper.make_graph(nodes, "G", inputs, outputs, list(inits))
    return helper.make_model(g, producer_name="P",
                             opset_imports=[helper.make_operatorsetid("", 21)])

def vi(name, shape, t=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, t, shape)

CASES = {}

# Transpose: swap the last two axes of a 3-D tensor
CASES["transpose"] = (
    model([helper.make_node("Transpose", ["a"], ["o"], perm=[0, 2, 1])],
          [vi("a", [1, 64, 32])], [vi("o", [1, 32, 64])]),
    [torch.rand(1, 64, 32)],
)

# Reshape: fold two axes together
_shape = numpy_helper.from_array(np.array([1, 2048], dtype=np.int64), name="s")
CASES["reshape"] = (
    model([helper.make_node("Reshape", ["a", "s"], ["o"])],
          [vi("a", [1, 64, 32])], [vi("o", [1, 2048])], [_shape]),
    [torch.rand(1, 64, 32)],
)

# Cast: f32 -> bf16 (the typecast dispatch site)
CASES["typecast"] = (
    model([helper.make_node("Cast", ["a"], ["o"], to=TensorProto.BFLOAT16)],
          [vi("a", [1, 64, 32])], [vi("o", [1, 64, 32], TensorProto.BFLOAT16)]),
    [torch.rand(1, 64, 32)],
)

for name, (m, inputs) in CASES.items():
    cfg = CompilerConfig()
    cfg.default_df_override = forge._C.DataFormat.Float16_b
    try:
        c = forge.compile(m, inputs, module_name=f"probe_{name}", compiler_cfg=cfg)
        src = json.loads(c.compiled_binary.as_json()).get("mlir", {}).get("source", "")
        ops = sorted({l.split('"')[1] for l in src.splitlines()
                      if '"ttnn.' in l for _ in [0]}) if src else []
        ops = sorted({tok for l in src.splitlines() for tok in l.split('"')
                      if tok.startswith("ttnn.")})
        print(f"[{name}] emits: {', '.join(ops) or '(none)'}", flush=True)
    except Exception as e:
        print(f"[{name}] COMPILE FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
