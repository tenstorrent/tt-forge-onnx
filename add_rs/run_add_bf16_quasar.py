"""Run a single ONNX Add on the Quasar simulator through forge, in bf16.

f32 hangs on this pin; bf16 is the hypothesis. Compiles with
default_df_override=Float16_b so the emitted TTNN tensors are bf16, then actually
executes and checks PCC against torch.
"""
import json, sys, time
import torch
from onnx import TensorProto, helper
import forge
from forge.config import CompilerConfig

SHAPE = [2, 32, 32]
n = helper.make_node("Add", inputs=["a", "b"], outputs=["o"])
g = helper.make_graph([n], "AddGraph",
    [helper.make_tensor_value_info("a", TensorProto.FLOAT, SHAPE),
     helper.make_tensor_value_info("b", TensorProto.FLOAT, SHAPE)],
    [helper.make_tensor_value_info("o", TensorProto.FLOAT, SHAPE)])
m = helper.make_model(g, producer_name="AddModel",
                      opset_imports=[helper.make_operatorsetid("", 21)])

torch.manual_seed(0)
inputs = [torch.rand(SHAPE), torch.rand(SHAPE)]
golden = inputs[0] + inputs[1]

from forge._C.runtime.experimental import TTSystem
_d = TTSystem.get_system().devices
print("[bf16] DEVICE ARCH:", _d[0].arch if _d else "no device", flush=True)

cfg = CompilerConfig()
cfg.default_df_override = forge._C.DataFormat.Float16_b

t0 = time.time()
compiled = forge.compile(m, inputs, module_name="add_bf16_quasar", compiler_cfg=cfg)
print(f"[bf16] COMPILED in {time.time()-t0:.1f}s", flush=True)

src = json.loads(compiled.compiled_binary.as_json()).get("mlir", {}).get("source", "")
print("[bf16] IR dtype:", "bf16" if "bf16" in src else "f32", flush=True)

t1 = time.time()
out = compiled(*inputs)
print(f"[bf16] EXECUTED in {time.time()-t1:.1f}s", flush=True)

got = (out[0] if isinstance(out, (list, tuple)) else out).to(torch.float32).cpu()
g_, a_ = golden.flatten().float(), got.flatten()
pcc = torch.corrcoef(torch.stack([g_, a_]))[0, 1].item()
print(f"[bf16] RESULT pcc={pcc:.6f} max_abs_err={(g_-a_).abs().max().item():.5f}", flush=True)
print("[bf16] VERDICT:", "PASS" if pcc > 0.99 else "FAIL", flush=True)
