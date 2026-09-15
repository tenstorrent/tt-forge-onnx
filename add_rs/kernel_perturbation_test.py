"""Single bf16 ONNX Add, printing enough to tell where the numbers came from.

Used by the kernel-perturbation experiment: run once with the Quasar eltwise
compute kernel untouched, once with its output pack_tile removed. If the host
sees the same answer both times, the host is not getting its numbers from that
kernel.
"""
import sys, time
import torch
from onnx import TensorProto, helper
import forge
from forge.config import CompilerConfig
from forge._C.runtime.experimental import TTSystem

TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
SHAPE = [2, 32, 32]

n = helper.make_node("Add", inputs=["a", "b"], outputs=["o"])
g = helper.make_graph([n], "G",
    [helper.make_tensor_value_info("a", TensorProto.FLOAT, SHAPE),
     helper.make_tensor_value_info("b", TensorProto.FLOAT, SHAPE)],
    [helper.make_tensor_value_info("o", TensorProto.FLOAT, SHAPE)])
m = helper.make_model(g, producer_name="P",
                      opset_imports=[helper.make_operatorsetid("", 21)])

d = TTSystem.get_system().devices
print(f"[{TAG}] arch:", d[0].arch if d else "no device", flush=True)

torch.manual_seed(0)
inputs = [torch.rand(SHAPE), torch.rand(SHAPE)]
golden = inputs[0] + inputs[1]

cfg = CompilerConfig()
cfg.default_df_override = forge._C.DataFormat.Float16_b

t0 = time.time()
c = forge.compile(m, inputs, module_name=f"perturb_{TAG}", compiler_cfg=cfg)
print(f"[{TAG}] compiled in {time.time()-t0:.1f}s", flush=True)

t1 = time.time()
out = c(*inputs)
print(f"[{TAG}] executed in {time.time()-t1:.1f}s", flush=True)

got = (out[0] if isinstance(out, (list, tuple)) else out).to(torch.float32).cpu()
g_, a_ = golden.flatten().float(), got.flatten()
pcc = torch.corrcoef(torch.stack([g_, a_]))[0, 1].item()
print(f"[{TAG}] golden[0:4] = {[round(v,4) for v in g_[:4].tolist()]}", flush=True)
print(f"[{TAG}] device[0:4] = {[round(v,4) for v in a_[:4].tolist()]}", flush=True)
print(f"[{TAG}] pcc={pcc:.6f} max_abs_err={(g_-a_).abs().max().item():.5f}", flush=True)
print(f"[{TAG}] MATCHES_A_PLUS_B:", "YES" if pcc > 0.99 else "NO", flush=True)
