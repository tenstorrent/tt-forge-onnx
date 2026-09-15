"""Same build, real Wormhole n150: does the Quasar dispatch leak onto Wormhole?

The fork is `utils::isQuasar()` -> `getArch() == Arch::Quasar`, and getArch() is
`tt::tt_metal::hal::get_arch()` -- the live HAL of the attached device. So on
Wormhole the ternary must take `::ttnn::add`.

Two checks:
  1. Add still numerically correct on Wormhole.
  2. Mul -- which we never routed to Quasar -- also passes. On Quasar it dies in
     the mainline DataMovementKernel ctor; here it must not, which shows the
     mainline path is untouched.
"""
import time, traceback
import torch
from onnx import TensorProto, helper
import forge
from forge._C.runtime.experimental import TTSystem

SHAPE = [2, 32, 32]

def build(op):
    n = helper.make_node(op, inputs=["a", "b"], outputs=["o"])
    g = helper.make_graph([n], "G",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, SHAPE),
         helper.make_tensor_value_info("b", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("o", TensorProto.FLOAT, SHAPE)])
    return helper.make_model(g, producer_name="P",
                             opset_imports=[helper.make_operatorsetid("", 21)])

d = TTSystem.get_system().devices
print("[wh] driver-reported arch:", d[0].arch if d else "no device", flush=True)

torch.manual_seed(0)
inputs = [torch.rand(SHAPE), torch.rand(SHAPE)]

for op, ref in (("Add", lambda a, b: a + b), ("Mul", lambda a, b: a * b)):
    try:
        t0 = time.time()
        c = forge.compile(build(op), inputs, module_name=f"wh_{op.lower()}")
        out = c(*inputs)
        got = (out[0] if isinstance(out, (list, tuple)) else out).to(torch.float32).cpu()
        g_ = ref(*inputs).flatten().float()
        a_ = got.flatten()
        pcc = torch.corrcoef(torch.stack([g_, a_]))[0, 1].item()
        print(f"[wh] {op}: pcc={pcc:.6f} in {time.time()-t0:.1f}s -> "
              f"{'PASS' if pcc > 0.99 else 'FAIL'}", flush=True)
    except Exception as e:
        msg = "".join(traceback.format_exception_only(type(e), e)).strip()
        print(f"[wh] {op}: RAISED {msg[:300]}", flush=True)
        print(f"[wh] {op}: quasar-guard hit?"
              f" {'YES' if 'not supported on Quasar' in msg else 'no'}", flush=True)
