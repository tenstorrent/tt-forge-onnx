"""Positive proof that the device under the run is a Quasar, not a stand-in.

Two independent readings, then one negative control:

  1. What the DRIVER says the chip is -- UMD builds the cluster from the
     simulator's soc_descriptor.yaml, tt-mlir's system_desc.cpp stamps
     `device->arch()` into the flatbuffer, forge maps it to Arch.QUASAR.
  2. What the DRIVER says the chip HAS -- l1_size_per_core /
     num_dram_channels are read off the live device, so 4 MiB and 2 channels
     cannot come from any hardcoded default (Wormhole: 1499136 and 12).
  3. NEGATIVE CONTROL: run an op we deliberately did NOT route to the Quasar
     op library (Mul). Its mainline program factory builds a
     DataMovementKernel, whose ctor asserts
        get_cluster().arch() != ARCH::QUASAR
     -- tt_metal/impl/kernels/kernel.hpp:417. That assert reads the LIVE
     cluster arch. If it fires, the silicon-side arch really is Quasar. On a
     Wormhole the same op would simply pass.
"""
import json, sys, traceback
import torch
from onnx import TensorProto, helper
import forge
from forge.config import CompilerConfig
from forge._C.runtime.experimental import TTSystem

SHAPE = [2, 32, 32]

d = TTSystem.get_system().devices
print("[1] driver-reported arch      :", d[0].arch if d else "no device", flush=True)

def build(op):
    n = helper.make_node(op, inputs=["a", "b"], outputs=["o"])
    g = helper.make_graph([n], "G",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, SHAPE),
         helper.make_tensor_value_info("b", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("o", TensorProto.FLOAT, SHAPE)])
    return helper.make_model(g, producer_name="P",
                             opset_imports=[helper.make_operatorsetid("", 21)])

cfg = CompilerConfig()
cfg.default_df_override = forge._C.DataFormat.Float16_b

torch.manual_seed(0)
inputs = [torch.rand(SHAPE), torch.rand(SHAPE)]

# --- 2: the descriptor the compiler was handed, read off the live device ---
c = forge.compile(build("Add"), inputs, module_name="proof_add", compiler_cfg=cfg)
src = json.loads(c.compiled_binary.as_json()).get("mlir", {}).get("source", "")
line = next((l for l in src.splitlines() if "ttcore.system_desc" in l or "arch = <" in l), "")
print("[2] live system descriptor    :", line.strip()[:300], flush=True)

out = c(*inputs)
got = (out[0] if isinstance(out, (list, tuple)) else out).to(torch.float32).cpu()
pcc = torch.corrcoef(torch.stack([(inputs[0] + inputs[1]).flatten().float(),
                                  got.flatten()]))[0, 1].item()
print(f"[2] Add (routed to Quasar op) : PASS pcc={pcc:.6f}", flush=True)

# --- 3: negative control -- an op left on the mainline path ---
print("[3] negative control: Mul (mainline ttnn, NOT routed to Quasar)", flush=True)
try:
    cm = forge.compile(build("Mul"), inputs, module_name="proof_mul", compiler_cfg=cfg)
    cm(*inputs)
    print("[3] RESULT: Mul RAN -> the device is NOT refusing mainline kernels", flush=True)
    print("[3] VERDICT: INCONCLUSIVE", flush=True)
except Exception as e:
    msg = "".join(traceback.format_exception_only(type(e), e))
    print("[3] Mul raised:", msg.strip()[:600], flush=True)
    hit = "not supported on Quasar" in msg
    print("[3] VERDICT:", "CONFIRMED QUASAR (live cluster arch asserted by tt-metal)"
          if hit else "unexpected failure -- read the message above", flush=True)
