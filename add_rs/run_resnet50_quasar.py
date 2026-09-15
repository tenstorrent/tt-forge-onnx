"""Run the whole ResNet-50 graph on Quasar and report how far it gets.

Argv: input spatial size (default 32). The op inventory is identical at any
resolution -- same 53 conv2d, same prepare_conv2d_weights/bias, same
to_memory_config, same mean/max_pool2d/linear -- so a small input exercises every
op the real 224x224 model needs at a cost the simulator can actually pay.

PROBE_OPT=1 selects the BENCHMARK-optimised pipeline. That matters: the default
pipeline lowers the global average pool to mean(dim=[3]), the unported W reduce,
so only the optimised pipeline can get through.
"""
import os, sys, json, tempfile, traceback
import numpy as np
import torch
import forge
from forge.config import CompilerConfig, MLIRConfig
import torchvision

SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 32
torch.manual_seed(0)

if os.environ.get("PROBE_OPT") == "1":
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(2)
        .set_compute_cfg_math_fidelity(forge._C.MathFidelity.HiFi2)
        # remove_dead_values is left OFF: it crashes the compiler on this graph with
        # an upstream MLIR assertion (RemoveDeadValues.cpp:318, processFuncOp). That
        # is a compiler bug, not a Quasar one -- the rest of the optimised pipeline
        # is what matters here, because it is what lowers the global average pool to
        # mean(dim=[-2]) instead of the unported W reduce.
        .set_max_legal_layouts(8)
    )
    cfg = CompilerConfig(mlir_config=mlir_config)
    cfg.enable_optimization_passes = True
    print("[rn50] pipeline: OPTIMISED", flush=True)
else:
    cfg = CompilerConfig()
    print("[rn50] pipeline: DEFAULT", flush=True)
cfg.default_df_override = forge._C.DataFormat.Float16_b

model = torchvision.models.resnet50(weights=None).eval()
inputs = [torch.rand(1, 3, SIZE, SIZE)]
print(f"[rn50] input {tuple(inputs[0].shape)}", flush=True)

tmp = tempfile.mkdtemp(prefix="rn50_qsr_")
onnx_path = os.path.join(tmp, "resnet50.onnx")
torch.onnx.export(model, inputs[0], onnx_path, opset_version=17,
                  input_names=["a"], output_names=["o"])
import onnx as _onnx
onnx_model = _onnx.load(onnx_path)
print("[rn50] onnx exported", flush=True)

try:
    c = forge.compile(forge.OnnxModule("rn50_qsr", onnx_model), inputs,
                      module_name="rn50_qsr", compiler_cfg=cfg)
except Exception as e:
    print(f"[rn50] RESULT: COMPILE_FAIL {type(e).__name__}: {str(e)[:400]}", flush=True)
    raise SystemExit(0)

src = json.loads(c.compiled_binary.as_json()).get("mlir", {}).get("source", "")
ops = {}
for l in src.splitlines():
    for t in l.split('"'):
        if t.startswith("ttnn."):
            ops[t] = ops.get(t, 0) + 1
print("[rn50] ttnn ops: " + ", ".join(f"{k}={v}" for k, v in sorted(ops.items())), flush=True)
with open("/proj_sw/user_dev/ctr-lelanchelian/tt-forge-onnx/add_rs/ir/rn50_qsr.mlir", "w") as f:
    f.write(src)
print("[rn50] compile OK", flush=True)

try:
    out = c(*inputs)
except Exception as e:
    kind = "QUASAR_REFUSED" if "not supported on Quasar" in str(e) else "RUN_FAIL"
    print(f"[rn50] RESULT: {kind} {type(e).__name__}", flush=True)
    print("[rn50] FULL EXCEPTION >>>", flush=True)
    print(str(e)[:6000], flush=True)
    print("[rn50] <<< END EXCEPTION", flush=True)
    raise SystemExit(0)

got = (out[0] if isinstance(out, (list, tuple)) else out).to(torch.float32).cpu().flatten()
with torch.no_grad():
    g = model(inputs[0]).to(torch.float32).flatten()
pcc = torch.corrcoef(torch.stack([g, got]))[0, 1].item()
denom = g.abs().clamp(min=1e-3)
max_rel = ((g - got).abs() / denom).max().item()
ok = (pcc > 0.99) and (max_rel < 0.1)
print(f"[rn50] RESULT: {'PASS' if ok else 'NUMERIC_FAIL'} pcc={pcc:.6f} "
      f"max_abs_err={(g - got).abs().max().item():.5f} max_rel_err={max_rel:.5f}", flush=True)
