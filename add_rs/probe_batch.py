"""Run SEVERAL probe cases in ONE process.

The emulator pays a fixed 38-66 s ZeBu model load per process, so one-op-per-process
wastes most of a run on loading. This reuses probe_exec_one_op.py's case table (parsed
out, so that file stays the single source of shapes) and runs every requested op against
one device session.

Argv: op names. One RESULT line per op; a failure in one op does not stop the rest.
"""
import ast, os, sys, json, traceback
import numpy as np
import torch
from onnx import TensorProto, helper, numpy_helper
import forge
from forge.config import CompilerConfig

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_exec_one_op.py")
_tree = ast.parse(open(SRC).read())
_ns = {"np": np, "torch": torch, "TensorProto": TensorProto, "helper": helper,
       "numpy_helper": numpy_helper, "forge": forge}
for _node in _tree.body:
    if isinstance(_node, ast.FunctionDef) and _node.name in {"vi", "model", "case"}:
        exec(compile(ast.Module(body=[_node], type_ignores=[]), SRC, "exec"), _ns)
case = _ns["case"]

torch.manual_seed(0)
cfg = CompilerConfig()
cfg.default_df_override = forge._C.DataFormat.Float16_b

ops = sys.argv[1:]
print(f"[batch] {len(ops)} ops in one device session: {', '.join(ops)}", flush=True)

for op in ops:
    try:
        m, inputs, ref = case(op)
    except Exception as e:
        print(f"[{op}] RESULT: NO_CASE {type(e).__name__}", flush=True)
        continue
    try:
        c = forge.compile(m, inputs, module_name=f"b_{op}", compiler_cfg=cfg)
    except Exception as e:
        print(f"[{op}] RESULT: COMPILE_FAIL {type(e).__name__}: {str(e)[:160]}", flush=True)
        continue
    try:
        out = c(*inputs)
    except Exception as e:
        kind = "QUASAR_REFUSED" if "not supported on Quasar" in str(e) else "RUN_FAIL"
        print(f"[{op}] RESULT: {kind} {type(e).__name__}: {str(e)[:160]}", flush=True)
        continue
    try:
        got = (out[0] if isinstance(out, (list, tuple)) else out).to(torch.float32).cpu().flatten()
        g = ref(*inputs).to(torch.float32).flatten()
        if got.numel() != g.numel():
            print(f"[{op}] RESULT: SHAPE_MISMATCH got {got.numel()} want {g.numel()}", flush=True)
            continue
        pcc = torch.corrcoef(torch.stack([g, got]))[0, 1].item()
        max_abs = (g - got).abs().max().item()
        # Normalised by the tensor's own scale: per-element relative error is
        # meaningless once an op emits exact zeros (any relu).
        norm = max_abs / max(g.abs().max().item(), 1e-6)
        ok = (pcc > 0.999) and (norm < 0.02)
        print(f"[{op}] RESULT: {'PASS' if ok else 'NUMERIC_FAIL'} pcc={pcc:.6f} "
              f"max_abs_err={max_abs:.5f} err/scale={norm:.5f}", flush=True)
    except Exception as e:
        print(f"[{op}] RESULT: COMPARE_FAIL {type(e).__name__}: {str(e)[:160]}", flush=True)

print("[batch] done", flush=True)
