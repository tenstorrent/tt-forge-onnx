"""Run a PREFIX of ResNet-50 on Quasar, to find where the full model diverges.

Argv: prefix name -- stem | layer1 | layer2 | layer3 | layer4
Each prefix is the real torchvision module chain up to that point, so the first
prefix that fails names the stage that breaks. Reference is the same torch
modules in fp32; torch in pure bf16 scores 0.999959 on the full model, so bf16 is
not a plausible explanation for a failure here.
"""
import os, sys, json, tempfile
import numpy as np
import torch, torch.nn as nn
import torchvision
import forge
from forge.config import CompilerConfig

PREFIX = sys.argv[1]
SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 32
torch.manual_seed(0)

m = torchvision.models.resnet50(weights=None).eval()
if PREFIX == "idblock":
    # One IDENTITY-skip bottleneck on its own: layer1[1] takes 256 channels in and
    # out, so its skip is the input tensor itself with no projection convolution.
    # That is the structure the projection-skip probe cannot cover, and it keeps
    # the block input live across the whole block.
    net = nn.Sequential(m.layer1[1]).eval()
    inputs = [torch.rand(1, 256, SIZE, SIZE)]
elif PREFIX == "l3idblock":
    # ONE identity-skip bottleneck at layer3's dimensions: 1024 channels in and
    # out, 2x2 spatial at SIZE=32. layer3's block 0 (the projection block) is
    # correct at 0.999888 and block 1 is where the model falls to 0.951957, so
    # this is the smallest unit that reproduces the drop.
    net = nn.Sequential(m.layer3[1]).eval()
    inputs = [torch.rand(1, 1024, max(SIZE // 16, 1), max(SIZE // 16, 1))]
elif PREFIX.startswith("l3chain"):
    # The layer3 identity block's convolution chain WITHOUT the residual add:
    # l3chain1 = conv1+bn1+relu, l3chain2 = +conv2+bn2+relu, l3chain3 = +conv3+bn3.
    # Every ingredient passes on its own, so this splits "chaining the convolutions
    # is wrong" from "the skip that is held across the block is wrong".
    _b = m.layer3[1]
    _n = int(PREFIX[len("l3chain"):])
    _parts = [_b.conv1, _b.bn1, _b.relu]
    if _n >= 2:
        _parts += [_b.conv2, _b.bn2, _b.relu]
    if _n >= 3:
        _parts += [_b.conv3, _b.bn3]
    net = nn.Sequential(*_parts).eval()
    inputs = [torch.rand(1, 1024, max(SIZE // 16, 1), max(SIZE // 16, 1))]
elif PREFIX == "l3idblock2":
    net = nn.Sequential(m.layer3[1], m.layer3[2]).eval()
    inputs = [torch.rand(1, 1024, max(SIZE // 16, 1), max(SIZE // 16, 1))]
elif PREFIX == "idblock2":
    # Two identity blocks chained, as layer1[1:3] is in the real model.
    net = nn.Sequential(m.layer1[1], m.layer1[2]).eval()
    inputs = [torch.rand(1, 256, SIZE, SIZE)]
else:
    net = None

chain = [m.conv1, m.bn1, m.relu, m.maxpool]
import re as _re
_mb = _re.fullmatch(r"l(\d)b(\d+)", PREFIX or "")
if _mb:
    # stem + every earlier stage + the first N+1 bottleneck blocks of stage S, to
    # bisect inside a stage. layer3 is 6 blocks, so l3b0..l3b5, and l3b5 == layer3.
    _stage, _n = int(_mb.group(1)), int(_mb.group(2))
    for _s in range(1, _stage):
        chain.append(getattr(m, f"layer{_s}"))
    chain.extend(list(getattr(m, f"layer{_stage}"))[: _n + 1])
else:
    for name in ["layer1", "layer2", "layer3", "layer4"]:
        if PREFIX == "stem":
            break
        chain.append(getattr(m, name))
        if PREFIX == name:
            break
if net is None:
    net = nn.Sequential(*chain).eval()
    inputs = [torch.rand(1, 3, SIZE, SIZE)]
with torch.no_grad():
    ref = net(inputs[0]).float().flatten()
print(f"[{PREFIX}] input {tuple(inputs[0].shape)} ref |max|={ref.abs().max().item():.4f} "
      f"n={ref.numel()}", flush=True)

tmp = tempfile.mkdtemp(prefix=f"rn50_{PREFIX}_")
onnx_path = os.path.join(tmp, "m.onnx")
torch.onnx.export(net, inputs[0], onnx_path, opset_version=17,
                  input_names=["a"], output_names=["o"])
import onnx as _onnx
onnx_model = _onnx.load(onnx_path)

cfg = CompilerConfig()
cfg.default_df_override = forge._C.DataFormat.Float16_b
try:
    c = forge.compile(forge.OnnxModule(f"rn50_{PREFIX}", onnx_model), inputs,
                      module_name=f"rn50_{PREFIX}", compiler_cfg=cfg)
except Exception as e:
    print(f"[{PREFIX}] RESULT: COMPILE_FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)
    raise SystemExit(0)

src = json.loads(c.compiled_binary.as_json()).get("mlir", {}).get("source", "")
ops = {}
for l in src.splitlines():
    for t in l.split('"'):
        if t.startswith("ttnn."):
            ops[t] = ops.get(t, 0) + 1
print(f"[{PREFIX}] ttnn ops: " + ", ".join(f"{k}={v}" for k, v in sorted(ops.items())), flush=True)

try:
    out = c(*inputs)
except Exception as e:
    kind = "QUASAR_REFUSED" if "not supported on Quasar" in str(e) else "RUN_FAIL"
    print(f"[{PREFIX}] RESULT: {kind} {type(e).__name__}: {str(e)[:300]}", flush=True)
    raise SystemExit(0)

got = (out[0] if isinstance(out, (list, tuple)) else out).to(torch.float32).cpu().flatten()
if got.numel() != ref.numel():
    print(f"[{PREFIX}] RESULT: SHAPE_MISMATCH got {got.numel()} want {ref.numel()}", flush=True)
    raise SystemExit(0)
pcc = torch.corrcoef(torch.stack([ref, got]))[0, 1].item()
# Per-element relative error is useless after a relu: most of the tensor is
# exactly 0, and any tiny absolute error divided by the 1e-3 floor looks huge.
# Normalise by the tensor's own scale instead.
max_abs = (ref - got).abs().max().item()
norm_err = max_abs / max(ref.abs().max().item(), 1e-6)
ok = (pcc > 0.999) and (norm_err < 0.02)
print(f"[{PREFIX}] RESULT: {'PASS' if ok else 'NUMERIC_FAIL'} pcc={pcc:.6f} "
      f"max_abs_err={max_abs:.5f} err/scale={norm_err:.5f} "
      f"got|max|={got.abs().max().item():.4f} ref|max|={ref.abs().max().item():.4f}", flush=True)
# Is the device output saturating? A max that lands on an exact power of two, with
# many elements sitting on it, is a clamp rather than arithmetic drift.
# Is the device systematically attenuating? Fit got = a*ref + b; a well below 1
# means large values are being under-produced, which random rounding does not do.
_a = (torch.dot(ref - ref.mean(), got - got.mean()) /
      torch.clamp(torch.dot(ref - ref.mean(), ref - ref.mean()), min=1e-12)).item()
_big = ref.abs() > 0.5 * ref.abs().max()
_ratio = (got[_big].abs().sum() / torch.clamp(ref[_big].abs().sum(), min=1e-12)).item()
print(f"[{PREFIX}] FIT slope={_a:.6f}  mean|got|/|ref| over top-half values={_ratio:.6f} "
      f"(n={int(_big.sum())})", flush=True)
gmax = got.abs().max().item()
at_max = int((got.abs() >= gmax * (1 - 1e-6)).sum().item())
print(f"[{PREFIX}] SAT: elements at |max| = {at_max}/{got.numel()} "
      f"({100.0*at_max/got.numel():.3f}%)  ref elements above got|max| = "
      f"{int((ref.abs() > gmax).sum().item())}", flush=True)
