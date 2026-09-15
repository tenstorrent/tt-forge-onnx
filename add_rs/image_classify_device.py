"""Run a real image through pretrained ResNet-50 on the attached Quasar device and
compare against the CPU fp32 golden and the CPU bf16 floor.

Argv: image path, input size. The image is preprocessed exactly as the CPU script
does, so the two are comparing the same tensor.
"""
import os, sys, json, tempfile, time
import torch, torchvision
from torchvision.models import ResNet50_Weights
from PIL import Image
import forge
from forge.config import CompilerConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
IMG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "testimg.jpg")
SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 224
torch.manual_seed(0)

weights = ResNet50_Weights.IMAGENET1K_V1
cats = weights.meta["categories"]
model = torchvision.models.resnet50(weights=weights).eval()

tf = torchvision.transforms.Compose([
    torchvision.transforms.Resize(int(SIZE * 256 / 224)),
    torchvision.transforms.CenterCrop(SIZE),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
x = tf(Image.open(IMG).convert("RGB")).unsqueeze(0)
inputs = [x]
with torch.no_grad():
    golden = model(x).float()
    bf16 = model.bfloat16()(x.bfloat16()).float()
    model.float()
print(f"[img] {IMG} -> input {tuple(x.shape)}", flush=True)

cfg = CompilerConfig()
cfg.default_df_override = forge._C.DataFormat.Float16_b

tmp = tempfile.mkdtemp(prefix="rn50_img_")
onnx_path = os.path.join(tmp, "resnet50.onnx")
torch.onnx.export(model, x, onnx_path, opset_version=17,
                  input_names=["a"], output_names=["o"])
import onnx as _onnx
onnx_model = _onnx.load(onnx_path)
print("[img] onnx exported", flush=True)

t0 = time.time()
try:
    c = forge.compile(forge.OnnxModule("rn50_img", onnx_model), inputs,
                      module_name="rn50_img", compiler_cfg=cfg)
except Exception as e:
    print(f"[img] RESULT: COMPILE_FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)
    raise SystemExit(0)
print(f"[img] compiled in {time.time()-t0:.1f}s", flush=True)

t1 = time.time()
out = c(*inputs)
dev = (out[0] if isinstance(out, (list, tuple)) else out).float().reshape(1, -1)
print(f"[img] device run in {time.time()-t1:.1f}s", flush=True)

def top5(logits, label):
    p = torch.softmax(logits, dim=1)[0]
    v, i = p.topk(5)
    print(f"[img] {label}:", flush=True)
    for r in range(5):
        print(f"[img]     {r+1}. {cats[i[r]]:<26s} {v[r].item()*100:6.2f}%  (class {i[r].item()})", flush=True)
    return i[0].item()

a = top5(golden, "fp32 CPU (golden)")
b = top5(bf16,   "bf16 CPU (floor)")
d = top5(dev,    "DEVICE")
pcc = torch.corrcoef(torch.stack([golden.flatten(), dev.flatten()]))[0, 1].item()
pccb = torch.corrcoef(torch.stack([bf16.flatten(), dev.flatten()]))[0, 1].item()
print(f"[img] RESULT: device_vs_golden_pcc={pcc:.6f} device_vs_bf16cpu_pcc={pccb:.6f} "
      f"top1_golden={a} top1_device={d} MATCH={a==d}", flush=True)
