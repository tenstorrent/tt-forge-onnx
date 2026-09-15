"""Classify a real image with pretrained ResNet-50 on CPU, in fp32 and in bf16.

Prints the top-5 and the raw logits, so a device run can be compared against both
the fp32 golden and the bf16 floor rather than against fp32 alone.
"""
import sys, torch, torchvision
from torchvision.models import ResNet50_Weights
from PIL import Image

img_path = sys.argv[1] if len(sys.argv) > 1 else "image.png"
SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 224

weights = ResNet50_Weights.IMAGENET1K_V1
cats = weights.meta["categories"]
m = torchvision.models.resnet50(weights=weights).eval()

img = Image.open(img_path).convert("RGB")
print(f"image: {img_path}  original size {img.size}")
tf = torchvision.transforms.Compose([
    torchvision.transforms.Resize(int(SIZE * 256 / 224)),
    torchvision.transforms.CenterCrop(SIZE),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
x = tf(img).unsqueeze(0)
torch.save(x, f"add_rs/image_input_{SIZE}.pt")
print(f"input tensor {tuple(x.shape)} saved to add_rs/image_input_{SIZE}.pt")

with torch.no_grad():
    fp32 = m(x).float()
    bf16 = m.bfloat16()(x.bfloat16()).float()
    m.float()
torch.save(fp32, f"add_rs/image_logits_fp32_{SIZE}.pt")

def top5(logits, label):
    p = torch.softmax(logits, dim=1)[0]
    v, i = p.topk(5)
    print(f"  {label}:")
    for r in range(5):
        print(f"    {r+1}. {cats[i[r]]:<28s} {v[r].item()*100:6.2f}%   (class {i[r].item()})")

top5(fp32, "fp32 CPU (golden)")
top5(bf16, "bf16 CPU (device precision floor)")
pcc = torch.corrcoef(torch.stack([fp32.flatten(), bf16.flatten()]))[0, 1].item()
print(f"  bf16 vs fp32 logits: pcc={pcc:.6f}  argmax same={bool(fp32.argmax()==bf16.argmax())}")
