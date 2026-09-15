"""Golden (fp32) vs the same model on CPU in the device's precision (bf16).

The device runs ResNet-50 in bfloat16. Comparing its output against an fp32 torch
golden therefore measures two things at once: whether the device is correct, and
what bf16 costs. This isolates the second, on CPU, so the device's residual can be
judged against the floor rather than against zero.
"""
import sys, torch, torchvision, torch.nn as nn

SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 32
torch.manual_seed(0)
m = torchvision.models.resnet50(weights=None).eval()

def chain(prefix):
    c = [m.conv1, m.bn1, m.relu, m.maxpool]
    for name in ["layer1", "layer2", "layer3", "layer4"]:
        if prefix == "stem":
            break
        c.append(getattr(m, name))
        if prefix == name:
            break
    return nn.Sequential(*c).eval()

def stats(ref, got):
    pcc = torch.corrcoef(torch.stack([ref, got]))[0, 1].item()
    mx = (ref - got).abs().max().item()
    return pcc, mx, mx / max(ref.abs().max().item(), 1e-6)

print(f"{'prefix':8s} {'bf16-CPU pcc':>13s} {'err/scale':>10s} {'|max| fp32':>11s} {'|max| bf16':>11s}")
for p in ["stem", "layer1", "layer2", "layer3", "layer4"]:
    net = chain(p)
    x = torch.rand(1, 3, SIZE, SIZE)
    with torch.no_grad():
        ref = net(x).float().flatten()
        gb = net.bfloat16()(x.bfloat16()).float().flatten()
        net.float()
    pcc, mx, es = stats(ref, gb)
    print(f"{p:8s} {pcc:13.6f} {es:10.5f} {ref.abs().max().item():11.4f} {gb.abs().max().item():11.4f}")

# the whole model
x = torch.rand(1, 3, SIZE, SIZE)
with torch.no_grad():
    ref = m(x).float().flatten()
    gb = m.bfloat16()(x.bfloat16()).float().flatten()
pcc, mx, es = stats(ref, gb)
print(f"{'FULL':8s} {pcc:13.6f} {es:10.5f} {ref.abs().max().item():11.4f} {gb.abs().max().item():11.4f}")
