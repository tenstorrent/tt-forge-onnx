"""Print the live system descriptor of whatever Quasar device is attached, and
diff it against the constants tt-mlir PR #9287 hardcodes into
createDefaultQuasarSystemDesc (lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp).

Compiles a two-element Add purely to make the compiler emit a descriptor; the
graph is never run, so this costs one model load and no program launches.
"""
import json, re, sys
import numpy as np
import torch
from onnx import TensorProto, helper
import forge
from forge.config import CompilerConfig

LABEL = sys.argv[1] if len(sys.argv) > 1 else "device"

# What the PR claims. Keys are the field names as they appear in #system_desc.
PR_MOCK = {
    "grid": "4x8",
    "coord_translation_offsets": "2x2",
    "l1_size": 4194304,
    "num_dram_channels": 2,
    "dram_channel_size": 1073741824,
    "noc_l1_address_align_bytes": 16,
    "pcie_address_align_bytes": 64,
    "noc_dram_address_align_bytes": 64,
    "l1_unreserved_base": 313088,
    "erisc_l1_unreserved_base": 88576,
    "dram_unreserved_base": 1048704,
    "dram_unreserved_end": 1068732416,
    "dst_physical_size_tiles": 16,
    "num_cbs": 64,
    "num_compute_threads": 4,
    "num_datamovement_threads": 6,
    "dram_grid": "1x2",
}
PR_DTYPES = "[<f32>, <f16>, <bf16>, <u8>, <si32>]"

vi = lambda n, s: helper.make_tensor_value_info(n, TensorProto.FLOAT, s)
m = helper.make_model(
    helper.make_graph([helper.make_node("Add", ["a", "b"], ["o"])], "G",
                      [vi("a", [1, 32]), vi("b", [1, 32])], [vi("o", [1, 32])]),
    producer_name="P", opset_imports=[helper.make_operatorsetid("", 21)])

cfg = CompilerConfig()
cfg.default_df_override = forge._C.DataFormat.Float16_b
c = forge.compile(m, [torch.rand(1, 32), torch.rand(1, 32)],
                  module_name="sysdesc_probe", compiler_cfg=cfg)
src = json.loads(c.compiled_binary.as_json()).get("mlir", {}).get("source", "")
line = next((l for l in src.splitlines() if "#system_desc" in l and "arch =" in l), "")
if not line:
    print(f"[{LABEL}] NO SYSTEM DESC FOUND", flush=True)
    raise SystemExit(1)

def scalar(name):
    mm = re.search(rf"\b{name} = (-?\d+)", line)
    return int(mm.group(1)) if mm else None

def gridlike(name):
    mm = re.search(rf"\b{name} = (\d+x\d+)", line)
    return mm.group(1) if mm else None

dtypes = re.search(r"supported_data_types = (\[[^\]]*\])", line)
tiles = re.search(r"supported_tile_sizes = (\[[^\]]*\])", line)
arch = re.search(r"arch = <(\w+)>", line)

print(f"[{LABEL}] arch = {arch.group(1) if arch else '?'}", flush=True)
bad = 0
for k, want in PR_MOCK.items():
    got = gridlike(k) if isinstance(want, str) else scalar(k)
    ok = (got == want)
    bad += 0 if ok else 1
    print(f"[{LABEL}] {'OK  ' if ok else 'DIFF'} {k:32s} device={got!s:14s} pr_mock={want}", flush=True)

got_dt = dtypes.group(1) if dtypes else None
ok = (got_dt == PR_DTYPES)
bad += 0 if ok else 1
print(f"[{LABEL}] {'OK  ' if ok else 'DIFF'} supported_data_types            device={got_dt}", flush=True)
print(f"[{LABEL}] ---- supported_tile_sizes = {tiles.group(1) if tiles else '?'}", flush=True)
print(f"[{LABEL}] MISMATCHES: {bad}", flush=True)
