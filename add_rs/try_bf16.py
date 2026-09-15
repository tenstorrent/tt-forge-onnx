"""Does default_df_override actually change the emitted TTNN tensor dtype?"""
import json, os, sys
import torch
from onnx import TensorProto, helper
import forge
from forge.config import CompilerConfig, MLIRConfig

SHAPE = [2, 32, 32]
def model():
    n = helper.make_node("Add", inputs=["a", "b"], outputs=["o"])
    g = helper.make_graph([n], "G",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, SHAPE),
         helper.make_tensor_value_info("b", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("o", TensorProto.FLOAT, SHAPE)])
    return helper.make_model(g, producer_name="t",
                             opset_imports=[helper.make_operatorsetid("", 21)])

for label, df in (("baseline_f32", None), ("df_override_bf16", forge._C.DataFormat.Float16_b)):
    cfg = CompilerConfig(mlir_config=MLIRConfig().set_target_arch(forge._C.Arch.QUASAR))
    if df is not None:
        cfg.default_df_override = df
    c = forge.compile(model(), [torch.rand(SHAPE), torch.rand(SHAPE)],
                      module_name=f"bf16try_{label}", compiler_cfg=cfg)
    src = json.loads(c.compiled_binary.as_json()).get("mlir", {}).get("source", "")
    dtypes = sorted(set(t for t in ("f32", "bf16", "f16") if f"x{t}," in src or f"x{t}>" in src))
    add_line = [l.strip()[:100] for l in src.splitlines() if '"ttnn.add"' in l]
    print(f"\n=== {label}: dtypes in IR = {dtypes}")
    print(f"    {add_line[0] if add_line else '(no ttnn.add)'}")
