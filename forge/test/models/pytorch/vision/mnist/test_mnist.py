# # SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# # SPDX-License-Identifier: Apache-2.0
# import pytest
# import torch
# from third_party.tt_forge_models.mnist.pytorch import ModelLoader

# import forge
# from forge._C import DataFormat
# from forge.config import CompilerConfig
# from forge.forge_property_utils import (
#     Framework,
#     ModelArch,
#     Source,
#     Task,
#     record_model_properties,
# )
# from forge.verify.config import VerifyConfig
# from forge.verify.value_checkers import AutomaticValueChecker
# from forge.verify.verify import verify


# @pytest.mark.nightly
# def test_mnist():

#     # Record Forge Property
#     module_name = record_model_properties(
#         framework=Framework.PYTORCH,
#         model=ModelArch.MNIST,
#         source=Source.GITHUB,
#         task=Task.CV_IMAGE_CLASSIFICATION,
#     )

#     # Load model and input
#     loader = ModelLoader()
#     framework_model = loader.load_model(dtype_override=torch.bfloat16)
#     input_tensor = loader.load_inputs(dtype_override=torch.bfloat16)
#     inputs = [input_tensor]

#     data_format_override = DataFormat.Float16_b
#     compiler_cfg = CompilerConfig(default_df_override=data_format_override)

#     # Forge compile framework model
#     compiled_model = forge.compile(
#         framework_model,
#         sample_inputs=inputs,
#         module_name=module_name,
#         compiler_cfg=compiler_cfg,
#     )

#     # Model Verification and Inference
#     _, co_out = verify(
#         inputs,
#         framework_model,
#         compiled_model,
#         verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.97)),
#     )

#     # Post Processing
#     loader.print_cls_results(co_out)
