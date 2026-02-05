# # SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# #
# # SPDX-License-Identifier: Apache-2.0

# import pytest
# import torch
# from third_party.tt_forge_models.yolov10.pytorch import ModelLoader, ModelVariant

# import forge
# from forge._C import DataFormat
# from forge.config import CompilerConfig
# from forge.forge_property_utils import (
#     Framework,
#     ModelArch,
#     ModelGroup,
#     ModelPriority,
#     Source,
#     Task,
#     record_model_properties,
# )
# from forge.verify.verify import verify

# from test.models.pytorch.vision.yolo.model_utils.yolo_utils import YoloWrapper

# variants = [
#     ModelVariant.YOLOV10X,
#     ModelVariant.YOLOV10N,
# ]


# @pytest.mark.nightly
# @pytest.mark.parametrize("variant", variants)
# def test_yolov10(variant):
#     # Set group and priority based on variant
#     if variant == ModelVariant.YOLOV10X:
#         group = ModelGroup.RED
#         priority = ModelPriority.P1
#     else:
#         group = ModelGroup.GENERALITY
#         priority = ModelPriority.P2

#     # Record Forge Property
#     module_name = record_model_properties(
#         framework=Framework.PYTORCH,
#         model=ModelArch.YOLOV10,
#         variant=variant,
#         task=Task.CV_OBJECT_DETECTION,
#         source=Source.GITHUB,
#         group=group,
#         priority=priority,
#     )

#     # Load model and inputs
#     loader = ModelLoader(variant=variant)
#     model = loader.load_model(dtype_override=torch.bfloat16)
#     framework_model = YoloWrapper(model)
#     input_tensor = loader.load_inputs(dtype_override=torch.bfloat16)
#     inputs = [input_tensor]

#     data_format_override = DataFormat.Float16_b
#     compiler_cfg = CompilerConfig(default_df_override=data_format_override)

#     # Forge compile framework model
#     compiled_model = forge.compile(
#         framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
#     )

#     # Model Verification
#     verify(inputs, framework_model, compiled_model)
