# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify, DeprecatedVerifyConfig
import math
import onnx
from torchvision.models.detection import _utils as det_utils
from test.models.onnx.vision.vision_utils.utils import load_vision_model_and_input
from transformers import BartForSequenceClassification


@pytest.mark.skip_model_analysis
@pytest.mark.xfail(
    reason="AssertionError: Data mismatch on output 0 between framework and Forge codegen, PCC got=0.4923030518607919"
)  # https://github.com/tenstorrent/tt-forge-onnx/issues/1793
@pytest.mark.nightly
def test_inplace_updation():
    class Inplace_updation(nn.Module):
        def __init__(self):
            super().__init__()
            self.shift_size = 4
            self.window_size = 8

        def forward(self, x):

            img_mask = torch.zeros((1, 64, 64, 1), dtype=torch.float32)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            return img_mask + x

    inputs = [torch.randn((1, 64, 64, 1), dtype=torch.float32)]
    model = Inplace_updation()
    model.eval()

    compiled_model = forge.compile(
        model,
        sample_inputs=inputs,
    )
    verify(inputs, model, compiled_model)


@pytest.mark.parametrize(
    "input_shape, clamp_min, clamp_max, dtype",
    [
        ((64, 3, 64, 64), None, math.log(100.0), torch.float32),
        ((11, 30, 11, 11), math.log(101.0), None, torch.float32),
        ((27, 16, 27, 27), -math.log(103.0), None, torch.float32),
        ((45, 2, 45, 45), None, 5.0, torch.float32),
        ((3, 21, 3, 3), None, math.log(50.0), torch.float32),
        ((12, 6, 12, 12), -103.0, None, torch.float32),
        ((18, 11, 18, 18), -50, None, torch.int32),
        ((8, 1, 8, 8), None, 876, torch.int32),
    ],
)
@pytest.mark.skip_model_analysis
@pytest.mark.push
@pytest.mark.nightly
def test_clamp(input_shape, clamp_min, clamp_max, dtype):
    class Clamp(nn.Module):
        def __init__(self):
            super().__init__()
            log_value = torch.log(10 * torch.ones(input_shape[1], 1, 1))
            self.logit_scale = nn.Parameter(log_value)

        def forward(self, attn):
            clamped = torch.clamp(self.logit_scale, min=clamp_min, max=clamp_max)
            logit_scale = clamped.exp()
            return attn * logit_scale

    model = Clamp()
    model.eval()

    if dtype == torch.float32:
        attn = torch.randn(input_shape)
    elif dtype == torch.int32:
        attn = torch.randint(low=torch.iinfo(dtype).min, high=torch.iinfo(dtype).max, size=input_shape, dtype=dtype)

    inputs = [attn]

    # Export to ONNX
    torch.onnx.export(model, (inputs[0],), "temp.onnx", opset_version=17)

    # Load and check ONNX
    onnx_model = onnx.load("temp.onnx")
    onnx.checker.check_model(onnx_model)

    compiled_model = forge.compile(onnx_model, inputs)
    verify(inputs, model, compiled_model)


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
def test_rotary_pos_emb():
    class RotaryPosEmb(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, q, cos, sin):
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

            rotary_dim = cos.shape[-1]
            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]

            q_embed = torch.cat([q_pass, (q_rot * cos) + (self.rotate_half(q_rot) * sin)], dim=-1)
            return q_embed

        def rotate_half(self, x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

    framework_model = RotaryPosEmb()
    framework_model.eval()

    query = torch.rand((1, 32, 256, 96))
    cos_emb = torch.rand([1, 256, 96])
    sin_emb = torch.rand([1, 256, 96])
    inputs = [query, cos_emb, sin_emb]

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    verify(inputs, framework_model, compiled_model)


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
@pytest.mark.parametrize(
    "dim",
    [-4, -3, -2, -1, 0, 1, 2, 3],
)
def test_remove_concat_pass(dim):
    class SliceConcat(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, q):
            sl = [slice(None)] * len(q.shape)
            sl[self.dim] = slice(q.shape[self.dim], None)
            q_pass = q[tuple(sl)]
            q_embed = torch.cat([q_pass, (q * q)], dim=self.dim)
            return q_embed

    model = SliceConcat(dim)

    query = torch.rand((1, 32, 256, 96))
    inputs = [query]

    compiled_model = forge.compile(model, sample_inputs=inputs)
    verify(inputs, model, compiled_model)


@pytest.mark.skip_model_analysis
@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize(
    "input_shape, flip_dim",
    [
        ((1, 1, 1024, 72), 0),
        ((1, 1, 104, 72), 1),
        ((1, 12, 1, 92), 2),
        ((1, 33, 11, 1), 3),
        ((1, 1, 17, 16), 1),
    ],
)
def test_flip(input_shape, flip_dim):
    class Flip(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return x.flip(dims=(self.dim,))

    torch_model = Flip(dim=flip_dim)
    inputs = [torch.rand(*input_shape)]

    # Export model to ONNX
    onnx_path = "flip.onnx"
    torch.onnx.export(torch_model, (inputs[0],), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule("sanity", onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs)

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
    )


@pytest.mark.skip_model_analysis
@pytest.mark.push
@pytest.mark.nightly
def test_stack_and_reshape_onnx():
    class stack_reshape(nn.Module):
        def __init__(self):
            super().__init__()
            self.boxes_shape = (3234, 4)

        def forward(self, boxes_x, boxes_y):
            clipped_boxes = torch.stack((boxes_x, boxes_y), dim=2)
            return clipped_boxes.reshape(self.boxes_shape)

    torch_model = stack_reshape()
    x = torch.randn(3234, 2)
    y = torch.randn(3234, 2)
    inputs = [x, y]

    onnx_path = "stack_reshape.onnx"
    torch.onnx.export(
        torch_model,
        (inputs[0], inputs[1]),
        onnx_path,
        opset_version=17,
    )

    module_name = "stack_reshape"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    verify(inputs, framework_model, compiled_model)


variants_with_weights = {
    "ssdlite320_mobilenet_v3_large": "SSDLite320_MobileNet_V3_Large_Weights",
}


@pytest.mark.xfail
@pytest.mark.nightly
@pytest.mark.skip_model_analysis
@pytest.mark.parametrize("variant", variants_with_weights.keys())
def test_ssdlite320_mobilenet_v3_large_problematic_block(variant):

    pytest.xfail(reason="Fatal Python error: Segmentation fault")

    class postprocess_detections(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.image_sizes = [(320, 320)]

        def forward(self, input1, input2, input3):

            head_outputs = {"bbox_regression": input1, "cls_logits": input2}
            anchors = [input3]

            op = self.model.postprocess_detections(head_outputs, anchors, self.image_sizes)
            return op

    # Load model
    weight_name = variants_with_weights[variant]
    framework_model, _ = load_vision_model_and_input(variant, "detection", weight_name)
    framework_model = postprocess_detections(framework_model)

    bbox_regression = torch.randn(1, 3234, 4)
    cls_logits = torch.randn(1, 3234, 91)
    anchors = torch.randn(3234, 4)
    inputs = [bbox_regression, cls_logits, anchors]

    onnx_path = "problematic_block.onnx"
    torch.onnx.export(
        framework_model,
        (inputs[0], inputs[1], inputs[2]),
        onnx_path,
        opset_version=17,
    )

    module_name = "problematic_block"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )
    verify(inputs, framework_model, compiled_model)


@pytest.mark.skip_model_analysis
@pytest.mark.push
@pytest.mark.xfail
@pytest.mark.nightly
def test_gather_to_take_onnx():
    class take(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, boxes, score):

            keep_idxs = score > 0.001
            score = score[keep_idxs]
            box = boxes[keep_idxs]
            _, idxs = score.topk(300)
            box = box[idxs]

            return box

    torch_model = take()
    boxes = torch.rand(3234, 4)
    scores = torch.rand(3234)
    inputs = [boxes, scores]

    onnx_path = "take.onnx"
    torch.onnx.export(
        torch_model,
        (inputs[0], inputs[1]),
        onnx_path,
        opset_version=17,
        verbose=True,
    )

    module_name = "gather_to_take_onnx"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    framework_model = forge.OnnxModule(module_name, onnx_model)
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    verify(inputs, framework_model, compiled_model)


@pytest.mark.skip_model_analysis
@pytest.mark.xfail
@pytest.mark.push
@pytest.mark.nightly
def test_concat_block():
    class concat(nn.Module):
        def __init__(self):
            super().__init__()
            self.score_thresh = 0.001
            self.topk_candidates = 300

        def forward(self, score):

            keep_idxs = score > self.score_thresh
            score = score[keep_idxs]
            num_topk = det_utils._topk_min(score, self.topk_candidates, 0)

            return num_topk

    torch_model = concat()
    scores = torch.rand(3234)
    inputs = [scores]

    onnx_path = "concat.onnx"
    torch.onnx.export(
        torch_model,
        (inputs[0]),
        onnx_path,
        opset_version=17,
        verbose=True,
    )

    module_name = "concat"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.skip_model_analysis
@pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-onnx/issues/2899")
@pytest.mark.parametrize(
    "tensor_size,max_length",
    [
        (24, 160),
        (10, 50),
        (32, 100),
        (16, 200),
        (8, 5),
    ],
)
def test_index_put_speecht5_tts(tensor_size, max_length):
    class index_put(nn.Module):
        def __init__(self, max_length):
            super().__init__()
            self.max_length = max_length

        def forward(self, pos_seq):

            pos_seq[pos_seq < -self.max_length] = -self.max_length
            return pos_seq

    pos_seq = torch.arange(tensor_size).unsqueeze(1) - torch.arange(tensor_size).unsqueeze(0)

    inputs = [pos_seq]
    model = index_put(max_length)
    model.eval()

    compiled_model = forge.compile(
        model,
        sample_inputs=inputs,
    )
    verify(inputs, model, compiled_model)


@pytest.mark.xfail
@pytest.mark.nightly
@pytest.mark.skip_model_analysis
@pytest.mark.parametrize(
    "tensor_size,num_heads,head_dim,max_length",
    [
        (24, 12, 64, 160),
        (16, 8, 32, 100),
        (32, 16, 64, 200),
        (8, 4, 16, 50),
        (12, 6, 128, 80),
    ],
)
def test_view_speecht5_tts(tensor_size, num_heads, head_dim, max_length):
    class view(nn.Module):
        def __init__(self, num_heads, head_dim, max_length):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.dim = head_dim
            self.max_length = max_length
            self.pe_k = torch.nn.Embedding(2 * max_length, head_dim)

        def forward(self, pos_seq, reshape_q):
            pos_seq[pos_seq < -self.max_length] = -self.max_length
            pos_seq[pos_seq >= self.max_length] = self.max_length - 1
            pos_seq = pos_seq + self.max_length
            position_bias = self.pe_k(pos_seq)
            rel_pos_bias = torch.matmul(reshape_q, position_bias.transpose(-2, -1))
            rel_pos_bias = rel_pos_bias.transpose(0, 1)
            rel_pos_bias = rel_pos_bias.view(1 * self.num_heads, position_bias.size(0), position_bias.size(1))
            return rel_pos_bias

    x = torch.arange(tensor_size).unsqueeze(1) - torch.arange(tensor_size).unsqueeze(0)
    y = torch.randn((tensor_size, num_heads, head_dim), dtype=torch.float32)

    inputs = [x, y]
    model = view(num_heads, head_dim, max_length)
    model.eval()

    compiled_model = forge.compile(
        model,
        sample_inputs=inputs,
    )
    verify(inputs, model, compiled_model)


@pytest.mark.xfail
@pytest.mark.nightly
@pytest.mark.skip_model_analysis
@pytest.mark.parametrize(
    "shape",
    [
        (1, 24, 768),
        (1, 16, 512),
        (1, 32, 128),
        (1, 8, 64),
        (1, 160, 256),
        (1, 200, 512),
        (2, 24, 768),
    ],
)
def test_scatter_elements(shape):
    class scatter_elements(nn.Module):
        def __init__(self):
            super().__init__()
            self.max_length = 160

        def forward(self, hidden_states):
            seq_len = hidden_states.shape[1]
            pos_seq = torch.arange(0, seq_len).long().to(hidden_states.device)
            pos_seq = pos_seq[:, None] - pos_seq[None, :]
            pos_seq[pos_seq < -self.max_length] = -self.max_length
            return pos_seq

    inputs = [torch.randn(shape, dtype=torch.float32)]
    model = scatter_elements()
    model.eval()

    compiled_model = forge.compile(model, sample_inputs=inputs)
    verify(inputs, model, compiled_model)


@pytest.mark.skip_model_analysis
@pytest.mark.xfail
@pytest.mark.push
@pytest.mark.nightly
def test_bart_cls_head_onnx():
    class wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.classification_head

        def forward(self, hidden_states, eos_mask):

            sentence_representation = hidden_states[eos_mask, :].view(
                hidden_states.size(0), -1, hidden_states.size(-1)
            )[:, -1, :]
            op = self.model(sentence_representation)
            return op

    # prepare model and input
    model = BartForSequenceClassification.from_pretrained("facebook/bart-large-mnli", torchscript=True)
    model.eval()
    torch_model = wrapper(model)
    hidden_states = torch.randn(1, 256, 1024)
    eos_mask = torch.zeros((1, 256), dtype=torch.bool)
    true_indices = [52, 55, 56]
    eos_mask[0, true_indices] = True
    inputs = [hidden_states, eos_mask]

    # Export model to ONNX
    onnx_path = "bart_cls_head.onnx"
    torch.onnx.export(torch_model, (inputs[0], inputs[1]), onnx_path, opset_version=17, verbose=True)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule("bart_cls_head", onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name="bart_cls_head")

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
    )


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
@pytest.mark.push
@pytest.mark.parametrize(
    "shape,axis,index",
    [
        ((1, 3, 1024), 1, -1),
        ((1, 32, 124), 2, 0),
        ((1, 2, 24), 2, -1),
        ((1, 51, 83), 1, 2),
        ((2, 40, 60), 0, 1),
    ],
)
def test_gather_onnx(shape, axis, index):
    class Gather(nn.Module):
        def __init__(self, axis, index):
            super().__init__()
            self.axis = axis
            self.index = index

        def forward(self, x):
            if self.axis == 0:
                return x[self.index, :, :]
            elif self.axis == 1:
                return x[:, self.index, :]
            elif self.axis == 2:
                return x[:, :, self.index]

    torch_model = Gather(axis, index)
    inputs = [torch.randn(shape)]

    onnx_path = f"gather_axis_{axis}.onnx"
    torch.onnx.export(torch_model, (inputs[0],), onnx_path, opset_version=17, verbose=True)

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule("gather", onnx_model)

    compiled_model = forge.compile(onnx_model, inputs, module_name="gather")

    verify(inputs, framework_model, compiled_model)


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
def test_shift_tokens_right():
    class ShiftTokensRight(nn.Module):
        def __init__(self, pad_token_id: int, decoder_start_token_id: int):
            """
            Module that shifts input tokens to the right by one position.

            Args:
                pad_token_id (int): The token ID used for padding.
                decoder_start_token_id (int): The token ID to place at the beginning.
            """
            super().__init__()
            self.pad_token_id = pad_token_id
            self.decoder_start_token_id = decoder_start_token_id

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            """
            Args:
                input_ids (torch.Tensor): Tensor of shape (batch_size, seq_len)

            Returns:
                torch.Tensor: Shifted input_ids of the same shape
            """
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            # shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()  ## Original code
            shifted_input_ids = torch.cat([shifted_input_ids[:, :1], input_ids[:, :-1].clone()], dim=1)
            shifted_input_ids[:, 0] = self.decoder_start_token_id
            shifted_input_ids.masked_fill_(shifted_input_ids == -100, self.pad_token_id)
            return shifted_input_ids + (input_ids * 0)

    vocab_size = 1000
    input_ids = torch.randint(low=0, high=vocab_size, size=(1, 256), dtype=torch.int64)
    shift_module = ShiftTokensRight(pad_token_id=1, decoder_start_token_id=2)

    # Forge compile framework model
    compiled_model = forge.compile(
        shift_module,
        sample_inputs=[input_ids],
        module_name="shift_tokens_right",
        verify_cfg=DeprecatedVerifyConfig(verify_forge_codegen_vs_framework=True),
    )

    verify([input_ids], shift_module, compiled_model)
