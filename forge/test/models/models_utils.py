# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import requests
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor
import torch
from tabulate import tabulate
import json
from typing import Optional, Tuple
from transformers import Cache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from third_party.tt_forge_models.tools.utils import get_file
from datasets import load_dataset
from torch import Tensor
from loguru import logger

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def preprocess_input_data():
    input_image = get_file("http://images.cocodataset.org/val2017/000000397133.jpg")
    input_image = Image.open(str(input_image))
    input_tensor = transforms.ToTensor()(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def build_optimum_cli_command(variant, tmp_path):
    """
    Constructs the command list for exporting a model using optimum-cli.

    Args:
        variant (str): The model variant to export.
        tmp_path (str): The temporary path where the model will be saved.

    Returns:
        list: A list of command components.
    """
    return [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        variant,
        tmp_path,
        "--opset",
        "17",
        "--monolith",
        "--framework",
        "pt",
        "--trust-remote-code",
        "--task",
        "text-generation",
        "--library-name",
        "transformers",
        "--batch_size",
        "1",
        "--legacy",
    ]


def get_sample_data(model_name):
    input_image = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
    image = Image.open(str(input_image))
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    return [pixel_values]


def load_class_labels(file_path):
    """Load class labels from a JSON or TXT file."""
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            class_idx = json.load(f)
        return [class_idx[str(i)][1] for i in range(len(class_idx))]
    elif file_path.endswith(".txt"):
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip()]


def print_cls_results(fw_out, compiled_model_out, use_1k_labels: bool = True):

    if use_1k_labels:
        imagenet_class_index_path = "forge/test/models/files/labels/imagenet_class_index.json"
    else:
        imagenet_class_index_path = str(
            get_file(
                "https://raw.githubusercontent.com/mosjel/ImageNet_21k_Original_OK/main/imagenet_21k_original_OK.txt"
            )
        )

    class_labels = load_class_labels(imagenet_class_index_path)
    fw_top1_probabilities, fw_top1_class_indices = torch.topk(fw_out.softmax(dim=1) * 100, k=1)
    compiled_model_top1_probabilities, compiled_model_top1_class_indices = torch.topk(
        compiled_model_out.softmax(dim=1) * 100, k=1
    )

    # Directly get the top 1 predicted class and its probability for both models
    fw_top1_class_idx = fw_top1_class_indices[0, 0].item()
    compiled_model_top1_class_idx = compiled_model_top1_class_indices[0, 0].item()
    fw_top1_class_prob = fw_top1_probabilities[0, 0].item()
    compiled_model_top1_class_prob = compiled_model_top1_probabilities[0, 0].item()

    # Get the class labels for top 1 class
    fw_top1_class_label = class_labels[fw_top1_class_idx]
    compiled_model_top1_class_label = class_labels[compiled_model_top1_class_idx]

    # Prepare the results for displaying
    table = [
        ["Metric", "Framework Model", "Compiled Model"],
        ["Top 1 Predicted Class Label", fw_top1_class_label, compiled_model_top1_class_label],
        ["Top 1 Predicted Class Probability", fw_top1_class_prob, compiled_model_top1_class_prob],
    ]
    print(tabulate(table, headers="firstrow", tablefmt="grid"))


def generate_no_cache(max_new_tokens, model, inputs, seq_len, tokenizer):
    """
    Generates text autoregressively without using a KV cache, iteratively predicting one token at a time.
    The function stops generation if the maximum number of new tokens is reached or an end-of-sequence (EOS) token is encountered.

    Args:
        max_new_tokens (int): The maximum number of new tokens to generate.
        model (torch.nn.Module): The language model used for token generation.
        inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len), representing tokenized text.
        seq_len (int): The current sequence length before generation starts.
        tokenizer: The tokenizer used to decode token IDs into text.

    Returns:
        str: The generated text after decoding the new tokens.
    """
    current_pos = seq_len

    for _ in range(max_new_tokens):
        logits = model(inputs)

        # Get only the logits corresponding to the last valid token
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        elif isinstance(logits, torch.Tensor):
            logits = logits
        else:
            raise TypeError(f"Expected logits to be a list or tuple or torch.Tensor, but got {type(logits)}")
        next_token_logits = logits[:, current_pos - 1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        # Stop if EOS token is encountered
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        inputs[:, current_pos] = next_token_id

        current_pos += 1  # Move to next position

    # Decode valid tokens
    valid_tokens = inputs[:, seq_len:current_pos].view(-1).tolist()
    answer = tokenizer.decode(valid_tokens, skip_special_tokens=True)

    return answer


def pad_inputs(inputs, max_new_tokens=512):
    batch_size, seq_len = inputs.shape
    max_seq_len = seq_len + max_new_tokens
    padded_inputs = torch.zeros((batch_size, max_seq_len), dtype=inputs.dtype, device=inputs.device)
    padded_inputs[:, :seq_len] = inputs
    return padded_inputs, seq_len


class TextModelWrapper(torch.nn.Module):
    def __init__(self, model, text_embedding=None):
        super().__init__()
        self.model = model
        self.text_embedding = text_embedding

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is not None and self.text_embedding is not None:
            inputs_embeds = self.text_embedding(input_ids)
            past_key_values_length = 0
            causal_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_ids.shape, inputs_embeds, past_key_values_length
            )
            logits = self.model(attention_mask=causal_attention_mask, inputs_embeds=inputs_embeds).logits
        else:
            logits = self.model(input_ids=input_ids).logits
        return logits


def generate_no_cache_for_encoder_decoder_model(
    max_new_tokens, model, input_ids, decoder_input_ids, seq_len, tokenizer
):

    """
    Generates text from an encoder-decoder model without using KV cache, by iteratively predicting one token at a time using static padded decoder input.
    The function stops generation if the maximum number of new tokens is reached or an end-of-sequence (EOS) token is encountered.

    Args:
        max_new_tokens (int): The maximum number of new tokens to generate.
        model (torch.nn.Module): The encoder-decoder model used for generation.
        input_ids (torch.Tensor): Input tensor of shape (batch_size, encoder_seq_len),
                                  representing the encoder inputs.
        decoder_input_ids (torch.Tensor): Preallocated padded decoder input tensor of shape
                                          (batch_size, max_seq_len), where tokens are filled
                                          in-place during generation.
        seq_len (int): Initial decoder sequence length (typically 1).
        tokenizer: HuggingFace tokenizer used to decode token IDs into text.

    Returns:
        str: The generated text string (decoded without special tokens).
    """

    current_pos = seq_len

    for _ in range(max_new_tokens):
        logits = model(input_ids, decoder_input_ids)

        # Get only the logits corresponding to the last valid token
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        elif isinstance(logits, torch.Tensor):
            pass
        else:
            raise TypeError(f"Expected logits to be a list or tuple or torch.Tensor, but got {type(logits)}")
        next_token_logits = logits[:, current_pos - 1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)

        # Stop if EOS token is encountered
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        decoder_input_ids[:, current_pos] = next_token_id
        current_pos += 1  # Move to next position

    # Decode valid tokens
    valid_tokens = decoder_input_ids[:, seq_len:current_pos].view(-1).tolist()
    answer = tokenizer.decode(valid_tokens, skip_special_tokens=True)

    return answer


def _prepare_4d_causal_attention_mask_with_cache_position(
    self,
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    cache_position: torch.Tensor,
    batch_size: int,
    **kwargs,
):
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0

            # Replace Implace Slice Update
            # causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            #     padding_mask, min_dtype
            # )

            if causal_mask.shape[-1] > mask_length:
                part_1 = causal_mask[:, :, :, :mask_length]
                part_2 = causal_mask[:, :, :, mask_length:]
                part_1 = part_1.masked_fill(padding_mask, min_dtype)
                causal_mask = torch.cat([part_1, part_2], dim=-1)
            else:
                causal_mask = causal_mask.masked_fill(padding_mask, min_dtype)

    return causal_mask


def _process_image_input(self, image_input):
    import surya.common.surya.processor as _surya_processor_module

    rotated = image_input.get("rotated", False)
    image = image_input.get("image", None)
    assert image is not None, "A PIL Image must be provided when the input type is `image`"
    image_tiles, grid_thw = self._process_and_tile(image)

    num_tokens = int(image_tiles.shape[0]) / self.merge_size**2
    assert float(num_tokens).is_integer(), f"Expected number of tokens to be an integer, got {num_tokens}"

    input_ids = [self.image_token_id] * int(num_tokens)
    input_ids += self.register_token_ids[: self.num_register_tokens]

    if rotated:
        input_ids = [self.image_rotated_token] + input_ids

    return _surya_processor_module.ProcessorOutput(
        input_ids=input_ids,
        image_tiles=image_tiles,
        grid_thw=grid_thw,
    )


def Gemma2DecoderLayer_patched_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    if self.is_sliding and attention_mask is not None:  # efficient SDPA and no padding
        # In prefill, we may be larger than sliding window
        effective_seq_len = max(cache_position.shape[0], self.sliding_window)
        # For FA2, the mask is 2D and is of shape [bs, processed_tokens] (not [bs, max_cache_len]),
        # thus we must slice from the right (at most `effective_seq_len` elements)
        if self.config._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask[:, -effective_seq_len:]
        # Otherwise, the mask is 4D of shape [bs, 1, query_len, max_cache_len] thus we must slice
        # from the left, with an offset if we are beyond the sliding window
        else:
            # min_dtype = torch.finfo(hidden_states.dtype).min

            # [Monkey patch] Cast scalar to tensor with hidden_states dtype to fix ONNX Where op type mismatch
            min_dtype = torch.tensor(torch.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype)

            sliding_window_mask = torch.tril(
                torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-self.sliding_window
            )
            attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)
            # In case we are beyond the sliding window, we need to correctly offset the mask slicing
            offset = cache_position[-1] - effective_seq_len + 1
            # Should only be used when beyond the sliding window (i.e. offset > 0)
            offset = torch.clamp(offset, min=0)
            # equivalent to: `attention_mask = attention_mask[:, :, :, offset : offset + effective_seq_len]`,
            # but without data-dependent slicing (i.e. torch.compile friendly)
            mask_indexes = torch.arange(min(effective_seq_len, attention_mask.shape[-1]), device=attention_mask.device)
            mask_indexes += offset
            attention_mask = attention_mask[:, :, :, mask_indexes]

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights = self.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    return outputs


def preprocess_inputs():

    try:
        dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, token=True)
        input_image = next(iter(dataset.skip(10)))["image"]

        # Prepare input
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
    except Exception as e:
        logger.warning(f"Error loading dataset: {e}")
        input_batch = torch.rand(1, 3, 224, 224)

    return [input_batch]
