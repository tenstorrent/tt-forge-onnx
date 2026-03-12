# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)

# SPDX-FileCopyrightText: Copyright (c) 2021 OpenAI
#
# SPDX-License-Identifier: MIT
# https://github.com/openai/CLIP


class CLIPTextWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.clip_model = model

    def forward(self, input_ids, attention_mask):

        # text_outputs = self.clip_model.text_model(input_ids, attention_mask, return_dict=False)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.clip_model.text_model.embeddings(input_ids=input_ids, position_ids=None)

        bsz, seq_len = input_shape
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.clip_model.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            return_dict=False,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.clip_model.text_model.final_layer_norm(last_hidden_state)

        return (last_hidden_state, *encoder_outputs)
