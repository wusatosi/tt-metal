# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


from torch import nn
import torch
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_t5_attention import ttnn_T5Attention
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_t5_layer_norm import ttnn_T5LayerNorm
from typing import Optional


class ttnn_T5LayerSelfAttention:
    def __init__(self, config, has_relative_attention_bias=False, layer_idx: Optional[int] = None, parameters=None):
        self.SelfAttention = ttnn_T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx
        )
        self.layer_norm = ttnn_T5LayerNorm(parameters=parameters["layer_norm"], eps=config.layer_norm_epsilon)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
        parameters=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            parameters=parameters["SelfAttention"],
        )
        ttnn.deallocate(normed_hidden_states)
        hidden_states = hidden_states + attention_output[0]
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs
