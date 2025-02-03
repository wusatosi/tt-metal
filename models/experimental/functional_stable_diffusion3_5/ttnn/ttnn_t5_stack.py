# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

import torch
from torch import nn
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_t5_block import ttnn_T5Block
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_t5_layer_norm import ttnn_T5LayerNorm
from transformers.utils.import_utils import is_torchdynamo_compiling
from dataclasses import dataclass


@dataclass
class ttnn_BaseModelOutputWithPastAndCrossAttentions:
    def __init__(self, last_hidden_state, past_key_values, hidden_states, attentions, cross_attentions):
        last_hidden_state = None
        past_key_values = None
        hidden_states = None
        attentions = None
        cross_attentions = None


class ttnn_T5Stack:
    def __init__(self, config, parameters=None):
        self.config = config  # added
        self.embed_tokens = ttnn.embedding
        self.is_decoder = config.is_decoder

        self.block = [
            ttnn_T5Block(config, has_relative_attention_bias=bool(i == 0), layer_idx=i, parameters=parameters.block[i])
            for i in range(config.num_layers)
        ]

        self.final_layer_norm = ttnn_T5LayerNorm(parameters.final_layer_norm, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        parameters=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = ttnn.reshape(input_ids, (-1, input_shape[-1]))
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(
                input_ids, weight=parameters.embed_tokens.weight, memory_config=ttnn.L1_MEMORY_CONFIG
            )

        batch_size, seq_length = input_shape

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values
        return_legacy_cache = False
        return_self_attention_cache = False

        past_key_values = None

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = ttnn.arange(
                past_key_values_length, past_key_values_length + seq_length, device=input_ids.device()
            )
            # converting 4D to 1D
            cache_position = ttnn.squeeze(cache_position, 0)
            cache_position = ttnn.squeeze(cache_position, 0)
            cache_position = ttnn.squeeze(cache_position, 0)
            cache_position = ttnn.to_layout(cache_position, layout=ttnn.TILE_LAYOUT)

        if attention_mask is None and not is_torchdynamo_compiling():
            # required mask seq length can be calculated via length of past cache
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = ttnn.ones([batch_size, mask_seq_length], device=input_ids.device())

        if attention_mask is not None:
            causal_mask = attention_mask
            causal_mask = ttnn.unsqueeze(causal_mask, 0)
            causal_mask = ttnn.unsqueeze(causal_mask, 0)
            # causal_mask=ttnn.to_layout(causal_mask,layout=ttnn.TILE_LAYOUT)
            causal_mask = ttnn.to_torch(causal_mask)
            causal_mask = (1.0 - causal_mask) * torch.finfo(
                ttnn.to_torch(inputs_embeds).dtype
            ).min  # Need to convert to ttnn
            causal_mask = ttnn.from_torch(causal_mask, layout=ttnn.TILE_LAYOUT, device=input_ids.device())
        else:
            causal_mask = None

        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = [None] * self.config.num_layers
        cross_attn_head_mask = [None] * self.config.num_layers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = inputs_embeds
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)

        for i, layer_module in enumerate(self.block):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=causal_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=return_dict,
                cache_position=cache_position,
                parameters=parameters.block[i],
            )

            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, next_decoder_cache = layer_outputs[:2]

            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_self_attention_cache:
            next_cache = past_key_values.self_attention_cache
        if return_legacy_cache:
            next_cache = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return ttnn_BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
