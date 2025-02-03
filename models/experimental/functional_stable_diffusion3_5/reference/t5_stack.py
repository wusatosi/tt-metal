# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from models.experimental.functional_stable_diffusion3_5.reference.t5_block import T5Block
from models.experimental.functional_stable_diffusion3_5.reference.t5_layer_norm import T5LayerNorm
from transformers.utils.import_utils import is_torchdynamo_compiling
from transformers.models.t5 import T5PreTrainedModel
from transformers.models.t5.modeling_t5 import BaseModelOutputWithPastAndCrossAttentions


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0), layer_idx=i) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
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
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
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
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values
        return_legacy_cache = False
        return_self_attention_cache = False
        if self.is_decoder and (use_cache or past_key_values is not None):
            if isinstance(past_key_values, Cache) and not isinstance(past_key_values, EncoderDecoderCache):
                return_self_attention_cache = True
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            elif not isinstance(past_key_values, EncoderDecoderCache):
                return_legacy_cache = True
                logger.warning_once(
                    "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.48.0. "
                    "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                    "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
                )
                past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
            elif past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        elif not self.is_decoder:
            # do not pass cache object down the line for encoder stack
            # it messes indexing later in decoder-stack because cache object is modified in-place
            past_key_values = None

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )

        if attention_mask is None and not is_torchdynamo_compiling():
            # required mask seq length can be calculated via length of past cache
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        if self.config.is_decoder:
            causal_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values.self_attention_cache if past_key_values is not None else None,
                output_attentions,
            )
        elif attention_mask is not None:
            causal_mask = attention_mask[:, None, None, :]
            causal_mask = causal_mask.to(dtype=inputs_embeds.dtype)
            causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min
        else:
            causal_mask = None

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, layer_module in enumerate(self.block):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if causal_mask is not None:
                    causal_mask = causal_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    causal_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
                    return_dict,
                    cache_position,
                )
            else:
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
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, next_decoder_cache = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

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
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

    # # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    # def _update_causal_mask(
    #     self,
    #     attention_mask: torch.Tensor,
    #     input_tensor: torch.Tensor,
    #     cache_position: torch.Tensor,
    #     past_key_values: Cache,
    #     output_attentions: bool,
    # ):
    #     if self.config._attn_implementation == "flash_attention_2":
    #         if attention_mask is not None and (attention_mask == 0.0).any():
    #             return attention_mask
    #         return None

    #     # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    #     # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    #     # to infer the attention mask.
    #     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    #     using_static_cache = isinstance(past_key_values, StaticCache)

    #     # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
    #     if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
    #         if AttentionMaskConverter._ignore_causal_mask_sdpa(
    #             attention_mask,
    #             inputs_embeds=input_tensor,
    #             past_key_values_length=past_seen_tokens,
    #             is_training=self.training,
    #         ):
    #             return None

    #     dtype, device = input_tensor.dtype, input_tensor.device
    #     sequence_length = input_tensor.shape[1]
    #     if using_static_cache:
    #         target_length = past_key_values.get_max_cache_shape()
    #     else:
    #         target_length = (
    #             attention_mask.shape[-1]
    #             if isinstance(attention_mask, torch.Tensor)
    #             else past_seen_tokens + sequence_length + 1
    #         )

    #     # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
    #     causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
    #         attention_mask,
    #         sequence_length=sequence_length,
    #         target_length=target_length,
    #         dtype=dtype,
    #         device=device,
    #         cache_position=cache_position,
    #         batch_size=input_tensor.shape[0],
    #     )

    #     if (
    #         self.config._attn_implementation == "sdpa"
    #         and attention_mask is not None
    #         and attention_mask.device.type == "cuda"
    #         and not output_attentions
    #     ):
    #         # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
    #         # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
    #         # Details: https://github.com/pytorch/pytorch/issues/110213
    #         min_dtype = torch.finfo(dtype).min
    #         causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    #     return causal_mask

    # @staticmethod
    # # Copied from transformers.models.llama.modeling_llama.LlamaPreTrainedModel._prepare_4d_causal_attention_mask_with_cache_position
    # def _prepare_4d_causal_attention_mask_with_cache_position(
    #     attention_mask: torch.Tensor,
    #     sequence_length: int,
    #     target_length: int,
    #     dtype: torch.dtype,
    #     device: torch.device,
    #     cache_position: torch.Tensor,
    #     batch_size: int,
    #     **kwargs,
    # ):
    #     """
    #     Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    #     `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    #     Args:
    #         attention_mask (`torch.Tensor`):
    #             A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
    #             `(batch_size, 1, query_length, key_value_length)`.
    #         sequence_length (`int`):
    #             The sequence length being processed.
    #         target_length (`int`):
    #             The target length: when generating with static cache, the mask should be as long as the static cache,
    #             to account for the 0 padding, the part of the cache that is not filled yet.
    #         dtype (`torch.dtype`):
    #             The dtype to use for the 4D attention mask.
    #         device (`torch.device`):
    #             The device to plcae the 4D attention mask on.
    #         cache_position (`torch.Tensor`):
    #             Indices depicting the position of the input sequence tokens in the sequence.
    #         batch_size (`torch.Tensor`):
    #             Batch size.
    #     """
    #     if attention_mask is not None and attention_mask.dim() == 4:
    #         # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
    #         causal_mask = attention_mask
    #     else:
    #         min_dtype = torch.finfo(dtype).min
    #         causal_mask = torch.full(
    #             (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
    #         )
    #         if sequence_length != 1:
    #             causal_mask = torch.triu(causal_mask, diagonal=1)
    #         causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    #         causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    #         if attention_mask is not None:
    #             causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
    #             mask_length = attention_mask.shape[-1]
    #             padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
    #             padding_mask = padding_mask == 0
    #             causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
    #                 padding_mask, min_dtype
    #             )

    #     return causal_mask
