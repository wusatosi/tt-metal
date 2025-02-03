# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Optional
import torch
import math


class ttnn_T5Attention:
    def __init__(
        self,
        config,
        has_relative_attention_bias=False,
        layer_idx: Optional[int] = None,
    ):
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.layer_idx = layer_idx
        if layer_idx is None and self.is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = ttnn.linear
        self.k = ttnn.linear
        self.v = ttnn.linear
        self.o = ttnn.linear

        if self.has_relative_attention_bias:
            self.relative_attention_bias = ttnn.embedding
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = ttnn.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            ttnn.div(ttnn.log(ttnn.div(relative_position, max_exact)), math.log(max_distance / max_exact))
            * (num_buckets - max_exact)
        )

        dim = ttnn.full_like(relative_position_if_large, num_buckets - 1)
        torch_dim = ttnn.to_torch(dim)
        device = relative_position_if_large.device()
        relative_position_if_large = ttnn.to_torch(relative_position_if_large)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch_dim
        )  # ttnn.min doesn't support min(tensor,tensor)

        relative_position_if_large = ttnn.from_torch(relative_position_if_large, layout=ttnn.TILE_LAYOUT, device=device)

        relative_buckets += ttnn.where(is_small, relative_position, relative_position_if_large)
        ttnn.deallocate(relative_position)
        ttnn.deallocate(relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None, cache_position=None, parameters=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        if cache_position is None:
            context_position = ttnn.arange(query_length, device=device)[:, None]
        else:
            context_position = ttnn.unsqueeze(cache_position, 1)
        memory_position = ttnn.arange(0, key_length, device=device)
        memory_position = ttnn.squeeze(memory_position, 0)
        memory_position = ttnn.squeeze(memory_position, 0)
        memory_position = ttnn.to_layout(memory_position, layout=ttnn.TILE_LAYOUT)
        context_position = ttnn.to_layout(context_position, layout=ttnn.TILE_LAYOUT)
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        ttnn.deallocate(memory_position)
        ttnn.deallocate(context_position)

        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        values = self.relative_attention_bias(
            relative_position_bucket,
            weight=parameters["relative_attention_bias"]["weight"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )  # shape (query_length, key_length, num_heads)
        ttnn.deallocate(relative_position_bucket)
        values = ttnn.permute(values, (2, 0, 1))
        values = ttnn.unsqueeze(values, 0)
        return values

    def __call__(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
        parameters=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        ttnn_device = hidden_states.device()
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None

        query_states = self.q(hidden_states, parameters["q"]["weight"], memory_config=ttnn.L1_MEMORY_CONFIG)
        query_states = ttnn.to_memory_config(query_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query_states = ttnn.reshape(query_states, (batch_size, -1, self.n_heads, self.key_value_proj_dim))
        query_states = ttnn.permute(query_states, (0, 2, 1, 3))

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_value = past_key_value.cross_attention_cache
            else:
                curr_past_key_value = past_key_value.self_attention_cache

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_value is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_value.key_cache[self.layer_idx]
            value_states = curr_past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k(current_states, parameters["k"]["weight"], memory_config=ttnn.L1_MEMORY_CONFIG)
            value_states = self.v(current_states, parameters["v"]["weight"], memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(current_states)

            key_states = ttnn.reshape(key_states, (batch_size, -1, self.n_heads, self.key_value_proj_dim))
            key_states = ttnn.permute(key_states, (0, 2, 1, 3))
            value_states = ttnn.reshape(value_states, (batch_size, -1, self.n_heads, self.key_value_proj_dim))
            value_states = ttnn.permute(value_states, (0, 2, 1, 3))

            if past_key_value is not None:
                # This is not called
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    past_key_value.is_updated[self.layer_idx] = True

        # compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = ttnn.matmul(query_states, ttnn.permute(key_states, (0, 1, 3, 2)))
        ttnn.deallocate(query_states)

        if position_bias is None:
            key_length = key_states.shape[-2]
            real_seq_length = query_length if query_length is not None else ttnn.to_torch(cache_position)[-1] + 1
            if not self.has_relative_attention_bias:
                position_bias = ttnn.zeros((1, self.n_heads, seq_length, key_length), device=ttnn_device)
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length,
                    key_length,
                    device=scores.device(),
                    cache_position=cache_position,
                    parameters=parameters,
                )
                position_bias = position_bias[:, :, -seq_length:, :]

            if mask is not None:
                causal_mask = mask[:, :, :, : key_states.shape[-2]]
                position_bias = ttnn.to_layout(position_bias, layout=ttnn.TILE_LAYOUT)
                position_bias = position_bias + causal_mask

        ttnn.deallocate(key_states)
        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores = ttnn.add(scores, position_bias, memory_config=ttnn.L1_MEMORY_CONFIG)

        attn_weights = ttnn.softmax(scores, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(scores)
        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = ttnn.matmul(
            attn_weights,
            value_states,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(value_states)

        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (batch_size, -1, self.inner_dim))
        attn_output = self.o(
            attn_output,
            parameters.o.weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        outputs = (attn_output, past_key_value, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs
