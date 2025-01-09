# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


def batch_to_head_dim(tensor, heads=8):
    _, batch_size, seq_len, dim = tensor.shape
    tensor = ttnn.reshape(tensor, (batch_size // heads, heads, seq_len, dim))
    tensor = ttnn.permute(tensor, (0, 2, 1, 3))
    tensor = ttnn.reshape(tensor, (1, batch_size // heads, seq_len, dim * heads))
    return tensor


def head_to_batch_dim(tensor, heads=8, device=None):
    batch_size, _, seq_len, dim = tensor.shape
    tensor = ttnn.reshape(tensor, (batch_size, seq_len, heads, dim // heads))
    tensor = ttnn.permute(tensor, (0, 2, 1, 3))
    tensor = ttnn.reshape(tensor, (1, batch_size * heads, seq_len, dim // heads))
    return tensor


# Equivalent code for scaled_dot_product_attention
def get_attention_scores(query, key, attention_mask=None, scale=None, device=None):
    t_key = ttnn.permute(key, (0, 1, 3, 2))
    temp = ttnn.matmul(query, t_key)
    attention_scores = ttnn.mul(temp, scale)
    ttnn.deallocate(key)
    ttnn.deallocate(t_key)
    ttnn.deallocate(temp)
    if attention_mask is not None:
        attention_scores = ttnn.add(attention_scores, attention_mask)
    attention_probs = ttnn.softmax(attention_scores, dim=-1)
    return attention_probs


def sd_attention(
    hidden_states,
    encoder_hidden_states,
    query_dim: int = None,
    cross_attention_dim=None,
    heads: int = 8,
    attention_mask=None,
    cross_attention_kwargs={},
    *,
    parameters,
    device,
):
    if hidden_states.is_sharded():
        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)
    memory_config = ttnn.create_sharded_memory_config(
        hidden_states.shape,
        core_grid=device.core_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    # hidden_states = ttnn.to_memory_config(
    #     hidden_states,
    #     memory_config=memory_config,
    #     # dtype=ttnn.bfloat8_b,
    # )
    query = ttnn.linear(
        hidden_states,
        parameters.to_q.weight,
        memory_config=memory_config,
        core_grid=device.core_grid,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
        ),
    )
    if query.is_sharded():
        query = ttnn.sharded_to_interleaved(query, ttnn.L1_MEMORY_CONFIG)
    query = head_to_batch_dim(query, heads=heads, device=device)
    encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
    encoder_hidden_states = ttnn.to_layout(encoder_hidden_states, layout=ttnn.TILE_LAYOUT)

    sharding = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
    # if(hidden_states.shape[2] == 1024):
    #     sharding = ttnn.L1_MEMORY_CONFIG

    print(encoder_hidden_states.shape)
    key = ttnn.linear(
        encoder_hidden_states,
        parameters.to_k.weight,
        dtype=ttnn.bfloat16,
        memory_config=sharding,
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
        ),
    )
    print("encoder_hidden_states: ", encoder_hidden_states.shape)

    print(encoder_hidden_states.shape)
    value = ttnn.linear(
        encoder_hidden_states,
        parameters.to_v.weight,
        dtype=ttnn.bfloat16,
        memory_config=sharding,
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
        ),
    )

    if key.is_sharded():
        key = ttnn.sharded_to_interleaved(key, ttnn.L1_MEMORY_CONFIG)
    if value.is_sharded():
        value = ttnn.sharded_to_interleaved(value, ttnn.L1_MEMORY_CONFIG)

    key = head_to_batch_dim(key, heads=heads, device=device)
    value = head_to_batch_dim(value, heads=heads, device=device)
    scale = query.shape[-1] ** -0.5
    attention_probs = get_attention_scores(query, key, attention_mask, scale=scale, device=device)
    ttnn.deallocate(key)
    ttnn.deallocate(query)
    hidden_states = ttnn.matmul(attention_probs, value)
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)
    hidden_states = batch_to_head_dim(hidden_states, heads=heads)
    print(hidden_states.shape)

    hidden_states = ttnn.linear(
        hidden_states,
        parameters.to_out[0].weight,
        bias=parameters.to_out[0].bias,
        dtype=ttnn.bfloat16,
        memory_config=memory_config,
        core_grid=device.core_grid,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
        ),
    )
    hidden_states = ttnn.squeeze(hidden_states, 0)
    return hidden_states
