# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Optional
import torch
from ttnn.model_preprocessing import (
    preprocess_linear_bias,
    preprocess_linear_weight,
)
import math


def attention(config, input, att_bias, parameters, device):
    B, T, C = input.shape
    weight = ttnn.to_layout(parameters.attn.c_attn.weight, layout=ttnn.TILE_LAYOUT)
    bias = ttnn.to_layout(parameters.attn.c_attn.bias, layout=ttnn.TILE_LAYOUT)
    x1 = ttnn.linear(
        input,
        weight,
        bias=bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    q = x1[:, :, : config.hidden_size]
    k = x1[:, :, config.hidden_size : config.hidden_size * 2]
    v = x1[:, :, config.hidden_size * 2 : config.hidden_size * 3]

    k = ttnn.reshape(k, (B, T, config.n_head, C // config.n_head))
    k = ttnn.permute(k, (0, 2, 1, 3))

    q = ttnn.reshape(q, (B, T, config.n_head, C // config.n_head))
    q = ttnn.permute(q, (0, 2, 1, 3))

    v = ttnn.reshape(v, (B, T, config.n_head, C // config.n_head))
    v = ttnn.permute(v, (0, 2, 1, 3))

    key_layer_transposed = ttnn.permute(k, (0, 1, 3, 2))

    att = q @ key_layer_transposed

    const_att = ttnn.full(att.shape, 1.0 / math.sqrt(k.shape[-1]))
    const_att = ttnn.to_device(const_att, device=device)
    const_att = ttnn.to_layout(const_att, layout=ttnn.TILE_LAYOUT)
    att = att * const_att

    att = att + att_bias

    att = ttnn.softmax(att, dim=-1)
    y = att @ v

    y = ttnn.permute(y, (0, 2, 1, 3))
    y = ttnn.reshape(y, (B, T, C))

    weight = ttnn.to_layout(parameters.attn.c_proj.weight, layout=ttnn.TILE_LAYOUT)
    bias = ttnn.to_layout(parameters.attn.c_proj.bias, layout=ttnn.TILE_LAYOUT)
    x = ttnn.linear(
        y,
        weight,
        bias=bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    return x


def nanogpt_block(config, input, att_bias, parameters, device):
    ln1 = ttnn.layer_norm(
        input,
        weight=parameters.ln_1.weight,
        bias=parameters.ln_1.bias,
        epsilon=1e-5,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    attn = attention(config, ln1, att_bias, parameters, device)

    x = input + attn
    residual = x
    ln2 = ttnn.layer_norm(
        x,
        weight=parameters.ln_2.weight,
        bias=parameters.ln_2.bias,
        epsilon=1e-5,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    weight = ttnn.to_layout(parameters.mlp.c_fc.weight, layout=ttnn.TILE_LAYOUT)
    bias = ttnn.to_layout(parameters.mlp.c_fc.bias, layout=ttnn.TILE_LAYOUT)
    mlp_1 = ttnn.linear(
        ln2,
        weight,
        bias=bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        activation="gelu",
    )

    weight = ttnn.to_layout(parameters.mlp.c_proj.weight, layout=ttnn.TILE_LAYOUT)
    bias = ttnn.to_layout(parameters.mlp.c_proj.bias, layout=ttnn.TILE_LAYOUT)
    mlp_2 = ttnn.linear(
        mlp_1,
        weight,
        bias=bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    op = mlp_2 + residual
    return op


def nanogpt_model(config, input_idx, position_ids, att_bias, parameters, device):
    b, t = input_idx.shape
    pos = ttnn.arange(start=0, end=t, dtype=ttnn.int32)

    tok_emb = ttnn.embedding(
        input_idx, parameters.transformer.wte.weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    pos_emb = ttnn.embedding(
        position_ids, parameters.transformer.wpe.weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    emb = tok_emb + pos_emb

    for i in range(config.n_layer):
        emb = nanogpt_block(config, emb, att_bias, parameters.transformer.h[i], device)

    x = ttnn.layer_norm(
        emb,
        weight=parameters.transformer.ln_f.weight,
        bias=parameters.transformer.ln_f.bias,
        epsilon=1e-5,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    weight = ttnn.to_layout(parameters.lm_head.weight, layout=ttnn.TILE_LAYOUT)
    output = ttnn.linear(
        x,
        weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    return output
