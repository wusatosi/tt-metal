# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn

from tt_lib.utils import (
    _nearest_y,
)
import math
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
from models.demos.stable_diffusion_xl_turbo.tt.resnetblock2d_utils import (
    get_inputs,
    get_weights,
    get_mask_tensor,
    run_conv,
    run_conv_with_split,
)


def batch_to_head_dim(tensor, heads=8):
    _, batch_size, seq_len, dim = tensor.shape
    tensor = ttnn.reshape(tensor, (batch_size // heads, heads, seq_len, dim))
    tensor = ttnn.permute(tensor, (0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)
    tensor = ttnn.reshape(tensor, (1, batch_size // heads, seq_len, dim * heads))
    return tensor


def head_to_batch_dim(tensor, heads=8):
    batch_size, seq_len, dim = tensor.shape
    tensor = ttnn.reshape(tensor, (batch_size, seq_len, heads, dim // heads))
    tensor = ttnn.permute(tensor, (0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)
    tensor = ttnn.reshape(tensor, (1, batch_size * heads, seq_len, dim // heads))
    return tensor


# Equivalent code for scaled_dot_product_attention
def get_attention_scores(query, key, attention_mask=None, scale=None, device=None):
    t_key = ttnn.permute(key, (0, 1, 3, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
    temp = ttnn.matmul(query, t_key)
    attention_scores = ttnn.mul(temp, scale)
    ttnn.deallocate(key)
    ttnn.deallocate(t_key)
    ttnn.deallocate(temp)
    if attention_mask is not None:
        attention_scores = ttnn.add(attention_scores, attention_mask, memory_config=ttnn.L1_MEMORY_CONFIG)
    attention_probs = ttnn.softmax(attention_scores, dim=-1)
    return attention_probs


def sd_geglu(
    hidden_states,
    parameters,
    device=None,
):
    x = ttnn.linear(
        hidden_states,
        parameters.proj.weight,
        bias=parameters.proj.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    x = ttnn.unsqueeze(x, 0)
    x = ttnn.geglu(x, memory_config=ttnn.L1_MEMORY_CONFIG)
    x = ttnn.squeeze(x, 0)
    return x


def sd_feed_forward(
    hidden_states,
    parameters,
    device,
):
    hidden_states = sd_geglu(hidden_states, parameters.net[0], device)
    hidden_states = ttnn.linear(
        hidden_states,
        parameters.net[2].weight,
        bias=parameters.net[2].bias,
        dtype=ttnn.bfloat16,
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return hidden_states


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
    batch_size, sequence_length, _ = hidden_states.shape

    query = ttnn.linear(
        hidden_states,
        parameters.to_q.weight,
        dtype=ttnn.bfloat16,
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    query = head_to_batch_dim(query, heads=heads)

    encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

    key = ttnn.linear(
        encoder_hidden_states,
        parameters.to_k.weight,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    value = ttnn.linear(
        encoder_hidden_states,
        parameters.to_v.weight,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    key = head_to_batch_dim(key, heads=heads)
    value = head_to_batch_dim(value, heads=heads)

    scale = query.shape[-1] ** -0.5

    attention_probs = get_attention_scores(query, key, attention_mask, scale=scale, device=device)

    hidden_states = ttnn.matmul(attention_probs, value, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    hidden_states = batch_to_head_dim(hidden_states, heads=heads)

    hidden_states = ttnn.linear(
        hidden_states,
        parameters.to_out[0].weight,
        bias=parameters.to_out[0].bias,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    hidden_states = ttnn.squeeze(hidden_states, 0)

    return hidden_states


def sd_basic_transformer_block(
    hidden_states,
    encoder_hidden_states=None,
    timestep=None,
    attention_mask=None,
    cross_attention_kwargs=None,
    class_labels=None,
    config=None,
    num_embeds_ada_norm=False,
    cross_attention_dim: int = None,
    only_cross_attention: bool = False,
    attention_head_dim=None,
    *,
    parameters,
    device,
):
    norm_hidden_states = ttnn.layer_norm(
        hidden_states,
        epsilon=1e-05,
        weight=parameters.norm1.weight,
        bias=parameters.norm1.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
    cross_attention_dim = config.cross_attention_dim if cross_attention_dim is None else cross_attention_dim

    attn_output = sd_attention(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states if only_cross_attention else None,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        cross_attention_dim=cross_attention_dim,
        heads=attention_head_dim,
        parameters=parameters.attn1,
        device=device,
    )

    hidden_states = ttnn.add(attn_output, hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)

    if cross_attention_dim is not None:
        norm_hidden_states = ttnn.layer_norm(
            hidden_states,
            epsilon=1e-05,
            weight=parameters.norm2.weight,
            bias=parameters.norm2.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        attn_output = sd_attention(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            cross_attention_dim=cross_attention_dim,
            heads=attention_head_dim,
            parameters=parameters.attn2,
            device=device,
        )

        hidden_states = ttnn.add(attn_output, hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)

        norm_hidden_states = ttnn.layer_norm(
            hidden_states,
            epsilon=1e-05,
            weight=parameters.norm3.weight,
            bias=parameters.norm3.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ff_output = sd_feed_forward(hidden_states=norm_hidden_states, parameters=parameters.ff, device=device)

        hidden_states = ttnn.add(ff_output, hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)

        return hidden_states


def sd_transformer_2d(
    hidden_states,
    encoder_hidden_states=None,
    timestep=None,
    attention_mask=None,
    cross_attention_kwargs=None,
    class_labels=None,
    config=None,
    num_embeds_ada_norm=False,
    cross_attention_dim: int = None,
    norm_num_groups=32,
    only_cross_attention: bool = False,
    attention_head_dim=None,
    return_dict=None,
    num_layers=1,
    eps=1e-5,
    transformer_layers_per_block=10,
    *,
    parameters,
    device,
):
    inner_dim = hidden_states.shape[1]

    residual = hidden_states

    N, C, H, W = hidden_states.shape

    grid_size = ttnn.CoreGrid(y=4, x=8)
    input_mask_tensor = get_mask_tensor(C, norm_num_groups, grid_size.y, device)

    hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
    hidden_states = ttnn.reshape(hidden_states, (N, 1, W * H, C))

    grid_size = ttnn.CoreGrid(y=4, x=8)
    input_mask_tensor = get_mask_tensor(C, norm_num_groups, grid_size.y, device)

    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // grid_size.x, C // grid_size.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR, False)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    hidden_states = ttnn.to_memory_config(hidden_states, sharded_mem_config)

    gamma_t, beta_t = get_weights(parameters.norm.weight, parameters.norm.bias, C, grid_size.y, device)

    hidden_states = ttnn.group_norm(
        input_tensor=hidden_states,
        num_groups=norm_num_groups,
        input_mask=input_mask_tensor,
        epsilon=eps,
        weight=gamma_t,
        bias=beta_t,
        memory_config=sharded_mem_config,
        core_grid=grid_size,
    )
    ttnn.deallocate(input_mask_tensor)
    hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
    hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
    hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))
    hidden_states = ttnn.reshape(hidden_states, (N, H * W, inner_dim))

    hidden_states = ttnn.linear(
        hidden_states,
        parameters.proj_in.weight,
        bias=parameters.proj_in.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    for d in range(transformer_layers_per_block):
        hidden_states = sd_basic_transformer_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            cross_attention_kwargs=cross_attention_kwargs,
            class_labels=class_labels,
            attention_head_dim=attention_head_dim,
            attention_mask=attention_mask,
            config=config,
            parameters=parameters.transformer_blocks[d],
            device=device,
            cross_attention_dim=cross_attention_dim,
            only_cross_attention=only_cross_attention,
        )

    hidden_states = ttnn.linear(
        hidden_states,
        parameters.proj_out.weight,
        bias=parameters.proj_out.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    hidden_states = ttnn.reshape(hidden_states, (N, H, W, inner_dim))
    hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
    residual = ttnn.to_layout(residual, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    output = ttnn.add(
        hidden_states,
        residual,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(hidden_states)
    ttnn.deallocate(residual)

    return output


def sd_downsample_2(input_tensor, parameters, device):
    tt_output_tensor_on_device = run_conv_with_split(
        device,
        input_tensor,
        batch_size=1,
        parameters=parameters,
        dtype=ttnn.bfloat16,
        kernel_size=3,
        stride=2,
        pad=1,
        weights_dtype=ttnn.bfloat8_b,
        split_factor=2,
        ttnn_weight=parameters.conv.weight,
        ttnn_bias=parameters.conv.bias,
    )
    return tt_output_tensor_on_device


def ResnetBlock2D(
    conifg,
    input_tensor=None,
    temb=None,
    parameters=None,
    device=None,
    eps=1e-5,
    groups=32,
    time_embedding_norm="default",
    non_linearity="silu",
    output_scale_factor=1.0,
    use_torch_conv=False,
):
    hidden_states = input_tensor
    input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    N = input_tensor.shape[0]
    batch_size = N
    C = input_tensor.shape[1]
    H = input_tensor.shape[2]
    W = input_tensor.shape[3]
    in_channels = C
    input_height = H
    input_width = W
    grid_size = ttnn.CoreGrid(y=4, x=8)

    use_torch_silu = False
    use_torch_gn = False
    if (C == 960 and H == 128) or (C == 640 and H == 128) or (C == 1920 and H == 64):
        use_torch_silu = True
    if H >= 128 or (C == 1920 and H == 64) or (C == 1280 and H == 64) or (C == 960 and H == 64):
        use_torch_gn = True
    if C == 960 and H == 128:
        use_torch_conv = True

    if use_torch_gn:
        hidden_states = ttnn.to_torch(hidden_states)
        torch_weight = ttnn.to_torch(parameters.norm1.weight)
        torch_bias = ttnn.to_torch(parameters.norm1.bias)
        hidden_states = (
            torch.nn.functional.group_norm(hidden_states, groups, weight=torch_weight, bias=torch_bias)
            .permute(0, 2, 3, 1)
            .view(N, 1, W * H, C)
        )
        hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    else:
        hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = ttnn.reshape(hidden_states, (batch_size, 1, input_width * input_height, in_channels))
        input_mask_tensor = get_mask_tensor(C, groups, grid_size.y, device)
        gamma_t, beta_t = get_weights(parameters.norm1.weight, parameters.norm1.bias, C, grid_size.y, device)

        # shard config
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_shape = N * H * W // grid_size.x, C // grid_size.y
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR, False)
        sharded_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        hidden_states = ttnn.to_memory_config(hidden_states, sharded_mem_config)
        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=groups,
            input_mask=input_mask_tensor,
            weight=gamma_t,
            bias=beta_t,
            epsilon=eps,
            memory_config=sharded_mem_config,
            core_grid=grid_size,
        )
        ttnn.deallocate(input_mask_tensor)

    if non_linearity == "silu":
        if use_torch_silu:
            torch_silu = torch.nn.SiLU()
            hidden_states = ttnn.to_torch(hidden_states)
            hidden_states = torch_silu(hidden_states)
            hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        else:
            hidden_states = get_inputs(device, hidden_states, grid_size)
            hidden_states = ttnn.silu(hidden_states)

    hidden_states = ttnn.reshape(hidden_states, (N, H, W, C))
    batch_size = hidden_states.shape[0]

    if use_torch_conv:
        weight = ttnn.to_torch(parameters.conv1.weight).to(torch.float)
        bias = ttnn.to_torch(parameters.conv1.bias).to(torch.float)
        conv = nn.Conv2d(
            in_channels=C, out_channels=parameters.conv1.bias.shape[-1], kernel_size=3, stride=1, padding=1
        )
        conv.weight = nn.Parameter(weight)
        conv.bias = nn.Parameter(bias)
        hidden_states = ttnn.to_torch(hidden_states).to(torch.float)
        hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))
        hidden_states = conv(hidden_states)
        hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    else:
        if parameters.conv1.use_split_conv:
            hidden_states = ttnn.to_torch(hidden_states)
            hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))
            hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            hidden_states = run_conv_with_split(
                device,
                hidden_states,
                hidden_states.shape[0],
                parameters,
                kernel_size=3,
                stride=1,
                pad=1,
                split_factor=parameters.conv1.split_factor,
                ttnn_weight=parameters.conv1.weight,
                ttnn_bias=parameters.conv1.bias,
            )
        else:
            hidden_states = run_conv(
                device,
                output_channels=parameters.conv1.bias.shape[-1],
                input_channels=C,
                input_height=H,
                input_width=W,
                filter_height=3,
                stride_h=1,
                pad_h=1,
                tt_input_tensor=hidden_states,
                tt_weight_tensor=parameters.conv1.weight,
                tt_bias_tensor=parameters.conv1.bias,
            )

    if temb is not None:
        temb = ttnn.silu(temb)
        temb = ttnn.linear(
            temb,
            parameters.time_emb_proj.weight,
            bias=parameters.time_emb_proj.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
        )

    if temb is not None and time_embedding_norm == "default":
        temb = ttnn.reshape(temb, (temb.shape[0], temb.shape[1], 1, 1))
        hidden_states = ttnn.add(hidden_states, temb, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(temb)

    N = hidden_states.shape[0]
    C = hidden_states.shape[1]
    H = hidden_states.shape[2]
    W = hidden_states.shape[3]

    if use_torch_gn:
        hidden_states = ttnn.to_torch(hidden_states).to(torch.float)
        torch_weight = ttnn.to_torch(parameters.norm2.weight).to(torch.float)
        torch_bias = ttnn.to_torch(parameters.norm2.bias).to(torch.float)
        hidden_states = (
            torch.nn.functional.group_norm(hidden_states, 32, weight=torch_weight, bias=torch_bias)
            .permute(0, 2, 3, 1)
            .view(N, 1, W * H, C)
        )

        hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    else:
        input_mask_tensor = get_mask_tensor(C, groups, grid_size.y, device)

        input_mask_tensor = ttnn.create_group_norm_input_mask(C, groups, grid_size.y)
        input_mask_tensor = ttnn.from_torch(
            input_mask_tensor,
            dtype=ttnn.DataType.BFLOAT8_B,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gamma_t, beta_t = get_weights(parameters.norm2.weight, parameters.norm2.bias, C, grid_size.y, device)

        hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = ttnn.reshape(hidden_states, (N, 1, W * H, C))

        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_shape = N * H * W // grid_size.x, C // grid_size.y
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR, False)
        sharded_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )

        hidden_states = ttnn.to_memory_config(hidden_states, sharded_mem_config)

        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=groups,
            input_mask=input_mask_tensor,
            weight=gamma_t,
            bias=beta_t,
            epsilon=eps,
            memory_config=sharded_mem_config,
            core_grid=grid_size,
        )
        ttnn.deallocate(input_mask_tensor)

    if non_linearity == "silu":
        if use_torch_silu:
            torch_silu = torch.nn.SiLU()
            hidden_states = ttnn.to_torch(hidden_states)
            hidden_states = torch_silu(hidden_states)
            hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        else:
            hidden_states = get_inputs(device, hidden_states, grid_size)
            hidden_states = ttnn.silu(hidden_states)

    hidden_states = ttnn.reshape(hidden_states, (N, H, W, C))
    batch_size = hidden_states.shape[0]

    if use_torch_conv:
        weight = ttnn.to_torch(parameters.conv2.weight).to(torch.float)
        bias = ttnn.to_torch(parameters.conv2.bias).to(torch.float)
        conv = nn.Conv2d(
            in_channels=C, out_channels=parameters.conv2.bias.shape[-1], kernel_size=3, stride=1, padding=1
        )
        conv.weight = nn.Parameter(weight)
        conv.bias = nn.Parameter(bias)
        hidden_states = ttnn.to_torch(hidden_states).to(torch.float)
        hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))
        hidden_states = conv(hidden_states)
        hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    else:
        if parameters.conv2.use_split_conv:
            hidden_states = ttnn.to_torch(hidden_states)
            hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))
            hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            hidden_states = run_conv_with_split(
                device,
                hidden_states,
                hidden_states.shape[0],
                parameters,
                kernel_size=3,
                stride=1,
                pad=1,
                split_factor=parameters.conv2.split_factor,
                ttnn_weight=parameters.conv2.weight,
                ttnn_bias=parameters.conv2.bias,
            )
        else:
            hidden_states = run_conv(
                device,
                output_channels=parameters.conv2.bias.shape[-1],
                input_channels=C,
                input_height=H,
                input_width=W,
                filter_height=3,
                stride_h=1,
                pad_h=1,
                tt_input_tensor=hidden_states,
                tt_weight_tensor=parameters.conv2.weight,
                tt_bias_tensor=parameters.conv2.bias,
            )

    if "conv_shortcut" in parameters:
        if use_torch_conv:
            input_tensor = ttnn.to_torch(input_tensor).to(torch.float)
            weight = ttnn.to_torch(parameters.conv_shortcut.weight).to(torch.float)
            bias = ttnn.to_torch(parameters.conv_shortcut.bias).to(torch.float)
            conv = nn.Conv2d(
                in_channels=C, out_channels=parameters.conv_shortcut.bias.shape[-1], kernel_size=1, stride=1
            )
            conv.weight = nn.Parameter(weight)
            conv.bias = nn.Parameter(bias)
            input_tensor = conv(input_tensor)
            input_tensor = ttnn.from_torch(input_tensor, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        else:
            if parameters.conv_shortcut.use_split_conv:
                input_tensor = run_conv_with_split(
                    device,
                    input_tensor,
                    input_tensor.shape[0],
                    parameters,
                    kernel_size=1,
                    stride=1,
                    pad=0,
                    split_factor=parameters.conv_shortcut.split_factor,
                    ttnn_weight=parameters.conv_shortcut.weight,
                    ttnn_bias=parameters.conv_shortcut.bias,
                )
            else:
                input_tensor = ttnn.permute(input_tensor, (0, 2, 3, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
                input_tensor = run_conv(
                    device,
                    output_channels=parameters.conv_shortcut.bias.shape[-1],
                    input_channels=C,
                    input_height=H,
                    input_width=W,
                    filter_height=1,
                    stride_h=1,
                    pad_h=0,
                    tt_input_tensor=input_tensor,
                    tt_weight_tensor=parameters.conv_shortcut.weight,
                    tt_bias_tensor=parameters.conv_shortcut.bias,
                )
    output_tensor = ttnn.add(input_tensor, hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.mul(output_tensor, (1 / output_scale_factor), memory_config=ttnn.L1_MEMORY_CONFIG)

    return output_tensor


def sd_cross_attention_down_blocks2d(
    hidden_states,
    temb=None,
    encoder_hidden_states=None,
    timestep=None,
    attention_mask=None,
    cross_attention_kwargs=None,
    encoder_attention_mask=None,
    additional_residuals=None,
    config=None,
    conv_shortcut=True,
    use_torch_conv=False,
    class_labels=None,
    add_downsample=False,
    return_dict=None,
    attention_head_dim=None,
    num_layers=None,
    norm_num_groups=32,
    transformer_layers_per_block=10,
    device=None,
    parameters=None,
):
    output_states = ()
    for index, (resnet, attn) in enumerate(zip(parameters.resnets, parameters.attentions)):
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = ResnetBlock2D(
            config,
            input_tensor=hidden_states,
            temb=temb,
            parameters=resnet,
            device=device,
            use_torch_conv=use_torch_conv,
        )
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = sd_transformer_2d(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            parameters=attn,
            device=device,
            timestep=timestep,
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=return_dict,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            norm_num_groups=norm_num_groups,
            attention_mask=attention_mask,
            config=config,
            eps=1e-06,
        )

    if add_downsample:
        hidden_states = sd_downsample_2(hidden_states, parameters.downsamplers[0], device)

    return hidden_states
