# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn


def run_conv(
    device,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    stride_h,
    pad_h,
    tt_input_tensor=None,
    tt_weight_tensor=None,
    tt_bias_tensor=None,
    math_fidelity=ttnn.MathFidelity.LoFi,
    activations_dtype=ttnn.bfloat8_b,
    weights_dtype=ttnn.bfloat8_b,
    use_1d_systolic_array=True,
    config_override=None,
    use_shallow_conv_variant=False,
    dilation=1,
    fp32_accum=False,
    packer_l1_acc=False,
    output_layout=ttnn.TILE_LAYOUT,
    deallocate_activation=False,
    debug=False,
    groups=1,
    shard_layout=None,
    auto_shard=False,
):
    batch_size = tt_input_tensor.shape[0]
    total_batch_size = batch_size

    # memory_config = (None

    # tt_weight_tensor = ttnn.to_dtype(tt_weight_tensor, ttnn.bfloat8_b)
    tt_weight_tensor = ttnn.to_torch(tt_weight_tensor)
    tt_weight_tensor = ttnn.from_torch(tt_weight_tensor, dtype=ttnn.float32)
    tt_bias_tensor = ttnn.to_torch(tt_bias_tensor)
    tt_bias_tensor = ttnn.from_torch(tt_bias_tensor, dtype=ttnn.float32)
    # tt_weight_tensor = ttnn.from_device(tt_weight_tensor)
    tt_input_tensor = ttnn.to_layout(tt_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    # tt_input_tensor = ttnn.from_device(tt_input_tensor)
    # tt_input_tensor = ttnn.to_dtype(tt_input_tensor, ttnn.bfloat8_b)

    # tt_bias_tensor = ttnn.to_dtype(tt_bias_tensor, ttnn.bfloat8_b)
    # tt_bias_tensor = ttnn.from_device(tt_bias_tensor)
    # tt_input_tensor = ttnn.from_device(tt_input_tensor)
    tt_bias_tensor = ttnn.reshape(tt_bias_tensor, (1, 1, 1, tt_bias_tensor.shape[0]))
    print("Shapes :", tt_weight_tensor, " ", tt_bias_tensor, " ", tt_input_tensor)
    torch.manual_seed(0)

    reader_patterns_cache = {}

    if shard_layout is None and not auto_shard:
        shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if use_1d_systolic_array else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        shard_layout=shard_layout,
        input_channels_alignment=(
            16 if use_shallow_conv_variant or (input_channels == 16 and input_height == 115) else 32
        ),
        deallocate_activation=deallocate_activation,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        output_layout=output_layout,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=packer_l1_acc,
    )
    if config_override and "act_block_h" in config_override and not auto_shard:
        conv_config.act_block_h_override = config_override["act_block_h"]

    if config_override and "act_block_w_div" in config_override and not auto_shard:
        conv_config.act_block_w_div = config_override["act_block_w_div"]

    if config_override and "num_cores_nhw" in config_override:
        if config_override["num_cores_nhw"] == 98:
            conv_config.core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (11, 7)), ttnn.CoreRange((0, 8), (1, 8))})
            conv_config.override_sharding_config = True
            print("Setting num_cores_nhw to 98")

    [tt_output_tensor_on_device, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=(filter_height, filter_height),
        stride=(stride_h, stride_h),
        padding=(pad_h, pad_h),
        dilation=(dilation, dilation),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache=reader_patterns_cache,
        debug=debug,
        groups=groups,
        # memory_config=memory_config,
        return_weights_and_bias=True,
        return_output_dim=True,
    )

    conv_op = ttnn.reshape()
    tt_output_tensor_on_device = ttnn.reshape(
        tt_output_tensor_on_device, (N, out_height, out_width, tt_output_tensor_on_device.shape[-1])
    )
    return tt_output_tensor_on_device


def get_mask_tensor(C, groups, grid_size, device):
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, groups, grid_size)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return input_mask_tensor


def get_weights(weight, bias, C, grid_size, device):
    gamma = ttnn.create_group_norm_weight_bias_rm(ttnn.to_torch(weight), C, grid_size)
    beta = ttnn.create_group_norm_weight_bias_rm(ttnn.to_torch(bias), C, grid_size)

    gamma_t = ttnn.from_torch(
        gamma,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t = ttnn.from_torch(
        beta,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return gamma_t, beta_t


def ResnetBlock2D(
    conifg,
    input_tensor=None,
    temb=None,
    in_channels=None,
    input_height=None,
    input_width=None,
    parameters=None,
    device=None,
    eps=1e-5,
    groups=32,
    time_embedding_norm="default",
    non_linearity="silu",
    conv_shortcut=False,
    output_scale_factor=1.0,
):
    out_mem_config_l1 = ttnn.L1_MEMORY_CONFIG
    hidden_states = input_tensor
    N = hidden_states.shape[0]
    batch_size = N
    C = in_channels
    H = input_height
    W = input_width
    hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))
    hidden_states = ttnn.reshape(hidden_states, (batch_size, 1, input_width * input_height, in_channels))

    grid_size = ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x)

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

    if non_linearity == "silu":
        hidden_states = ttnn.silu(hidden_states)

    hidden_states = ttnn.reshape(hidden_states, (N, H, W, C))
    batch_size = hidden_states.shape[0]
    """
    weight = ttnn.to_torch(parameters.conv1.weight).to(torch.float)
    bias = ttnn.to_torch(parameters.conv1.bias).to(torch.float)
    conv = nn.Conv2d(in_channels = C, out_channels = parameters.conv1.bias.shape[-1], kernel_size = 3, stride = 1, padding = 1)
    conv.weight = nn.Parameter(weight)
    conv.bias = nn.Parameter(bias)
    hidden_states = ttnn.to_torch(hidden_states).to(torch.float)
    hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))
    hidden_states = conv(hidden_states)
    hidden_states = ttnn.from_torch(hidden_states, device = device, dtype = ttnn.bfloat16,layout = ttnn.TILE_LAYOUT )
    """

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
        hidden_states = ttnn.add(hidden_states, temb)

    N = hidden_states.shape[0]
    C = hidden_states.shape[1]
    H = hidden_states.shape[2]
    W = hidden_states.shape[3]

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

    hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))
    hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
    hidden_states = ttnn.reshape(hidden_states, (N, 1, W * H, C))
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

    if non_linearity == "silu":
        hidden_states = ttnn.silu(hidden_states)

    hidden_states = ttnn.reshape(hidden_states, (N, H, W, C))
    batch_size = hidden_states.shape[0]
    weight = ttnn.to_torch(parameters.conv2.weight).to(torch.float)
    bias = ttnn.to_torch(parameters.conv2.bias).to(torch.float)
    conv = nn.Conv2d(in_channels=C, out_channels=parameters.conv2.bias.shape[-1], kernel_size=3, stride=1, padding=1)
    conv.weight = nn.Parameter(weight)
    conv.bias = nn.Parameter(bias)
    hidden_states = ttnn.to_torch(hidden_states).to(torch.float)
    hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))
    hidden_states = conv(hidden_states)
    hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    if conv_shortcut:
        input_tensor = ttnn.to_torch(input_tensor).to(torch.float)
        weight = ttnn.to_torch(parameters.conv_shortcut.weight).to(torch.float)
        bias = ttnn.to_torch(parameters.conv_shortcut.bias).to(torch.float)
        conv = nn.Conv2d(in_channels=C, out_channels=parameters.conv_shortcut.bias.shape[-1], kernel_size=1, stride=1)
        conv.weight = nn.Parameter(weight)
        conv.bias = nn.Parameter(bias)
        input_tensor = conv(input_tensor)
        input_tensor = ttnn.from_torch(input_tensor, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    output_tensor = ttnn.add(input_tensor, hidden_states)
    output_tensor = ttnn.mul(output_tensor, (1 / output_scale_factor))

    return output_tensor
