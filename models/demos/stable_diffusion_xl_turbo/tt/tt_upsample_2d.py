# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def conv(
    device,
    input_tensor,
    batch_size,
    parameters,
    stride,
    pad,
    kernel_size,
    weights_dtype=ttnn.bfloat8_b,
    deallocate_activation=False,
    math_fidelity=ttnn.MathFidelity.LoFi,
    fp32_accum=False,
    dilation=1,
    packer_l1_acc=False,
    config_override=None,
    auto_shard=False,
):
    tt_weight = parameters.conv.weight
    tt_bias = parameters.conv.bias
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=None,
        input_channels_alignment=32,
        deallocate_activation=deallocate_activation,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        output_layout=ttnn.TILE_LAYOUT,
        reallocate_halo_output=True,
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

    tt_output_tensor_on_device, [out_height, out_width] = ttnn.conv2d(
        input_tensor=input_tensor,
        weight_tensor=tt_weight,
        in_channels=input_tensor.shape[-1],
        out_channels=tt_weight.shape[0],
        device=device,
        bias_tensor=tt_bias,
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(pad, pad),
        dilation=(dilation, dilation),
        batch_size=batch_size,
        input_height=input_tensor.shape[1],
        input_width=input_tensor.shape[2],
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache={},
        debug=False,
        groups=1,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        return_weights_and_bias=False,
        return_output_dim=True,
    )
    return tt_output_tensor_on_device, [out_height, out_width]


def upsample(input_tensor, parameters, device):
    input_tensor = ttnn.upsample(input_tensor, scale_factor=2, memory_config=ttnn.L1_MEMORY_CONFIG)

    output_tensor, [out_height, out_width] = conv(
        device,
        input_tensor,
        batch_size=1,
        parameters=parameters,
        stride=1,
        pad=1,
        kernel_size=3,
        weights_dtype=ttnn.bfloat8_b,
        deallocate_activation=True,
    )
    return output_tensor, [out_height, out_width]
