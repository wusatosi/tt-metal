# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn


def run_conv_with_split(
    device,
    input_tensor,
    batch_size,
    parameters,
    kernel_size,
    stride,
    pad,
    weights_dtype=ttnn.bfloat8_b,
    split_factor=2,
):
    input_channels = input_tensor.shape[1]
    assert input_channels % split_factor == 0
    split_input_channels = input_channels // split_factor
    ttnn_weight = parameters.conv.weight
    ttnn_bias = parameters.conv.bias
    split_input_tensors = ttnn.split(input_tensor, 2, 1, memory_config=ttnn.L1_MEMORY_CONFIG)
    split_weight_tensors = ttnn.split(ttnn_weight, 2, 1, memory_config=ttnn.L1_MEMORY_CONFIG)
    reader_patterns_cache = {}

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=weights_dtype,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    torch_output_tensor = None
    tt_weight_tensor = split_weight_tensors
    out_channels = tt_weight_tensor[1].shape[0]
    for i in range(split_factor):
        tt_input_tensor = ttnn.permute(split_input_tensors[i], (0, 2, 3, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        [tt_output_tensor_on_device, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv2d(
            input_tensor=tt_input_tensor,
            weight_tensor=tt_weight_tensor[i],
            in_channels=split_input_channels,
            out_channels=tt_weight_tensor[i].shape[0],
            device=device,
            bias_tensor=ttnn_bias,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(pad, pad),
            batch_size=batch_size,
            input_height=tt_input_tensor.shape[1],
            input_width=tt_input_tensor.shape[2],
            conv_config=conv_config,
            compute_config=compute_config,
            conv_op_cache=reader_patterns_cache,
            return_output_dim=True,
            return_weights_and_bias=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        torch_conv_output_tensor = ttnn.reshape(
            tt_output_tensor_on_device,
            (batch_size, out_height, out_width, out_channels),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        torch_conv_output_tensor = ttnn.sharded_to_interleaved(torch_conv_output_tensor)
        torch_conv_output_tensor = ttnn.permute(
            torch_conv_output_tensor, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        if i == 0:
            torch_output_tensor = torch_conv_output_tensor
        else:
            torch_output_tensor = ttnn.add(
                torch_output_tensor, torch_conv_output_tensor, memory_config=ttnn.L1_MEMORY_CONFIG
            )

    return torch_output_tensor


def downsample(input_tensor, parameters, device):
    tt_output_tensor_on_device = run_conv_with_split(
        device,
        input_tensor,
        batch_size=1,
        parameters=parameters,
        kernel_size=3,
        stride=2,
        pad=1,
        weights_dtype=ttnn.bfloat8_b,
        split_factor=2,
    )

    return tt_output_tensor_on_device
