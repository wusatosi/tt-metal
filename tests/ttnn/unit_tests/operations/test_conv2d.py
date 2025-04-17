# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
import torch
import math


def prepare_weight_bias(input_tensor, input_params):
    weight = torch.rand(input_params[3], input_tensor.shape[1], input_params[1], input_params[1])
    bias = torch.rand(input_params[3])
    bias = torch.reshape(bias, (1, 1, 1, -1))
    weight = ttnn.from_torch(weight, dtype=ttnn.float32)
    bias = ttnn.from_torch(bias, dtype=ttnn.float32)
    return weight, bias


class Conv2D:
    def __init__(
        self,
        input_params,
        parameters,
        device,
        batch_size,
        groups=1,
        dilation=1,
        act_block_h=False,
        block_shard=False,
        deallocate_activation=False,
        output_layout=ttnn.TILE_LAYOUT,
        width_shard=False,
        act_blocks=32,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        reshard_if_not_optimal=True,
        use_shallow_covariant=False,
        activation_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ):
        self.device = device
        self.parameters = parameters
        self.activation_dtype = activation_dtype
        self.input_params = input_params
        self.groups = groups
        self.dilation = dilation
        self.act_block_h = act_block_h
        self.block_shard = block_shard
        self.deallocate_activation = deallocate_activation
        self.output_layout = output_layout
        self.width_shard = width_shard
        self.act_blocks = act_blocks
        self.enable_act_double_buffer = enable_act_double_buffer
        self.enable_split_reader = enable_split_reader
        self.reshard_if_not_optimal = reshard_if_not_optimal
        self.batch_size = batch_size
        self.shard_layout = shard_layout
        if self.block_shard:
            self.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        if self.width_shard:
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED

        self.use_shallow_covariant = use_shallow_covariant
        self.conv_config = self._initialize_conv_config()
        self.compute_config = self._initialize_compute_config()
        self.weights, self.bias = self.parameters

    def _initialize_conv_config(self):
        conv_config = ttnn.Conv2dConfig(
            dtype=self.activation_dtype,
            weights_dtype=ttnn.bfloat8_b,
            activation="",
            shard_layout=self.shard_layout,
            input_channels_alignment=16 if self.use_shallow_covariant else 32,
            act_block_w_div=1,
            transpose_shards=False,
            deallocate_activation=self.deallocate_activation,
            enable_act_double_buffer=self.enable_act_double_buffer,
            enable_split_reader=self.enable_split_reader,
            enable_subblock_padding=False,
            output_layout=self.output_layout,
            reallocate_halo_output=False,
            reshard_if_not_optimal=self.reshard_if_not_optimal,
        )

        if self.act_block_h:
            conv_config.act_block_h_override = self.act_blocks

        if self.block_shard:
            conv_config.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

        return conv_config

    def _initialize_compute_config(self):
        return ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def __call__(self, x):
        input_height = x.shape[1]
        input_width = x.shape[2]

        [x, [h, w], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weights,
            in_channels=x.shape[3],
            out_channels=self.input_params[3],
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=(self.input_params[0], self.input_params[0]),
            stride=(self.input_params[1], self.input_params[1]),
            padding=(self.input_params[2], self.input_params[2]),
            dilation=(self.dilation, self.dilation),
            batch_size=self.batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=self.groups,
            return_weights_and_bias=True,
            return_output_dim=True,
        )

        return x, h, w


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_tensor",
    [torch.rand((1, 320, 7, 7))],
    ids=["input_tensor1"],
)
def test_conv_bs(device, input_tensor, reset_seeds):
    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn_input.reshape(
        1, 1, ttnn_input.shape[0] * ttnn_input.shape[1] * ttnn_input.shape[2], ttnn_input.shape[3]
    )
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    input_params = [1, 1, 0, 1280]
    parameters = prepare_weight_bias(input_tensor, input_params)
    conv = Conv2D(
        input_params,
        parameters,
        device,
        batch_size=1,
        block_shard=True,
    )
    output = conv(ttnn_input)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_tensor",
    [torch.rand((1, 320, 7, 7))],
    ids=["input_tensor1"],
)
def test_conv_ws(device, input_tensor, reset_seeds):
    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn_input.reshape(
        1, 1, ttnn_input.shape[0] * ttnn_input.shape[1] * ttnn_input.shape[2], ttnn_input.shape[3]
    )
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    input_params = [1, 1, 0, 1280]
    parameters = prepare_weight_bias(input_tensor, input_params)
    conv = Conv2D(input_params, parameters, device, batch_size=1, width_shard=True)
    output = conv(ttnn_input)
