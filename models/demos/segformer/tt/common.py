# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class Conv:
    def __init__(
        self,
        conv_params,
        parameters,
        *,
        act_block_h=None,
        reshard=False,
        deallocate=True,
        height_sharding=True,
        activation="",
        groups=1,
        dtype=ttnn.bfloat8_b,
    ) -> None:
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]

        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.conv_params = conv_params
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.deallocate = deallocate
        self.activation = activation
        self.groups = groups
        self.dtype = dtype
        # output_shape = ttnn.get_conv_output_dim(
        #     input_shape=(1, 1, 1, 1),
        #     kernel_size=self.kernel_size,
        #     stride=(self.conv_params[0], self.conv_params[1]),
        #     padding=(self.conv_params[2], self.conv_params[3]),
        # )
        # output_shape_nhw = output_shape[0] * output_shape[1] * output_shape[1]
        # core_count_height =
        self.shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )

    def __call__(self, device, input_tensor):
        batch_size = input_tensor.shape[0]
        input_height = input_tensor.shape[1]
        input_width = input_tensor.shape[2]

        output_shape = [
            batch_size,
            ttnn.get_conv_output_dim(input_height, self.kernel_size[0], self.conv_params[0], self.conv_params[2]),
            ttnn.get_conv_output_dim(input_width, self.kernel_size[1], self.conv_params[1], self.conv_params[3]),
            self.out_channels,
        ]
        output_shape_nhw = output_shape[0] * output_shape[1] * output_shape[2]
        core_count_height = output_shape_nhw // 32
        core_count_block = (min(output_shape_nhw // 32, 8)) * (output_shape[3] // 32)

        self.shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            if core_count_height >= core_count_block
            else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        print(f"core_count_height --> {core_count_height}, core_count_block --> {core_count_block}")

        conv_config = ttnn.Conv2dConfig(
            dtype=self.dtype,
            weights_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.LoFi,
            activation=self.activation,
            shard_layout=self.shard_layout,
            math_approx_mode_enabled=True,
            fp32_dest_acc_enabled=False,
            packer_l1_accum_enabled=False,
            input_channels_alignment=16 if input_tensor.shape[3] < 16 else 32,
            transpose_shards=False,
            reshard_if_not_optimal=self.reshard,
            deallocate_activation=self.deallocate,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_split_reader=False,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        # print(input_tensor)
        # print(ttnn.get_memory_config(input_tensor))

        if input_tensor.shape[3] != 3:
            input_tensor = ttnn.reshape(
                input_tensor,
                (1, 1, input_tensor.shape[0] * input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]),
            )
        # print(input_tensor.shape)
        # print(f"input shape --> {input_tensor.shape}")
        [output_tensor, _out_height, _out_width, self.weights, self.bias] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=input_tensor.shape[3],
            out_channels=self.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=(self.conv_params[0], self.conv_params[1]),
            padding=(self.conv_params[2], self.conv_params[3]),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=conv_config,
            groups=self.groups,
        )
        # print(f"output shape --> {output_tensor.shape}")
        # print(f"output config --> {ttnn.get_memory_config(output_tensor)}")
        return output_tensor, _out_height, _out_width
