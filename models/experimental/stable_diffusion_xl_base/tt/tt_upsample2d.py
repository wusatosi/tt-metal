# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn

from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_split_conv_params,
    split_conv2d,
)


class TtUpsample2D(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        stride,
        padding,
        dilation,
        groups,
    ):
        super().__init__()

        self.device = device

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.scale_factor = 2  # fixed number for now

        weights = state_dict[f"{module_path}.conv.weight"]
        bias = state_dict[f"{module_path}.conv.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # self.compute_config, self.conv_config, self.tt_weights, self.tt_bias, self.conv_params = prepare_conv_params(
        #     device, weights, bias, ttnn.bfloat8_b
        # )
        (
            self.compute_config,
            self.conv_config,
            self.tt_weights,
            self.tt_bias,
            self.conv_params,
        ) = prepare_split_conv_params(device, weights, bias, 2, 2, ttnn.bfloat8_b, act_block_h_override=32)

    def interpolate(self, hidden_states):
        hidden_states = ttnn.upsample(hidden_states, (self.scale_factor, self.scale_factor))
        B, H, W, C = list(hidden_states.shape)
        return hidden_states, [B, C, H, W]

    def forward(self, hidden_states):
        hidden_states, input_shape = self.interpolate(hidden_states)
        B, C, H, W = input_shape

        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.move(hidden_states)
        hidden_states, [C, H, W], [d_w, d_b] = split_conv2d(
            device=self.device,
            hidden_states=hidden_states,
            input_shape=[B, C, H, W],
            conv_weights=self.tt_weights,
            conv_bias=self.tt_bias,
            split_in=2,
            split_out=2,
            compute_config=self.compute_config,
            conv_config=self.conv_config,
            conv_params=self.conv_params,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        self.tt_weights = d_w
        self.tt_bias = d_b

        self.conv_config.preprocess_weights_on_device = False
        self.conv_config.always_preprocess_weights = False

        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        return hidden_states, [C, H, W]
