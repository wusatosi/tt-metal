# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.stable_diffusion.tt.ttnn_resnetblock2d import ResnetBlock2D
from models.demos.stable_diffusion.tt.resnetblock2d_utils import run_conv_with_split


def down_block_2d(device, parameters, config, input, temb, num_layers=2):
    output_states = ()
    for i in range(num_layers):
        hidden_states = ResnetBlock2D(
            config=config,
            input_tensor=input,
            temb=temb,
            parameters=parameters.resnets[i],
            device=device,
        )
        output_states = output_states + (hidden_states,)
    if hidden_states.is_sharded():
        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
    if hidden_states.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
    hidden_states = run_conv_with_split(
        device,
        hidden_states,
        hidden_states.shape[0],
        parameters=parameters.downsamplers[0],
        kernel_size=3,
        stride=2,
        pad=1,
        # split_factor=2,
        # weights_dtype=ttnn.bfloat8_b,
        ttnn_weight=parameters.downsamplers[0].conv.weight,
        ttnn_bias=parameters.downsamplers[0].conv.bias,
    )

    output_states = output_states + (hidden_states,)
    return hidden_states, output_states
