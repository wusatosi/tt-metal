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
from models.demos.stable_diffusion_xl_turbo.tt.tt_resnet_block_2d import ResnetBlock2D


def up_block_2d(
    device, parameters, config, input, temb, input_tuple, in_channels, input_height, input_width, num_layers=3
):
    hidden_states = input
    input_tuple = input_tuple[-1]

    for i in range(num_layers):
        if i > 0:
            hidden_states = ttnn.to_torch(hidden_states)
            hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device)
            # hidden_states = ttnn.to_dtype(hidden_states, dtype = ttnn.bfloat16)

        hidden_states = ttnn.concat([hidden_states, input_tuple], dim=1)

        hidden_states = ResnetBlock2D(
            config=config,
            input_tensor=hidden_states,
            temb=temb,
            parameters=parameters.resnets[i],
            device=device,
        )

    return hidden_states
