# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from typing import Optional
import ttnn

from models.experimental.functional_stable_diffusion3_5.ttnn.common import Conv


class ttnnUpsample2D:
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        kernel_size: Optional[int] = None,
        padding=1,
        interpolate=True,
        parameters=None,
    ):
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.interpolate = interpolate

        conv = None
        if kernel_size is None:
            kernel_size = 3
        conv = Conv([1, 1, padding, padding], height_sharding=False, act_block_h=62, parameters=parameters.conv)

        self.conv = conv

    def __call__(self, hidden_states: torch.Tensor, output_size: Optional[int] = None) -> torch.Tensor:
        # assert hidden_states.shape[1] == self.channels
        device = hidden_states.device()
        if self.interpolate:
            scale_factor = (
                2 if output_size is None else max([f / s for f, s in zip(output_size, hidden_states.shape[-2:])])
            )

            hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
            if output_size is None:
                hidden_states = ttnn.upsample(hidden_states, scale_factor=(2, 2), mode="nearest")
            else:
                hidden_states = ttnn.interpolate(hidden_states, size=(output_size, output_size), mode="nearest")
            hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)

        if self.use_conv:
            input_height, input_width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states = self.conv(device, hidden_states)
            hidden_states = ttnn.reshape(
                hidden_states, (hidden_states.shape[0], input_height, input_width, hidden_states.shape[3])
            )

        return hidden_states
