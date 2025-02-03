# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from torch import nn
import torch


class ttnn_T5LayerNorm:
    def __init__(self, parameters=None, eps=1e-6):
        self.weight = ttnn.to_layout(parameters["weight"], layout=ttnn.TILE_LAYOUT)
        self.variance_epsilon = eps

    def __call__(self, hidden_states):
        squared_hidden_states = ttnn.pow(hidden_states, 2)
        mean_squared_hidden_states = ttnn.mean(
            squared_hidden_states,
            dim=-1,
        )

        variance = mean_squared_hidden_states + self.variance_epsilon
        std = ttnn.rsqrt(variance)
        ttnn.deallocate(variance)
        ttnn.deallocate(mean_squared_hidden_states)

        hidden_states = hidden_states * std
        ttnn.deallocate(std)
        return self.weight * hidden_states
