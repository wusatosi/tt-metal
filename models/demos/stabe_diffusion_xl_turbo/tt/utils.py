# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import ttnn
from models.demos.stabe_diffusion_xl_turbo.tt.resnetblock2d_utils import get_weights, get_mask_tensor


def preprocess_conv_parameter(parameter, *, dtype):
    while len(parameter.shape) < 4:
        parameter = parameter.unsqueeze(0)
    parameter = ttnn.from_torch(parameter, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    return parameter


def custom_preprocessor(model, name):
    parameters = {}

    # if isinstance(model, nn.Conv2d):
    # weight = torch.permute(model.weight, (2, 3, 0, 1))
    # parameters["weight"] = preprocess_conv_parameter(weight, dtype=ttnn.bfloat8_b)
    # parameters["bias"] = preprocess_conv_parameter(model.bias, dtype=ttnn.bfloat8_b)
    if isinstance(model, nn.GroupNorm):
        weight = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
        bias = ttnn.from_torch(model.bias, dtype=ttnn.bfloat16)
        grid_size = ttnn.CoreGrid(y=4, x=8)
        parameters["weight"], parameters["bias"] = weight, bias
        parameters["tt_weight"], parameters["tt_bias"] = get_weights(weight, bias, model.num_channels, grid_size.y)
        parameters["input_mask_tensor"] = get_mask_tensor(model.num_channels, model.num_groups, grid_size.y)
    return parameters
