# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn
from models.demos.stable_diffusion_xl_turbo.tt.tt_resnet_block_2d import get_weights, get_mask_tensor


def preprocess_conv_parameter_resnet(parameter, *, dtype):
    while len(parameter.shape) < 4:
        parameter = parameter.unsqueeze(0)
    parameter = ttnn.from_torch(parameter, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    return parameter


def custom_preprocessor_resnet(model, name):
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


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype)
    return parameter


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, nn.Conv2d):
        parameters["weight"] = preprocess_conv_parameter(model.weight, dtype=ttnn.float32)
        bias = model.bias.reshape((1, 1, 1, -1))
        parameters["bias"] = preprocess_conv_parameter(bias, dtype=ttnn.float32)

    if isinstance(model, (nn.Linear, nn.LayerNorm)):
        weight = model.weight.T.contiguous()
        while len(weight.shape) < 4:
            weight = weight.unsqueeze(0)
        parameters["weight"] = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if model.bias is not None:
            bias = model.bias
            while len(bias.shape) < 4:
                bias = bias.unsqueeze(0)
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    return parameters
