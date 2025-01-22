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


def get_mask_tensor(C, groups, grid_size, device):
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, groups, grid_size)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return input_mask_tensor


def get_weights(weight, bias, C, grid_size, device):
    gamma = ttnn.create_group_norm_weight_bias_rm(ttnn.to_torch(weight), C, grid_size)
    beta = ttnn.create_group_norm_weight_bias_rm(ttnn.to_torch(bias), C, grid_size)

    gamma_t = ttnn.from_torch(
        gamma,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t = ttnn.from_torch(
        beta,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return gamma_t, beta_t


def get_inputs(device, input_tensor, grid_size):
    ncores = 32
    interleaved_mem_config = ttnn.L1_MEMORY_CONFIG
    input_tensor = ttnn.to_memory_config(input_tensor, interleaved_mem_config)

    input_2d_height = input_tensor.shape.with_tile_padding()[2]
    input_2d_width = input_tensor.shape.with_tile_padding()[3]
    shard_strategy = ttnn.ShardStrategy.HEIGHT

    ## input shard

    if shard_strategy == ttnn.ShardStrategy.BLOCK:
        input_2d_height_padded = _nearest_y(input_2d_height, grid_size[0] * 32)
        shard_height = math.ceil(input_2d_height_padded / grid_size[0])
        shard_width = math.ceil(input_2d_width / grid_size[1])
        shard_orientation = ttnn.ShardOrientation.COL_MAJOR
        tensor_memory_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1),
                )
            }
        )
    elif shard_strategy == ttnn.ShardStrategy.HEIGHT:
        input_2d_height_padded = _nearest_y(input_2d_height, ncores * 32)
        shard_height = math.ceil(input_2d_height_padded / ncores)
        shard_grid = get_shard_grid_from_num_cores(ncores, device)
        shard_width = input_2d_width
        shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
        tensor_memory_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    elif shard_strategy == ttnn.ShardStrategy.WIDTH:
        shard_height = input_2d_height
        input_2d_width_padded = _nearest_y(input_2d_width, ncores * 32)
        shard_width = math.ceil(input_2d_width_padded / ncores)
        shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
        tensor_memory_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        shard_grid = get_shard_grid_from_num_cores(ncores, device)

    shard_spec = ttnn.ShardSpec(shard_grid, (shard_height, shard_width), shard_orientation, False)
    in_sharded_mem_config = ttnn.MemoryConfig(tensor_memory_layout, ttnn.BufferType.L1, shard_spec)

    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=in_sharded_mem_config)
    return input_tensor


def update_params(parameters):
    down_block = {
        (0, 0): {
            "split_conv_1": True,
            "conv1_split_factor": 4,
            "split_conv_2": True,
            "conv2_split_factor": 4,
            "conv_shortcut": False,
        },
        (0, 1): {
            "split_conv_1": True,
            "conv1_split_factor": 4,
            "split_conv_2": True,
            "conv2_split_factor": 4,
            "conv_shortcut": False,
        },
        (1, 0): {
            "split_conv_1": False,
            "conv1_split_factor": 4,
            "split_conv_2": True,
            "conv2_split_factor": 2,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 4,
        },
        (1, 1): {
            "split_conv_1": True,
            "conv1_split_factor": 2,
            "split_conv_2": True,
            "conv2_split_factor": 2,
            "conv_shortcut": False,
        },
        (2, 0): {
            "split_conv_1": True,
            "conv1_split_factor": 4,
            "split_conv_2": True,
            "conv2_split_factor": 8,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 8,
        },
        (2, 1): {
            "split_conv_1": True,
            "conv1_split_factor": 8,
            "split_conv_2": True,
            "conv2_split_factor": 8,
            "conv_shortcut": False,
        },
    }

    mid_block = {
        (0, 0): {
            "split_conv_1": True,
            "conv1_split_factor": 8,
            "split_conv_2": True,
            "conv2_split_factor": 8,
            "conv_shortcut": False,
        },
        (1, 0): {
            "split_conv_1": True,
            "conv1_split_factor": 8,
            "split_conv_2": True,
            "conv2_split_factor": 8,
            "conv_shortcut": False,
        },
    }

    up_block = {
        (0, 0): {
            "split_conv_1": True,
            "conv1_split_factor": 16,
            "split_conv_2": True,
            "conv2_split_factor": 8,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 4,
        },
        (0, 1): {
            "split_conv_1": True,
            "conv1_split_factor": 16,
            "split_conv_2": True,
            "conv2_split_factor": 8,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 4,
        },
        (0, 2): {
            "split_conv_1": True,
            "conv1_split_factor": 16,
            "split_conv_2": True,
            "conv2_split_factor": 16,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 16,
        },
        (1, 0): {
            "split_conv_1": True,
            "conv1_split_factor": 12,
            "split_conv_2": True,
            "conv2_split_factor": 4,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 4,
        },
        (1, 1): {
            "split_conv_1": True,
            "conv1_split_factor": 8,
            "split_conv_2": True,
            "conv2_split_factor": 4,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 4,
        },
        (1, 2): {
            "split_conv_1": True,
            "conv1_split_factor": 4,
            "split_conv_2": True,
            "conv2_split_factor": 4,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 8,
        },
        (2, 0): {
            "split_conv_1": True,
            "conv1_split_factor": 24,
            "split_conv_2": True,
            "conv2_split_factor": 4,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 4,
        },
        (2, 1): {
            "split_conv_1": True,
            "conv1_split_factor": 8,
            "split_conv_2": True,
            "conv2_split_factor": 4,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 8,
        },
        (2, 2): {
            "split_conv_1": True,
            "conv1_split_factor": 8,
            "split_conv_2": True,
            "conv2_split_factor": 4,
            "conv_shortcut": True,
            "split_conv_3": True,
            "conv3_split_factor": 8,
        },
    }

    for k, v in down_block.items():
        index1 = k[0]
        index2 = k[1]
        parameters.down_blocks[index1].resnets[index2].conv1["use_split_conv"] = down_block[k]["split_conv_1"]
        parameters.down_blocks[index1].resnets[index2].conv1["split_factor"] = down_block[k]["conv1_split_factor"]
        parameters.down_blocks[index1].resnets[index2].conv2["use_split_conv"] = down_block[k]["split_conv_2"]
        parameters.down_blocks[index1].resnets[index2].conv2["split_factor"] = down_block[k]["conv2_split_factor"]
        parameters.down_blocks[index1].resnets[index2].conv2["conv_shortcut"] = False
        if down_block[k]["conv_shortcut"]:
            parameters.down_blocks[index1].resnets[index2].conv2["conv_shortcut"] = True
            parameters.down_blocks[index1].resnets[index2].conv_shortcut["use_split_conv"] = down_block[k][
                "split_conv_3"
            ]
            parameters.down_blocks[index1].resnets[index2].conv_shortcut["split_factor"] = down_block[k][
                "conv3_split_factor"
            ]

    for k, v in mid_block.items():
        index1 = k[0]
        index2 = k[1]
        parameters.mid_block.resnets[index1].conv1["use_split_conv"] = mid_block[k]["split_conv_1"]
        parameters.mid_block.resnets[index1].conv1["split_factor"] = mid_block[k]["conv1_split_factor"]
        parameters.mid_block.resnets[index1].conv2["use_split_conv"] = mid_block[k]["split_conv_2"]
        parameters.mid_block.resnets[index1].conv2["split_factor"] = mid_block[k]["conv2_split_factor"]
        parameters.mid_block.resnets[index1].conv2["conv_shortcut"] = False
        if mid_block[k]["conv_shortcut"]:
            parameters.mid_block.resnets[index1].conv2["conv_shortcut"] = True
            parameters.mid_block.resnets[index1].conv_shortcut["use_split_conv"] = mid_block[k]["split_conv_3"]
            parameters.mid_block.resnets[index1].conv_shortcut["split_factor"] = mid_block[k]["conv3_split_factor"]

    for k, v in up_block.items():
        index1 = k[0]
        index2 = k[1]
        parameters.up_blocks[index1].resnets[index2].conv1["use_split_conv"] = up_block[k]["split_conv_1"]
        parameters.up_blocks[index1].resnets[index2].conv1["split_factor"] = up_block[k]["conv1_split_factor"]
        parameters.up_blocks[index1].resnets[index2].conv2["use_split_conv"] = up_block[k]["split_conv_2"]
        parameters.up_blocks[index1].resnets[index2].conv2["split_factor"] = up_block[k]["conv2_split_factor"]
        parameters.up_blocks[index1].resnets[index2].conv2["conv_shortcut"] = False
        if up_block[k]["conv_shortcut"]:
            parameters.up_blocks[index1].resnets[index2].conv2["conv_shortcut"] = True
            parameters.up_blocks[index1].resnets[index2].conv_shortcut["use_split_conv"] = up_block[k]["split_conv_3"]
            parameters.up_blocks[index1].resnets[index2].conv_shortcut["split_factor"] = up_block[k][
                "conv3_split_factor"
            ]

    return parameters
