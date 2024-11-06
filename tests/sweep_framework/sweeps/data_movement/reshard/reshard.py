# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import random
import ttnn
import sys

from typing import Optional, Tuple

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


TIMEOUT = 15

parameters = {
    "down-square-0": {
        "shape": [],
        "layout": [ttnn.TILE_LAYOUT],
        "dtype": [ttnn.bfloat16],
        "input_core_grid": [ttnn.CoreGrid(x=4, y=4)],
        "input_shard_strategy": [ttnn.ShardStrategy.BLOCK],
        "output_core_grid": [ttnn.CoreGrid(x=2, y=2)],
        "output_shard_strategy": [ttnn.ShardStrategy.BLOCK],
    },
    "down-square-1": {
        "shape": [],
        "layout": [ttnn.TILE_LAYOUT],
        "dtype": [ttnn.bfloat16],
        "input_core_grid": [ttnn.CoreGrid(x=4, y=4)],
        "input_shard_strategy": [ttnn.ShardStrategy.BLOCK],
        "output_core_grid": [ttnn.CoreGrid(x=3, y=3)],
        "output_shard_strategy": [ttnn.ShardStrategy.BLOCK],
    },
    "down-rect-0": {
        "shape": [],
        "layout": [ttnn.TILE_LAYOUT],
        "dtype": [ttnn.bfloat16],
        "input_core_grid": [ttnn.CoreGrid(x=4, y=4)],
        "input_shard_strategy": [ttnn.ShardStrategy.BLOCK],
        "output_core_grid": [
            ttnn.CoreGrid(x=1, y=4),
            ttnn.CoreGrid(x=4, y=1),
        ],
        "output_shard_strategy": [ttnn.ShardStrategy.BLOCK],
    },
    "down-rect-1": {
        "shape": [],
        "layout": [ttnn.TILE_LAYOUT],
        "dtype": [ttnn.bfloat16],
        "input_core_grid": [ttnn.CoreGrid(x=4, y=4)],
        "input_shard_strategy": [ttnn.ShardStrategy.BLOCK],
        "output_core_grid": [
            ttnn.CoreGrid(x=2, y=4),
            ttnn.CoreGrid(x=4, y=2),
        ],
        "output_shard_strategy": [ttnn.ShardStrategy.BLOCK],
    },
    "down-rect-2": {
        "shape": [],
        "layout": [ttnn.TILE_LAYOUT],
        "dtype": [ttnn.bfloat16],
        "input_core_grid": [ttnn.CoreGrid(x=4, y=4)],
        "input_shard_strategy": [ttnn.ShardStrategy.BLOCK],
        "output_core_grid": [
            ttnn.CoreGrid(x=3, y=4),
            ttnn.CoreGrid(x=4, y=3),
        ],
        "output_shard_strategy": [ttnn.ShardStrategy.BLOCK],
    },
    "up-square-0": {
        "shape": [],
        "layout": [ttnn.TILE_LAYOUT],
        "dtype": [ttnn.bfloat16],
        "input_core_grid": [ttnn.CoreGrid(x=4, y=4)],
        "input_shard_strategy": [ttnn.ShardStrategy.BLOCK],
        "output_core_grid": [
            ttnn.CoreGrid(x=5, y=5),
        ],
        "output_shard_strategy": [ttnn.ShardStrategy.BLOCK],
    },
    "up-square-1": {
        "shape": [],
        "layout": [ttnn.TILE_LAYOUT],
        "dtype": [ttnn.bfloat16],
        "input_core_grid": [ttnn.CoreGrid(x=4, y=4)],
        "input_shard_strategy": [ttnn.ShardStrategy.BLOCK],
        "output_core_grid": [ttnn.CoreGrid(x=6, y=6)],
        "output_shard_strategy": [ttnn.ShardStrategy.BLOCK],
    },
    "up-square-2": {
        "shape": [],
        "layout": [ttnn.TILE_LAYOUT],
        "dtype": [ttnn.bfloat16],
        "input_core_grid": [ttnn.CoreGrid(x=4, y=4)],
        "input_shard_strategy": [ttnn.ShardStrategy.BLOCK],
        "output_core_grid": [ttnn.CoreGrid(x=7, y=7)],
        "output_shard_strategy": [ttnn.ShardStrategy.BLOCK],
    },
    "up-square-3": {
        "shape": [],
        "layout": [ttnn.TILE_LAYOUT],
        "dtype": [ttnn.bfloat16],
        "input_core_grid": [ttnn.CoreGrid(x=4, y=4)],
        "input_shard_strategy": [ttnn.ShardStrategy.BLOCK],
        "output_core_grid": [ttnn.CoreGrid(x=8, y=8)],
        "output_shard_strategy": [ttnn.ShardStrategy.BLOCK],
    },
    "up-rect-0": {
        "shape": [],
        "layout": [ttnn.TILE_LAYOUT],
        "dtype": [ttnn.bfloat16],
        "input_core_grid": [ttnn.CoreGrid(x=4, y=4)],
        "input_shard_strategy": [ttnn.ShardStrategy.BLOCK],
        "output_core_grid": [
            ttnn.CoreGrid(x=5, y=4),
            ttnn.CoreGrid(x=4, y=5),
        ],
        "output_shard_strategy": [ttnn.ShardStrategy.BLOCK],
    },
    "up-rect-1": {
        "shape": [],
        "layout": [ttnn.TILE_LAYOUT],
        "dtype": [ttnn.bfloat16],
        "input_core_grid": [ttnn.CoreGrid(x=4, y=4)],
        "input_shard_strategy": [ttnn.ShardStrategy.BLOCK],
        "output_core_grid": [
            ttnn.CoreGrid(x=6, y=4),
            ttnn.CoreGrid(x=4, y=6),
        ],
        "output_shard_strategy": [ttnn.ShardStrategy.BLOCK],
    },
    "up-rect-2": {
        "shape": [],
        "layout": [ttnn.TILE_LAYOUT],
        "dtype": [ttnn.bfloat16],
        "input_core_grid": [ttnn.CoreGrid(x=4, y=4)],
        "input_shard_strategy": [ttnn.ShardStrategy.BLOCK],
        "output_core_grid": [
            ttnn.CoreGrid(x=7, y=4),
            ttnn.CoreGrid(x=4, y=7),
        ],
        "output_shard_strategy": [ttnn.ShardStrategy.BLOCK],
    },
    "up-rect-3": {
        "shape": [],
        "layout": [ttnn.TILE_LAYOUT],
        "dtype": [ttnn.bfloat16],
        "input_core_grid": [ttnn.CoreGrid(x=4, y=4)],
        "input_shard_strategy": [ttnn.ShardStrategy.BLOCK],
        "output_core_grid": [
            ttnn.CoreGrid(x=8, y=4),
            ttnn.CoreGrid(x=4, y=8),
        ],
        "output_shard_strategy": [ttnn.ShardStrategy.BLOCK],
    },
}

TILE_SIZE = 32

for s, p in parameters.items():
    for i in range(p["input_core_grid"][0].x, 200, p["input_core_grid"][0].x):
        for j in range(p["input_core_grid"][0].y, 200, p["input_core_grid"][0].y):
            p["shape"].append([1, 1, j * TILE_SIZE, i * TILE_SIZE])


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if (test_vector["shape"][3] // test_vector["input_core_grid"].x) % TILE_SIZE != 0:
        return (
            True,
            f"shape: {test_vector['shape']} cannot be mapped to input core grid {test_vector['input_core_grid']}",
        )
    if (test_vector["shape"][3] // test_vector["output_core_grid"].x) % TILE_SIZE != 0:
        return (
            True,
            f"shape: {test_vector['shape']} cannot be mapped to output core grid {test_vector['output_core_grid']}",
        )
    if (test_vector["shape"][2] // test_vector["input_core_grid"].y) % TILE_SIZE != 0:
        return (
            True,
            f"shape: {test_vector['shape']} cannot be mapped to input core grid {test_vector['input_core_grid']}",
        )
    if (test_vector["shape"][2] // test_vector["output_core_grid"].y) % TILE_SIZE != 0:
        return (
            True,
            f"shape: {test_vector['shape']} cannot be mapped to output core grid {test_vector['output_core_grid']}",
        )

    datum_size = {ttnn.bfloat16: 2}
    if (
        datum_size[test_vector["dtype"]] * test_vector["shape"][3] * test_vector["shape"][2]
        > 1024 * 1024 * test_vector["input_core_grid"].x * test_vector["input_core_grid"].y
    ):
        return True, "Insufficient memory in input core grid"
    if (
        datum_size[test_vector["dtype"]] * test_vector["shape"][3] * test_vector["shape"][2]
        > 1024 * 1024 * test_vector["output_core_grid"].x * test_vector["output_core_grid"].y
    ):
        return True, "Insufficient memory in output core grid"
    return False, None


def run(
    shape,
    layout,
    dtype,
    input_core_grid,
    input_shard_strategy,
    output_core_grid,
    output_shard_strategy,
    *,
    device,
):
    torch_dtype = {
        ttnn.bfloat16: torch.bfloat16,
        ttnn.float32: torch.float32,
        ttnn.int32: torch.int32,
    }[dtype]

    input_shard_memory_config = ttnn.create_sharded_memory_config(
        shape, core_grid=input_core_grid, strategy=input_shard_strategy
    )
    output_shard_memory_config = ttnn.create_sharded_memory_config(
        shape, core_grid=output_core_grid, strategy=output_shard_strategy
    )

    torch_input_tensor = torch_random(shape, 0, 256, torch_dtype)
    sharded_input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=layout, device=device, memory_config=input_shard_memory_config
    )

    # reshard
    start_time = start_measuring_time()
    sharded_output_tensor = ttnn.to_memory_config(sharded_input_tensor, output_shard_memory_config)
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = ttnn.to_torch(sharded_output_tensor)

    return [check_with_pcc(torch_input_tensor, output_tensor, 0.999), e2e_perf]
