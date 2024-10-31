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
    "nightly": {
        "shape": [
            # [1, 1, 64, 64],
            # [1, 1, 128, 128],
            # [1, 1, 192, 192],
            # [1, 1, 256, 256],
            # [1, 1, 320, 320],
            # [1, 1, 384, 384],
            # [1, 1, 448, 448],
            # [1, 1, 512, 512],
            # [1, 1, 576, 576],
            # [1, 1, 640, 640],
        ],
        "layout": [ttnn.TILE_LAYOUT],
        "dtype": [ttnn.bfloat16],
        "input_core_grid": [ttnn.CoreGrid(x=1, y=1)],
        "input_shard_strategy": [ttnn.ShardStrategy.BLOCK],
        "output_core_grid": [ttnn.CoreGrid(x=2, y=2), ttnn.CoreGrid(x=1, y=4), ttnn.CoreGrid(x=4, y=1)],
        "output_shard_strategy": [ttnn.ShardStrategy.BLOCK],
    }
}

for i in range(1, 20):
    for j in range(1, 20):
        parameters["nightly"]["shape"].append([1, 1, i * 64, j * 64])


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if (test_vector["shape"][3] // test_vector["input_core_grid"].x) % 32 != 0:
        return (
            True,
            f"shape: {test_vector['shape']} cannot be mapped to input core grid {test_vector['input_core_grid']}",
        )
    if (test_vector["shape"][3] // test_vector["output_core_grid"].x) % 32 != 0:
        return (
            True,
            f"shape: {test_vector['shape']} cannot be mapped to output core grid {test_vector['output_core_grid']}",
        )
    if (test_vector["shape"][2] // test_vector["input_core_grid"].y) % 32 != 0:
        return (
            True,
            f"shape: {test_vector['shape']} cannot be mapped to input core grid {test_vector['input_core_grid']}",
        )
    if (test_vector["shape"][2] // test_vector["output_core_grid"].y) % 32 != 0:
        return (
            True,
            f"shape: {test_vector['shape']} cannot be mapped to output core grid {test_vector['output_core_grid']}",
        )
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
