# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import random
import ttnn
import sys
import itertools
import numpy as np

from typing import Optional, Tuple

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TILE_SIZE = 32
TILE_MULTIPLE = 2
DATUM_SIZE = {ttnn.bfloat16: 2, ttnn.bfloat8_b: 1, ttnn.uint16: 2, ttnn.uint32: 4, ttnn.float32: 4}
DTYPES = [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32, ttnn.uint16, ttnn.uint32]

general_parameters = {
    "layout": [ttnn.TILE_LAYOUT],
    "input_shard_strategy": [ttnn.ShardStrategy.BLOCK, ttnn.ShardStrategy.WIDTH],
    "output_shard_strategy": [ttnn.ShardStrategy.BLOCK, ttnn.ShardStrategy.WIDTH],
    "input_shard_orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
    "output_shard_orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
}


def get_shapes(input_core_grid, output_core_grid):
    shapes = []
    x_lcm = np.lcm(input_core_grid.x, output_core_grid.x)
    y_lcm = np.lcm(input_core_grid.y, output_core_grid.y)
    for i in range(1, 200):
        for j in range(1, 200):
            shapes.append([1, 1, y_lcm * j * TILE_MULTIPLE * TILE_SIZE, x_lcm * i * TILE_MULTIPLE * TILE_SIZE])

    return shapes


MIN_X_IN = 1  # set manually to avoid duplicate work on restart
MAX_X_IN = 2
parameters = {}
for [x_in, y_in] in itertools.product(range(1, 9), repeat=2):
    for [x_out, y_out] in itertools.product(range(1, 9), repeat=2):
        if x_in < MIN_X_IN or x_in >= MAX_X_IN:
            continue
        if (x_in, y_in) == (x_out, y_out):
            continue
        print(f"{x_in}x{y_in} {x_out}x{y_out}")
        shapes = get_shapes(ttnn.CoreGrid(x=x_in, y=y_in), ttnn.CoreGrid(x=x_out, y=y_out))
        for dtype in DTYPES:
            suite_name = f"{x_in}x{y_in}-{x_out}x{y_out}-{dtype}"
            parameters[suite_name] = {}
            parameters[suite_name].update(general_parameters)
            parameters[suite_name]["dtype"] = [dtype]
            parameters[suite_name]["input_core_grid"] = [ttnn.CoreGrid(x=x_in, y=y_in)]
            parameters[suite_name]["output_core_grid"] = [ttnn.CoreGrid(x=x_out, y=y_out)]
            parameters[suite_name]["shape"] = shapes


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    # won't fit in L1
    if (
        DATUM_SIZE[test_vector["dtype"]] * test_vector["shape"][3] * test_vector["shape"][2]
        > 1024 * 1024 * test_vector["input_core_grid"].x * test_vector["input_core_grid"].y
    ):
        return True, "Will not fit in L1"
    if (
        DATUM_SIZE[test_vector["dtype"]] * test_vector["shape"][3] * test_vector["shape"][2]
        > 1024 * 1024 * test_vector["output_core_grid"].x * test_vector["output_core_grid"].y
    ):
        return True, "Will not fit in L1"
    return False, None


def run(
    layout,
    input_shard_strategy,
    output_shard_strategy,
    input_shard_orientation,
    output_shard_orientation,
    dtype,
    input_core_grid,
    output_core_grid,
    shape,
    *,
    device,
):
    torch_dtype = {
        ttnn.bfloat16: torch.bfloat16,
        ttnn.float32: torch.float32,
        ttnn.uint32: torch.int32,
        ttnn.uint16: torch.int16,
        ttnn.bfloat8_b: torch.bfloat16,
    }[dtype]

    input_shard_memory_config = ttnn.create_sharded_memory_config(
        shape,
        core_grid=input_core_grid,
        strategy=input_shard_strategy,
        orientation=input_shard_orientation,
    )
    output_shard_memory_config = ttnn.create_sharded_memory_config(
        shape,
        core_grid=output_core_grid,
        strategy=output_shard_strategy,
        orientation=output_shard_orientation,
    )

    torch_input_tensor = torch_random(shape, 0, 128, torch_dtype)
    sharded_input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=layout, device=device, memory_config=input_shard_memory_config, dtype=dtype
    )

    # reshard
    start_time = start_measuring_time()
    sharded_output_tensor = ttnn.to_memory_config(sharded_input_tensor, output_shard_memory_config)
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = ttnn.to_torch(sharded_output_tensor).to(torch_dtype)

    return [check_with_pcc(torch_input_tensor, output_tensor, 0.99), e2e_perf]
