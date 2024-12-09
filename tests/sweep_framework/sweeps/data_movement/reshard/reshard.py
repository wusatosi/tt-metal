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
DATUM_SIZE = {ttnn.bfloat16: 2, ttnn.bfloat8_b: 1, ttnn.float32: 4}
DTYPES = [ttnn.bfloat16]

general_parameters = {
    "input_shard_strategy": [ttnn.ShardStrategy.BLOCK, ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.HEIGHT],
    "output_shard_strategy": [ttnn.ShardStrategy.BLOCK, ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.HEIGHT],
    "input_shard_orientation": [ttnn.ShardOrientation.ROW_MAJOR],
    "output_shard_orientation": [ttnn.ShardOrientation.ROW_MAJOR],
}


def get_shapes():
    return [[1, 1, j * TILE_SIZE, i * TILE_SIZE] for i in range(1, 250) for j in range(1, 250)]


def get_parameters(start_x=1, end_x=9, start_y=1):
    shapes = get_shapes()

    parameters = {}
    for [x_in, y_in] in itertools.product(range(1, 9), repeat=2):
        for [x_out, y_out] in itertools.product(range(1, 9), repeat=2):
            if x_in < start_x or x_in >= end_x:
                continue
            if y_in < start_y:
                continue
            if (x_in, y_in) == (x_out, y_out):
                continue
            print(f"{x_in}x{y_in} {x_out}x{y_out}")
            for dtype in DTYPES:
                suite_name = f"{x_in}x{y_in}-{x_out}x{y_out}-{dtype}"
                parameters[suite_name] = {}
                parameters[suite_name].update(general_parameters)
                parameters[suite_name]["dtype"] = [dtype]
                parameters[suite_name]["input_core_grid"] = [ttnn.CoreGrid(x=x_in, y=y_in)]
                parameters[suite_name]["output_core_grid"] = [ttnn.CoreGrid(x=x_out, y=y_out)]
                parameters[suite_name]["shape"] = shapes
    return parameters


def is_tilable(shape, coregrid, shard_strategy):
    if shard_strategy == ttnn.ShardStrategy.BLOCK:
        return (shape[3] // TILE_SIZE) % coregrid.x == 0 and (shape[2] // TILE_SIZE) % coregrid.y == 0
    elif shard_strategy == ttnn.ShardStrategy.HEIGHT:
        return (shape[2] // TILE_SIZE) % (coregrid.x * coregrid.y) == 0
    elif shard_strategy == ttnn.ShardStrategy.WIDTH:
        return (shape[3] // TILE_SIZE) % (coregrid.x * coregrid.y) == 0

    raise RuntimeError(f"Unsupported shard strategy {shard_strategy}")


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if not is_tilable(test_vector["shape"], test_vector["input_core_grid"], test_vector["input_shard_strategy"]):
        return True, "Input tensor cannot be tiled"

    if not is_tilable(test_vector["shape"], test_vector["output_core_grid"], test_vector["output_shard_strategy"]):
        return True, "Output tensor cannot be tiled"

    if (
        test_vector["input_shard_strategy"] == test_vector["output_shard_strategy"]
        and test_vector["input_shard_strategy"] != ttnn.ShardStrategy.BLOCK
    ):
        return True, "ignore"

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


def run_chunk(vec_chunk, device):
    results = []
    for vec in vec_chunk:
        res = run(**vec, device=device)
        results.append(res)
    return results


def run(
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

    sharded_input_tensor = ttnn.zeros(
        shape, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_shard_memory_config, dtype=dtype
    )

    # reshard
    for i in range(10):
        try:
            sharded_output_tensor = ttnn.to_memory_config(sharded_input_tensor, output_shard_memory_config)
        except Exception as e:
            ttnn.deallocate(sharded_input_tensor)
            raise e
        ttnn.deallocate(sharded_output_tensor)

    ttnn.deallocate(sharded_input_tensor)

    return [True, ""]
