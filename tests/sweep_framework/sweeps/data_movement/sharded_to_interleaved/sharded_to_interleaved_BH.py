# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import random
import ttnn
import traceback

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 15
# seed for random
random.seed(0)

parameters = {
    "nightly": {
        "shard_specs": [
            {
                "shape": [1, 32, 1280],
                "output_mem_config": {
                    "layout": ttnn.TensorMemoryLayout.INTERLEAVED,
                    "buffer_type": ttnn.BufferType.L1,
                    "shard_spec": None,
                },
                "dtype": ttnn.bfloat16,
            },
            {
                "shape": [1, 1, 32, 32064],
                "output_mem_config": {
                    "layout": ttnn.TensorMemoryLayout.INTERLEAVED,
                    "buffer_type": ttnn.BufferType.L1,
                    "shard_spec": None,
                },
                "dtype": ttnn.bfloat8_b,
            },
            {
                "shape": [1, 1, 32, 6144],
                "output_mem_config": {
                    "layout": ttnn.TensorMemoryLayout.INTERLEAVED,
                    "buffer_type": ttnn.BufferType.L1,
                    "shard_spec": None,
                },
                "dtype": ttnn.bfloat16,
            },
            {
                "shape": [1, 1, 32, 16032],
                "output_mem_config": {
                    "layout": ttnn.TensorMemoryLayout.INTERLEAVED,
                    "buffer_type": ttnn.BufferType.L1,
                    "shard_spec": None,
                },
                "dtype": ttnn.bfloat8_b,
            },
        ],
        "layout": [ttnn.TILE_LAYOUT],
    }
}


def run(
    shard_specs,
    layout,
    *,
    device,
):
    device.enable_async(False)

    shape = shard_specs["shape"]
    output_mem_config = shard_specs["output_mem_config"]

    # Parse memory configuration parameters
    mem_layout = output_mem_config["layout"]
    buffer_type = output_mem_config["buffer_type"]
    shard_spec = output_mem_config["shard_spec"]

    # Create the memory config
    memory_config = ttnn.MemoryConfig(
        mem_layout,
        buffer_type,
        shard_spec,
    )
    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))  # Define ranges
    grid = ttnn.CoreRangeSet({core_range})

    shard_spec = ttnn.ShardSpec(
        grid,  # Grid of shards
        shape[-2:],  # Shape of each shard
        ttnn.ShardOrientation.ROW_MAJOR,  # Shard orientation (ROW_MAJOR, COL_MAJOR)
        0,  # Halo size (set to 0 if unused)
        ttnn.ShardMode.PHYSICAL,  # Mode of sharding
    )
    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )

    # Create a random tensor of the specified shape
    torch.manual_seed(0)
    input_data = torch.randn(shape, dtype=torch.bfloat16)
    sharded_data = ttnn.from_torch(
        input_data,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_mem_config,
        dtype=ttnn.bfloat8_b,
    )

    # Measure performance of the sharded-to-interleaved operation
    start_time = start_measuring_time()

    # Use the pybind-defined function to convert sharded to interleaved
    interleaved_data = ttnn.sharded_to_interleaved(
        input_tensor=sharded_data,
        memory_config=memory_config,
        output_dtype=dtype,
        queue_id=0,
        is_l1_aligned=False,
    )

    e2e_perf = stop_measuring_time(start_time)

    output_data = ttnn.from_device(interleaved_data)
    output_data = ttnn.to_torch(output_data)

    # Compare the tensors and return performance and accuracy check
    result = check_with_pcc(input_data, output_data, 0.999)
    return [result, e2e_perf]
