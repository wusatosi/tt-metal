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
                "shape": [1, 32, 128],
            }
        ],
    }
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT:
        if test_vector["dtype"] == ttnn.bfloat8_b:
            return True, "bfloat8_b not supported with ROW_MAJOR_LAYOUT"

    return False, None


def run(
    shard_specs,
    strategy,
    orientation,
    core_grid,
    dtype,
    layout,
    input_buffer_type,
    output_buffer_type,
    *,
    device,
):
    device.enable_async(False)

    shape = shard_specs["shape"]

    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))  # Define ranges
    grid = ttnn.CoreRangeSet({core_range})

    shard_spec = ttnn.ShardSpec(
        grid,  # Grid of shards
        shape[-2:],  # Shape of each shard
        ttnn.ShardOrientation.ROW_MAJOR,  # Shard orientation (ROW_MAJOR, COL_MAJOR)
        0,  # Halo size (set to 0 if unused)
        ttnn.ShardMode.PHYSICAL,  # Mode of sharding
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED,
        ttnn.BufferType.L1,
        None,
    )

    # Create a random tensor of the specified shape
    torch.manual_seed(0)
    input_data = torch.randn(shape, dtype=torch.bfloat16)
    interleaved_data = ttnn.from_torch(
        input_data,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_mem_config,
        dtype=ttnn.bfloat16,
    )

    # Measure performance of the interleaved-to-sharded operation
    start_time = start_measuring_time()

    # Use the pybind-defined function to convert interleaved to sharded
    sharded_data = ttnn.interleaved_to_sharded(
        input_tensor=interleaved_data,
        grid=grid,
        shard_shape=shape[-2:],
        shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        output_dtype=ttnn.bfloat16,
        queue_id=0,
        keep_l1_aligned=False,
    )

    # Convert back to interleaved for validation
    # interleaved_output = ttnn.to_memory_config(sharded_data, output_buffer_type)

    e2e_perf = stop_measuring_time(start_time)

    output_data = ttnn.from_device(sharded_data)
    output_data = ttnn.to_torch(output_data)

    # Compare the concatenated tensors and return performance and accuracy check
    result = check_with_pcc(input_data, output_data, 0.999)
    return [result, e2e_perf]
