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
                "shard_shape": [32, 128],
                "output_mem_config": {
                    "layout": "TensorMemoryLayout::HEIGHT_SHARDED",
                    "buffer_type": "BufferType::L1",
                    "shard_spec": {
                        "grid": [[{"x": 0, "y": 0}, {"x": 0, "y": 0}]],
                        "shape": [32, 128],
                        "orientation": "ShardOrientation::ROW_MAJOR",
                        "halo": 0,
                        "mode": "ShardMode::PHYSICAL",
                        "physical_shard_shape": None,
                    },
                },
            },
        ],
        "strategy": [ttnn.ShardStrategy.HEIGHT],
        "orientation": [ttnn.ShardOrientation.ROW_MAJOR],
        "core_grid": [ttnn.CoreGrid(y=1, x=1)],
        "dtype": [ttnn.bfloat16],
        "layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_buffer_type": [ttnn.L1_MEMORY_CONFIG],
        "output_buffer_type": [ttnn.L1_MEMORY_CONFIG],
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
    shard_shape = shard_specs["shard_shape"]
    output_mem_config = shard_specs["output_mem_config"]

    # Parse memory configuration parameters
    mem_layout = output_mem_config["layout"]
    buffer_type = output_mem_config["buffer_type"]
    shard_spec = output_mem_config["shard_spec"]
    shard_grid = shard_spec["grid"]
    shard_shape = shard_spec["shape"]
    shard_orientation = shard_spec["orientation"]

    # Create the memory config using pybind-defined function
    shard_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=core_grid,
        strategy=strategy,
        orientation=orientation,
        use_height_and_width_as_shard_shape=True,
    )

    # Create a random tensor of the specified shape
    torch.manual_seed(0)
    input_data = torch.randn(shape, dtype=torch.bfloat16)
    interleaved_data = ttnn.from_torch(
        input_data,
        device=device,
        layout=layout,
        memory_config=input_buffer_type,
        dtype=ttnn.bfloat16,
    )

    # Measure performance of the interleaved-to-sharded operation
    start_time = start_measuring_time()

    # Use the pybind-defined function to convert interleaved to sharded
    sharded_data = ttnn.operations.data_movement.interleaved_to_sharded(
        input_tensor=interleaved_data,
        grid=ttnn.CoreGrid(*shard_grid[0][0].values()),
        shard_shape=shard_shape,
        shard_scheme=mem_layout,
        shard_orientation=shard_orientation,
        output_dtype=dtype,
        queue_id=0,
        keep_l1_aligned=False,
    )

    # Convert back to interleaved for validation
    interleaved_output = ttnn.to_memory_config(sharded_data, output_buffer_type)

    e2e_perf = stop_measuring_time(start_time)

    output_data = ttnn.from_device(interleaved_output)
    output_data = ttnn.to_torch(output_data)

    # Compare the concatenated tensors and return performance and accuracy check
    result = check_with_pcc(input_data, output_data, 0.999)
    return [result, e2e_perf]
