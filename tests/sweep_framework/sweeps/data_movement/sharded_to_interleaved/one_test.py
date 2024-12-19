# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import random
import ttnn
import traceback

from framework.device_fixtures import default_device
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
                    "layout": "TensorMemoryLayout::INTERLEAVED",
                    "buffer_type": "BufferType::L1",
                    "shard_spec": None,
                },
                "output_dtype": "DataType::BFLOAT16",
            },
        ],
        "dtype": [ttnn.bfloat16],
        "layout": [ttnn.TILE_LAYOUT],
    }
}


shard_specs = {
    "shape": [1, 32, 1280],
    "output_mem_config": {
        "layout": "TensorMemoryLayout::INTERLEAVED",
        "buffer_type": "BufferType::L1",
        "shard_spec": None,
    },
    "output_dtype": "DataType::BFLOAT16",
}

device = default_device()
device.enable_async(False)

shape = shard_specs["shape"]
output_mem_config = shard_specs["output_mem_config"]

# Parse memory configuration parameters
mem_layout = output_mem_config["layout"]
buffer_type = output_mem_config["buffer_type"]
shard_spec = output_mem_config["shard_spec"]
output_dtype = ttnn.bfloat16

# Create the memory config
memory_config = ttnn.create_interleaved_memory_config(
    layout=mem_layout,
    buffer_type=buffer_type,
    shard_spec=shard_spec,
)

# Create a random tensor of the specified shape
torch.manual_seed(0)
input_data = torch.randn(shape, dtype=torch.bfloat16)
sharded_data = ttnn.from_torch(
    input_data,
    device=device,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.L1_MEMORY_CONFIG,
    dtype=ttnn.bfloat16,
)

# Measure performance of the sharded-to-interleaved operation
start_time = start_measuring_time()

# Use the pybind-defined function to convert sharded to interleaved
interleaved_data = ttnn.operations.data_movement.sharded_to_interleaved(
    input_tensor=sharded_data,
    memory_config=memory_config,
    output_dtype=output_dtype,
    queue_id=0,
    is_l1_aligned=False,
)

e2e_perf = stop_measuring_time(start_time)

output_data = ttnn.from_device(interleaved_data)
output_data = ttnn.to_torch(output_data)

# Compare the tensors and return performance and accuracy check
result = check_with_pcc(input_data, output_data, 0.999)
print([result, e2e_perf])
