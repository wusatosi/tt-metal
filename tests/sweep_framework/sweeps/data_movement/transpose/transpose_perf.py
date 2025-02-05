# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import random
import ttnn

from typing import Optional, Tuple

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


TIMEOUT = 30
random.seed(0)


parameters = {
    "NC": {
        "N": [1, 17, 49, 95, 127],
        "C": [1, 17, 49, 95, 127],
        "H": [1, 32, 47, 65, 100],
        "W": [1, 32, 47, 65, 100],
        "dims_string": ["NC"],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "memory_config_str": ["L1", "DRAM"],
    },
    "NH": {
        "N": [1, 17, 49, 95, 127],
        "C": [1, 32, 47, 65, 100],
        "H": [1, 17, 49, 95, 127],
        "W": [1, 32, 47, 65, 100],
        "dims_string": ["NH"],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "memory_config_str": ["L1", "DRAM"],
    },
    "NW": {
        "N": [1, 17, 49, 95, 127],
        "C": [1, 32, 47, 65, 100],
        "H": [1, 32, 47, 65, 100],
        "W": [1, 17, 49, 95, 127],
        "dims_string": ["NW"],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "memory_config_str": ["L1", "DRAM"],
    },
    "CH": {
        "N": [1, 32, 47, 65, 100],
        "C": [1, 17, 49, 95, 127],
        "H": [1, 17, 49, 95, 127],
        "W": [1, 32, 47, 65, 100],
        "dims_string": ["CH"],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "memory_config_str": ["L1", "DRAM"],
    },
    "CW": {
        "N": [1, 32, 47, 65, 100],
        "C": [1, 17, 49, 95, 127],
        "H": [1, 32, 47, 65, 100],
        "W": [1, 17, 49, 95, 127],
        "dims_string": ["CW"],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "memory_config_str": ["L1", "DRAM"],
    },
    "HW": {
        "N": [1, 32, 47, 65, 100],
        "C": [1, 32, 47, 65, 100],
        "H": [1, 17, 49, 95, 127],
        "W": [1, 17, 49, 95, 127],
        "dims_string": ["HW"],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "memory_config_str": ["L1", "DRAM"],
    },
}

dims_map = {
    "NC": {"dim0": 0, "dim1": 1},
    "NH": {"dim0": 0, "dim1": 2},
    "NW": {"dim0": 0, "dim1": 3},
    "CH": {"dim0": 1, "dim1": 2},
    "CW": {"dim0": 1, "dim1": 3},
    "HW": {"dim0": 2, "dim1": 3},
}

memory_config_map = {
    "L1": ttnn.L1_MEMORY_CONFIG,
    "DRAM": ttnn.DRAM_MEMORY_CONFIG,
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    return False, None


def run(
    N,
    C,
    H,
    W,
    dims_string,
    layout,
    memory_config_str,
    *,
    device,
):
    device.enable_async(False)
    torch_input_tensor = torch_random([N, C, H, W], -0.1, 0.1, dtype=torch.bfloat16)  # returns to torch tensor
    memory_config = memory_config_map[memory_config_str]

    dims = dims_map[dims_string]

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor, device=device, dtype=ttnn.bfloat16, layout=layout, memory_config=memory_config
    )

    start_time = start_measuring_time()
    ttnn_output = ttnn.transpose(ttnn_input_tensor, dims["dim0"], dims["dim1"])
    e2e_perf = stop_measuring_time(start_time)

    return [(True, 1.0), e2e_perf]
