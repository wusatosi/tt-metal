# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import random
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 10
random.seed(0)

parameters = {
    "nightly": {
        "slice_specs": [
            {"dims": [1, 4], "dim": 1, "start": 0, "end": -1, "step": 4},
        ],
        "dtype": [ttnn.bfloat16],
        "layout": [ttnn.TILE_LAYOUT],
    }
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT:
        if test_vector["dtype"] == ttnn.bfloat8_b:
            return True, "bfloat8_b not supported with ROW_MAJOR_LAYOUT"
    if test_vector["dtype"] == ttnn.bfloat8_b:
        if len(test_vector["slice_specs"]["dims"]) < 2:
            return True, "bfloat8_b not supported with dims  < 2"

    return False, None


def run(
    slice_specs,
    dtype,
    layout,
    *,
    device,
):
    device.enable_async(False)

    dims = slice_specs["dims"]
    dim = slice_specs["dim"]
    start = slice_specs["start"]
    end = slice_specs["end"]
    step = slice_specs.get("step", 1)

    tensor = torch_random(dims, -0.1, 0.1, dtype=torch.bfloat16)
    # Create a slice object
    slice_obj = slice(start, end, step)

    # Prepare indices for slicing in the specified dimension
    indices = [slice(None)] * len(dims)  # By default, select all elements along every dimension
    indices[dim] = slice_obj  # Apply slicing to the target dimension
    indices = tuple(indices)

    # Apply slicing to the input_tensor
    torch_output_tensor = tensor[indices]

    ttnn_tensor = ttnn.from_torch(tensor, device=device, layout=layout, dtype=dtype)

    start_time = start_measuring_time()
    ttnn_output = ttnn_tensor[indices]
    e2e_perf = stop_measuring_time(start_time)
    ttnn_output_tensor = ttnn.to_torch(ttnn_output)
    return [check_with_pcc(torch_output_tensor, ttnn_output_tensor, 0.999), e2e_perf]
