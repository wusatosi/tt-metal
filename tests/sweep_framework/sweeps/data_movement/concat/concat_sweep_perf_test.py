# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import random
import ttnn

from tests.ttnn.utils_for_testing import (
    check_with_pcc,
    start_measuring_time,
    stop_measuring_time,
    get_per_core_size_and_num_cores,
)
from models.utility_functions import torch_random
import pytest

# Override the default timeout in seconds for hang detection.
TIMEOUT = 20
random.seed(0)

# List[Tensor] tensors = [<[1, 100, 14, 14]>, <[1, 100, 14, 14]>],
# int dim = 1
# List[Tensor] tensors = [<[1, 1056, 7, 7]>, <[1, 48, 7, 7]>],
# int dim = 1

parameters = {
    "nightly": {
        "concat_specs": [
            # {'dim': 1, 'shapes': [[1, 192, 28, 28]]},
            {"dim": 1, "shapes": [[1, 192, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28]]}
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

    return False, None


def run(
    concat_specs,
    dtype,
    layout,
    *,
    device,
) -> list:
    device.enable_async(False)
    torch_input_tensors = [torch_random(shape, -0.1, 0.1, dtype=torch.bfloat16) for shape in concat_specs["shapes"]]
    torch_output_tensor = torch.concat(torch_input_tensors, dim=concat_specs["dim"])

    ttnn_input_tensors = [
        ttnn.from_torch(torch_input_tensor, device=device, layout=layout, dtype=dtype)
        for torch_input_tensor in torch_input_tensors
    ]
    start_time = start_measuring_time()
    result_tensor = ttnn.concat(ttnn_input_tensors, dim=concat_specs["dim"])
    e2e_perf = stop_measuring_time(start_time)
    output_tensor = ttnn.to_torch(result_tensor)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]


@pytest.mark.parametrize("concat_spec", parameters["nightly"]["concat_specs"])
@pytest.mark.parametrize("dtype", parameters["nightly"]["dtype"])
@pytest.mark.parametrize("layout", parameters["nightly"]["layout"])
def test_concat_pytorch2(concat_spec, dtype, layout, device):
    shapes = concat_spec["shapes"]
    dim = concat_spec["dim"]
    device.enable_async(False)
    if dtype == ttnn.bfloat16 and any([shape[-1] % 2 != 0 for shape in shapes]) and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Skipping test for RM bfloat16 with odd last dimension")

    torch_input_tensors = [torch_random(shape, -0.1, 0.1, dtype=torch.bfloat16) for shape in shapes]
    torch_output_tensor = torch.cat(torch_input_tensors, dim=dim)

    ttnn_input_tensors = [
        ttnn.from_torch(torch_input_tensor, device=device, layout=layout, dtype=dtype)
        for torch_input_tensor in torch_input_tensors
    ]

    start_time = start_measuring_time()
    result_tensor = ttnn.concat(ttnn_input_tensors, dim=dim)
    e2e_perf = stop_measuring_time(start_time)
    output_tensor = ttnn.to_torch(result_tensor)
    print(f"{layout} layout: {e2e_perf}")

    assert check_with_pcc(
        torch_output_tensor, output_tensor, 0.999
    ), "Output tensors do not match within the specified precision"
