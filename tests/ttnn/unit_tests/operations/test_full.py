# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn as nn
import ttnn
from models.utility_functions import comp_allclose
from loguru import logger

from tests.ttnn.utils_for_testing import assert_equal, tt_dtype_to_torch_dtype


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 2],  # single tile with rank 1
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [0],
)
@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
def test_full_float(device, input_shape, fill_value, tt_dtype):
    torch_any = torch.zeros(input_shape, dtype=torch.bfloat16)
    any = ttnn.from_torch(torch_any, device=device, layout=ttnn.TILE_LAYOUT)
    tt_output = ttnn.moreh_full(input_shape, fill_value, any, dtype=tt_dtype)
    tt_output_cpu = ttnn.to_torch(tt_output, dtype=torch.float32)

    print(tt_output_cpu * float(pow(2, 126)))
