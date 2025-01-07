# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

import ttnn

from loguru import logger
from models.utility_functions import is_grayskull, is_blackhole, torch_random
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal
from models.utility_functions import skip_for_grayskull, skip_for_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [256])
def test_untilize(device, n, c, h, w):
    torch.manual_seed(2005)
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.transpose(2, 3)
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_output_tensor = ttnn.to_layout(tt_input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert_with_pcc(torch_input_tensor, tt_output_tensor, 0.9999)
