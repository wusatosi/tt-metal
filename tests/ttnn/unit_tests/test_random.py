# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout
from models.utility_functions import is_grayskull, is_blackhole, torch_random, skip_for_grayskull


@pytest.mark.parametrize("shape", [[1, 9, 91, 7, 9]])
def test_to_layout(shape, device):
    torch.manual_seed(2005)

    torch_tensor = torch.rand(shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.to_torch(output_tensor)

    torch_output = torch_tensor
    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 0.9999)
