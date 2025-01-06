# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [32])
@pytest.mark.parametrize("return_outputs", [[True, True]])
def test_example_multiple_return(device, height, width, return_outputs):
    torch.manual_seed(0)

    return_output1, return_output2 = return_outputs

    # run torch
    torch_input_tensor = torch.ones((height, width), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor

    # run TT
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output1, output2 = ttnn.prim.example_multiple_return(
        input_tensor, return_output1=return_output1, return_output2=return_output2
    )
