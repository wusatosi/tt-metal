# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [32])
def test_example(device, height, width):
    torch.manual_seed(0)

    torch_input = torch.rand((height, width), dtype=torch.bfloat16)
    torch_output = torch_input

    input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.prim.example(input)
    output = ttnn.to_torch(output)

    print("output", output[0][:16])

    assert_equal(torch_output, output)
