# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize("height", [16])
@pytest.mark.parametrize("width", [32])
def test_example(device, height, width):
    torch.manual_seed(0)

    torch_input_tensor1 = torch.ones((height, width), dtype=torch.bfloat16) * 13
    torch_input_tensor2 = torch.ones((height, width), dtype=torch.bfloat16) * 12
    torch_output_tensor = torch_input_tensor1 + torch_input_tensor2
    print(f"torch_input_tensor1: {torch_input_tensor1}")
    print(f"torch_input_tensor2: {torch_input_tensor2}")
    input_tensor1 = ttnn.from_torch(
        torch_input_tensor1, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device
    )
    input_tensor2 = ttnn.from_torch(
        torch_input_tensor2, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device
    )
    print(f"input_tensor1: {input_tensor1}")
    print(f"input_tensor2: {input_tensor2}")
    multiplier = 3.1
    output_tensor = ttnn.my_new_op(input_tensor1, input_tensor2, multiplier)
    output_tensor = ttnn.to_torch(output_tensor)
    print(f"output_tensor: {output_tensor}")
    assert_equal(torch_output_tensor, output_tensor)
