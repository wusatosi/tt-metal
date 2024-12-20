# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("shape", [(32 * 100, 32 * 10), (32 * 1, 32 * 1), (32 * 17, 32 * 163)])
def test_mul_add_2D_tensors(device, shape, use_program_cache):
    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_c = torch.rand(shape, dtype=torch.bfloat16)

    torch_output_tensor = torch.add(torch.mul(torch_input_tensor_a, torch_input_tensor_b), torch_input_tensor_c)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_c = ttnn.from_torch(torch_input_tensor_c, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.mul_add(input_tensor_a, input_tensor_b, input_tensor_c)
    output = ttnn.to_torch(output)
    assert_with_pcc(torch_output_tensor, output, 0.9999)

    # test program cache
    output = ttnn.mul_add(input_tensor_a, input_tensor_b, input_tensor_c)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)
