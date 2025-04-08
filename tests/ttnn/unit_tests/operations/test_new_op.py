# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_my_op(mesh_device):
    input_tensor_shape = (128, 128)

    torch_input_tensor1 = torch.ones(input_tensor_shape) * 7
    tensor1 = ttnn.from_torch(torch_input_tensor1, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    torch_input_tensor2 = torch.ones(input_tensor_shape) * 2
    tensor2 = ttnn.from_torch(torch_input_tensor2, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    scalar = 2

    torch_output_tensor = torch.sqrt(torch_input_tensor1 + torch_input_tensor2) * scalar

    ttnn_output = ttnn.my_new_op(tensor1, tensor2, scalar)
    ttnn_output_tensor = ttnn.to_torch(ttnn_output)

    valid_pcc = 0.95
    print(f"ttnn_output_tensor: {ttnn_output_tensor}")
    print(f"torch_output_tensor: {torch_output_tensor}")
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output_tensor, torch_output_tensor, pcc=valid_pcc)
    print(pcc_message)
    print(f"pcc_passed: {pcc_passed}")
