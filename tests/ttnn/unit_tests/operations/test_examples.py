# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize("input_shape", [(1, 1)])
def test_noc_inline_dw_write_example(device, input_shape):
    torch_input_tensor = torch.tensor([0xA,1,2,3,4,5,6,7,8,9], dtype=torch.uint8)
    torch_output_tensor = torch_input_tensor

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    print(input_tensor)
    output_tensor = ttnn.prim.noc_inline(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    print("---------- pytest result -----------")
    print(input_tensor)
    print(torch_output_tensor)
    print(output_tensor)
    print("Tensor in hex:", " ".join(hex(x) for x in output_tensor.cpu().numpy().flatten()))

    # assert torch.equal(torch_output_tensor, output_tensor)
