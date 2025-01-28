# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest


import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

# @pytest.mark.parametrize("shape", [[32, 288]])
# def test_untilize_with_unpadding_uint32(shape, device):
#     torch.manual_seed(2005)
#     input_a = torch.randint(1, 64, shape, dtype=torch.int32)
#     input_tensor = ttnn.from_torch(input_a, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32)
#     output_tensor = ttnn.untilize_with_unpadding(input_tensor, [3,279])
#     output_tensor = ttnn.to_torch(output_tensor)
#     assert_with_pcc(input_a[:4, :280], output_tensor)

# @pytest.mark.parametrize("shape", [[32, 512]])
# def test_untilize_uint32(shape, device):
#     torch.manual_seed(2005)
#     input_a = torch.randint(1, 64, shape, dtype=torch.int32)
#     input_tensor = ttnn.from_torch(input_a, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32)
#     output_tensor = ttnn.untilize(input_tensor)
#     output_tensor = ttnn.to_torch(output_tensor)
#     assert_with_pcc(input_a, output_tensor)

# @pytest.mark.parametrize("shape", [[15, 15]])
# def test_tilize_with_val_padding_uint32(shape, device):
#     torch.manual_seed(2005)
#     input_a = torch.randint(0, 64, shape, dtype=torch.int32)
#     input_tensor = ttnn.from_torch(input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)
#     output_tensor = ttnn.tilize_with_val_padding(input_tensor, [32,32], 70)
#     output_tensor = ttnn.to_torch(output_tensor)
#     assert_with_pcc(input_a, output_tensor)


@pytest.mark.parametrize("shape", [[32, 32]])
def test_tilize_uint32(shape, device):
    torch.manual_seed(2005)
    input_a = torch.randint(0, 64, shape, dtype=torch.int32)
    # input_a = torch.arange(0, 64, dtype=torch.int32).reshape(8, 8)

    input_tensor = ttnn.from_torch(input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)
    output_tensor = ttnn.tilize(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(input_a, output_tensor)
