# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def torch_make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def ttnn_make_grid(nx=20, ny=20, device=None):
    x_coords = ttnn.arange(0, nx, device=device)
    y_coords = ttnn.arange(0, ny, device=device)

    x_coords = ttnn.repeat_interleave(x_coords, ny, 2)
    y_coords = ttnn.repeat_interleave(y_coords, nx, 2)

    y_coords = ttnn.permute(y_coords, (0, 1, 3, 2))

    x_coords = ttnn.unsqueeze(x_coords, -1)
    y_coords = ttnn.unsqueeze(y_coords, -1)

    grid = ttnn.concat((x_coords, y_coords), dim=-1)
    grid = ttnn.reshape(grid, (1, 1, ny, nx, 2))

    return grid


@pytest.mark.parametrize("repeats", [20, 40, 80])
def test_pcc_make_grid(device, reset_seeds, repeats):
    tensor_a = torch_make_grid(repeats, repeats)
    tensor_b = ttnn_make_grid(repeats, repeats, device)
    tensor_b = ttnn.to_torch(tensor_b)
    assert_with_pcc(tensor_a, tensor_b, 1.0)


"""
TESTCASE 1: repeats = 20 -----> PASSED
TESTCASE 2: repeats = 40 -----> PASSED
TESTCASE 3: repeats = 80 -----> FAILED

E       RuntimeError: TT_THROW @ ../tt_metal/impl/kernels/kernel.cpp:241: tt::exception
E       info:
E       403 unique+common runtime args targeting kernel reader_concat_stick_layout_interleaved_start_id on (x=0,y=0) are too large. Max allowable is 256
"""
