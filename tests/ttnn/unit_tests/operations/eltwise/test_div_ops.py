# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import pytest
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.remainder,
    ],
)
def test_remainder_fp32(device, ttnn_function):
    x_torch = torch.tensor([[15]], dtype=torch.float32)
    y_torch = torch.tensor([[10]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_div = ttnn.remainder(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_div)

    # print("torch out in ttnn", ttnn.to_torch(z_tt))
    # print("tt out in torch", tt_out)
    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.abs,
    ],
)
def test_abs_fp32(device, ttnn_function):
    x_torch = torch.tensor([[0, -1, 1, 1.99]], dtype=torch.float32)
    y_torch = torch.tensor([[10]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_div = ttnn.abs(x_tt)
    tt_out = ttnn.to_torch(z_tt_div)

    print("torch out in ttnn", ttnn.to_torch(z_tt))
    print("tt out in torch", tt_out)
    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status
