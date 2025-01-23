# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import pytest
from models.utility_functions import comp_allclose_and_pcc
from loguru import logger
from models.utility_functions import is_wormhole_b0

from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
)


def get_torch_dtype(dtype):
    if dtype == ttnn.int32:
        return torch.int32
    elif dtype == ttnn.float32:
        return torch.float32
    else:
        return torch.bfloat16


def run_moreh_softmax_test(
    shape,
    dim,
    ttnn_dtype,
    layout,
    device,
    rtol,
    atol,
    use_randint,
    use_optional_output_tensor=False,
    strategy=ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.SMALL_W,
    compute_kernel_options=None,
):
    if ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip(f"bfloat8_b is not supported")
    torch_dtype = get_torch_dtype(ttnn_dtype)
    if use_randint == True:
        torch_input = torch.randint(low=0, high=4, size=shape).to(torch_dtype) + 100
    else:
        torch_input = torch.rand(size=shape, dtype=torch_dtype) + 100
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=layout, device=device)

    torch_output = torch.softmax(torch_input, dim)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    if strategy is None:
        ttnn_output = ttnn.operations.moreh.softmax(ttnn_input, dim, compute_kernel_config=compute_kernel_config)
    else:
        ttnn_output = ttnn.operations.moreh.softmax(
            ttnn_input, dim, compute_kernel_config=compute_kernel_config, strategy=strategy
        )


@pytest.mark.parametrize(
    "shape_dim",
    [
        [[32, 32], 1],  # single tile
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
def test_softmax_for_dim_hw(shape_dim, dtype, device):
    compute_kernel_options = True
    shape, dim = shape_dim
    torch.manual_seed(0)
    rtol = atol = 0.05
    run_moreh_softmax_test(
        shape,
        dim,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        rtol,
        atol,
        True,
        compute_kernel_options=compute_kernel_options,
    )
