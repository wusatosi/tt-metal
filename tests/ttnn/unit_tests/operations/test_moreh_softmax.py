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


def run_moreh_softmax(
    shape,
    dim,
    ttnn_dtype,
    layout,
    device,
    rtol,
    atol,
    use_randint,
    use_optional_output_tensor=False,
    strategy=None,
    compute_kernel_options=None,
):
    torch_dtype = get_torch_dtype(ttnn_dtype)
    if use_randint == True:
        torch_input = torch.randint(low=0, high=4, size=shape).to(torch_dtype) * 20 + 10
    else:
        torch_input = torch.rand(size=shape, dtype=torch_dtype) * 20 + 10
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=layout, device=device)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    ttnn_output = ttnn.operations.moreh.softmax(
        ttnn_input, dim, strategy=strategy, compute_kernel_config=compute_kernel_config
    )

    ttnn_output = ttnn.to_torch(ttnn_output).to(torch_dtype)

    print(ttnn_output)


@pytest.mark.parametrize(
    "shape_dim_strategy",
    [
        [[4, 4], 1, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.SMALL_W],
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_softmax_bfp8_with_compute_kernel_options(shape_dim_strategy, dtype, compute_kernel_options, device):
    shape, dim, strategy = shape_dim_strategy
    torch.manual_seed(0)
    rtol = atol = 0.05
    if compute_kernel_options:
        print("FP32 mode turned on")
    else:
        print("FP32 mode turned off")
    run_moreh_softmax(
        shape,
        dim,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        rtol,
        atol,
        True,
        strategy=strategy,
        compute_kernel_options=compute_kernel_options,
    )
