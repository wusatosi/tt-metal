# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.sweep_framework.sweep_utils.utils import tensor_to_dtype


activations_dict = {"relu": [torch.relu, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]}


def run_eltwise_multiply_apply_activation_tests(
    input_shape,
    activations,
    dtype,
    dlayout,
    in_mem_cfg,
    out_mem_cfg,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = tensor_to_dtype(torch.Tensor(size=input_shape).uniform_(-100, 100), dtype[0])
    y = tensor_to_dtype(torch.Tensor(size=input_shape).uniform_(-100, 100), dtype[1])

    try:
        # get ref result
        # ref_value = x * y
        # if activations is not None:
        #    for activation in activations:
        #        golden_function = activations_dict[activation][0]
        #        ref_value = golden_function(ref_value)
        golden_function = ttnn.get_golden_function(ttnn.multiply)
        ref_value = golden_function(x, y, activations=activations)

        # import inspect as i
        # import sys
        # sys.stdout.write(i.getsource(golden_function))

        tt_x = ttnn.from_torch(x, dtype=dtype[0], layout=dlayout, memory_config=in_mem_cfg[0], device=device)
        tt_y = ttnn.from_torch(y, dtype=dtype[1], layout=dlayout, memory_config=in_mem_cfg[1], device=device)

        activation_list = []
        if activations is not None:
            for activation in activations:
                activation_list.append(activations_dict[activation][1])
        else:
            activation_list = None

        tt_result = ttnn.multiply(tt_x, tt_y, activations=activation_list, memory_config=out_mem_cfg)
        tt_result = ttnn.to_torch(tt_result)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.999)


test_sweep_args = [
    (
        [1, 8, 99, 180],
        ["relu"],
        [ttnn.bfloat16, ttnn.bfloat16],
        ttnn.TILE_LAYOUT,
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        11079580,
    ),
    (
        [3, 2, 192, 32],
        ["relu"],
        [ttnn.bfloat16, ttnn.bfloat16],
        ttnn.TILE_LAYOUT,
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        11079580,
    ),
    (
        [3, 2, 192, 32],
        None,
        [ttnn.bfloat16, ttnn.bfloat16],
        ttnn.TILE_LAYOUT,
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        11079580,
    ),
]


@pytest.mark.parametrize(
    "input_shape, activation, dtype, dlayout, in_mem_cfg, out_mem_cfg, data_seed",
    (test_sweep_args),
)
def test_multiply_apply_activation(input_shape, activation, dtype, dlayout, in_mem_cfg, out_mem_cfg, data_seed, device):
    run_eltwise_multiply_apply_activation_tests(
        input_shape, activation, dtype, dlayout, in_mem_cfg, out_mem_cfg, data_seed, device
    )
