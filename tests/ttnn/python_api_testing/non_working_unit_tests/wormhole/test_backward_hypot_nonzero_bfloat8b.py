# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback
from itertools import product
from functools import partial

from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.sweep_framework.sweep_utils.utils import gen_rand_exclude_range

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_backward_hypot_tests(
    input_shape,
    exclude_range,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)

    x = gen_func_with_cast_tt(
        partial(gen_rand_exclude_range, excluderange=exclude_range[0], low=-100, high=100), dtype[0]
    )(input_shape[0])
    y = gen_func_with_cast_tt(
        partial(gen_rand_exclude_range, excluderange=exclude_range[0], low=-100, high=100), dtype[1]
    )(input_shape[0])
    z = gen_func_with_cast_tt(
        partial(gen_rand_exclude_range, excluderange=exclude_range[0], low=-100, high=100), dtype[2]
    )(input_shape[0])

    y.requires_grad = True
    z.requires_grad = True

    try:
        # get ref result
        golden_function = ttnn.get_golden_function(ttnn.hypot_bw)
        ref_value = golden_function(x, y, z)

        tt_x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])
        tt_y = ttnn_ops.setup_ttnn_tensor(y, device, dlayout[0], in_mem_config[1], dtype[1])
        tt_z = ttnn_ops.setup_ttnn_tensor(z, device, dlayout[0], in_mem_config[2], dtype[2])

        tt_result = ttnn.hypot_bw(tt_x, tt_y, tt_z, memory_config=output_mem_config)
        tt_result = [
            ttnn_ops.ttnn_tensor_to_torch(tt_result[0]),
            ttnn_ops.ttnn_tensor_to_torch(tt_result[1]),
        ]

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    for i in range(2):
        assert len(tt_result[i].shape) == len(ref_value[i].shape)
        assert tt_result[i].shape == ref_value[i].shape
        assert_with_pcc(ref_value[i], tt_result[i], 0.99)


test_sweep_args = [
    (
        [[4, 226, 2]],
        [[-1, 1]],
        [ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        0,
    ),
    (
        [[2, 3, 243, 9]],
        [[-1, 1]],
        [ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        0,
    ),
    (
        [[8, 141, 15]],
        [[-1, 1]],
        [ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        0,
    ),
]


@pytest.mark.parametrize(
    "input_shape, exclude_range, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_backward_hypot(input_shape, exclude_range, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_backward_hypot_tests(
        input_shape, exclude_range, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device
    )
