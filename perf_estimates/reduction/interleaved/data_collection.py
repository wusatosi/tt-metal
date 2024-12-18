# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
from functools import partial

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random

from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt


# FULL GRID TESTS
def gen_hwd():
    hw_list = [64, 128]
    num = [3 * 32, 16 * 32, 47 * 32]
    ret = []
    for n in num:
        for hw in hw_list:
            ret.extend([((hw + h) * 32, n, -1) for h in range(1, 64, 3)])
            ret.extend([(n, (hw + w) * 32, -2) for w in range(1, 64, 3)])
    return ret


@pytest.mark.parametrize("hwd", gen_hwd())
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_sum(device, hwd, c, n, input_dtype, input_memory_config, output_memory_config):
    torch.manual_seed(0)
    h, w, dim = hwd
    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )((n, c, h, w))
    golden_function = ttnn.get_golden_function(ttnn.sum)
    torch_output_tensor = golden_function(torch_input_tensor, dim=dim, memory_config=output_memory_config, keepdim=True)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
    )

    output_tensor = ttnn.sum(input_tensor, dim=dim, memory_config=output_memory_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


def fixed_and_range_generator():
    h = 32 * 32
    return [(h, w * 32) for w in range(64 + 1, 256)]


@pytest.mark.parametrize("hw", fixed_and_range_generator())
@pytest.mark.parametrize("dim", [-2])
@pytest.mark.parametrize("input_dtype", [ttnn.float32])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_overworked_impact(device, hw, dim, input_dtype, input_memory_config, output_memory_config):
    torch.manual_seed(0)
    batch_size = 1

    h, w = hw
    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype  # torch.int for tnn.uint?
    )((1, 1, h, w))

    golden_function = ttnn.get_golden_function(ttnn.sum)
    torch_output_tensor = golden_function(torch_input_tensor, dim=dim, memory_config=output_memory_config, keepdim=True)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
    )
    output_tensor = ttnn.sum(input_tensor, dim=dim, memory_config=output_memory_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


# NON FULL GRID TESTS
def gen_hwd_nf():
    hw_list = [3 * 32, 16 * 32, 31 * 32, 64 * 32, 3 * 32 * 32, 83 * 32, 120 * 32, 2 * 64 * 32, 5 * 32 * 32, 167 * 32]
    # num = 32*32
    ret = []
    for num in hw_list:
        ret.extend([(h * 32, num, -1) for h in range(1, 64, 3)])
        ret.extend([(num, w * 32, -2) for w in range(1, 64, 3)])
    return ret


@pytest.mark.parametrize("hwd", gen_hwd_nf())
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("input_memory_config", [ttnn.L1_MEMORY_CONFIG])  # , ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_sum_not_full(device, hwd, c, n, input_dtype, input_memory_config, output_memory_config):
    torch.manual_seed(0)
    h, w, dim = hwd
    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )((n, c, h, w))
    golden_function = ttnn.get_golden_function(ttnn.sum)
    torch_output_tensor = golden_function(torch_input_tensor, dim=dim, memory_config=output_memory_config, keepdim=True)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
    )

    output_tensor = ttnn.sum(input_tensor, dim=dim, memory_config=output_memory_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
