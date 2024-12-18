# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
from functools import partial

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random

from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
    ],
)
@pytest.mark.parametrize("h", [64 * 32, 3 * 32 * 32, 2 * 64 * 32, 5 * 32 * 32, 3 * 64 * 32])  # , 7*32*32, 4*64*32])
@pytest.mark.parametrize("w", [64 * 32, 3 * 32 * 32, 2 * 64 * 32, 5 * 32 * 32, 3 * 64 * 32])  # , 7*32*32, 4*64*32])
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_sum(device, batch_size, h, w, c, n, dim, input_dtype, input_memory_config, output_memory_config):
    torch.manual_seed(0)

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

    output_tensor = ttnn.to_torch(output_tensor).squeeze()
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


def gen_hwd():
    hw_list = [64, 128]
    num = [3 * 32, 16 * 32, 47 * 32]
    ret = []
    for n in num:
        for hw in hw_list:
            # ret.extend([((hw + h) * 32, n, -1) for h in range(1, 64, 3)])
            ret.extend([(n, (hw + w) * 32, -2) for w in range(1, 64, 3)])
    return ret


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
    ],
)
@pytest.mark.parametrize("hwd", gen_hwd())
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_sum_hwd(device, batch_size, hwd, c, n, input_dtype, input_memory_config, output_memory_config):
    torch.manual_seed(0)
    h, w, dim = hwd
    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )((n, c, h, w))
    # golden_function = ttnn.get_golden_function(ttnn.sum)
    # torch_output_tensor = golden_function(torch_input_tensor, dim=dim, memory_config=output_memory_config)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
    )

    output_tensor = ttnn.sum(input_tensor, dim=dim, memory_config=output_memory_config)
    # output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    # output_tensor = ttnn.from_device(output_tensor)

    # if dim:
    #     rank = 4
    #     if isinstance(dim, tuple):
    #         for d in dim:
    #             if d < 0:
    #                 d += rank
    #     else:
    #         if dim < 0:
    #             dim += rank
    #     output_tensor = ttnn.to_torch(output_tensor).squeeze(dim=dim)
    # else:
    #     output_tensor = ttnn.to_torch(output_tensor).squeeze()
    # assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


def fixed_and_range_generator():
    h = 32 * 32
    return [(h, w * 32) for w in range(64 + 1, 256)]


@pytest.mark.parametrize("hw", fixed_and_range_generator())
@pytest.mark.parametrize("dim", [-2])
@pytest.mark.parametrize("input_dtype", [ttnn.float32])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_same_tpc(device, hw, dim, input_dtype, input_memory_config, output_memory_config):
    torch.manual_seed(0)
    # input_memory_config = output_memory_config = ttnn.DRAM_MEMORY_CONFIG
    # dim = -2
    batch_size = 1

    h, w = hw
    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype  # torch.int for tnn.uint?
    )((1, 1, h, w))
    # torch_input_tensor = torch.randn((batch_size, h, w), dtype=torch.bfloat16)
    # golden_function = ttnn.get_golden_function(ttnn.sum)
    # torch_output_tensor = golden_function(torch_input_tensor, dim=dim,  memory_config=output_memory_config)
    # torch_output_tensor = torch.max(torch_input_tensor, dim=dim, keepdim=True).values

    # shard_config = ttnn.create_sharded_memory_config(
    #         (1,1,h,w), core_grid=ttnn.CoreGrid(y=core_grid_y, x=core_grid_x), strategy=ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=False
    #     )

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
    )
    # input_tensor = ttnn.transpose(input_tensor, 0, 1)
    output_tensor = ttnn.sum(input_tensor, dim=dim, memory_config=output_memory_config)
    # output_tensor = ttnn.max(input_tensor, dim=dim)
    # output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    # output_tensor = ttnn.from_device(output_tensor)

    # output_tensor = ttnn.to_torch(output_tensor)
    # if dim:
    #     rank = 4
    #     if isinstance(dim, tuple):
    #         for d in dim:
    #             if d < 0:
    #                 d += rank
    #     else:
    #         if dim < 0:
    #             dim += rank
    #     output_tensor = ttnn.to_torch(output_tensor).squeeze(dim=dim)
    # else:
    #     output_tensor = ttnn.to_torch(output_tensor).squeeze()
    # assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


def hw_gen():
    tpc = 63 * 32
    h_list = [8 * 32, 16 * 32, 32 * 32]
    ret = []
    for h in h_list:
        ret.extend([(h, ((tpc // h) * 64 + c) * 32) for c in range(1, 64)])
    return ret


@pytest.mark.parametrize("hw", hw_gen())
@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize("input_dtype", [ttnn.float32])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_same_tpc_cols(device, hw, dim, input_dtype, input_memory_config, output_memory_config):
    torch.manual_seed(0)
    # input_memory_config = output_memory_config = ttnn.DRAM_MEMORY_CONFIG
    # dim = -2
    batch_size = 1

    w, h = hw
    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype  # torch.int for tnn.uint?
    )((1, 1, h, w))
    # torch_input_tensor = torch.randn((batch_size, h, w), dtype=torch.bfloat16)
    # golden_function = ttnn.get_golden_function(ttnn.sum)
    # torch_output_tensor = golden_function(torch_input_tensor, dim=dim,  memory_config=output_memory_config)
    # torch_output_tensor = torch.max(torch_input_tensor, dim=dim, keepdim=True).values

    # shard_config = ttnn.create_sharded_memory_config(
    #         (1,1,h,w), core_grid=ttnn.CoreGrid(y=core_grid_y, x=core_grid_x), strategy=ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=False
    #     )

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
    )
    # input_tensor = ttnn.transpose(input_tensor, 0, 1)
    output_tensor = ttnn.sum(input_tensor, dim=dim, memory_config=output_memory_config)
    # output_tensor = ttnn.max(input_tensor, dim=dim)
    # output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    # output_tensor = ttnn.from_device(output_tensor)

    # output_tensor = ttnn.to_torch(output_tensor)
    # if dim:
    #     rank = 4
    #     if isinstance(dim, tuple):
    #         for d in dim:
    #             if d < 0:
    #                 d += rank
    #     else:
    #         if dim < 0:
    #             dim += rank
    #     output_tensor = ttnn.to_torch(output_tensor).squeeze(dim=dim)
    # else:
    #     output_tensor = ttnn.to_torch(output_tensor).squeeze()
    # assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


def w_generator():
    return [w * 32 for w in range(8, 64, 8)]


def h_generator():
    return [h * 32 for h in range(1, 9)]


@pytest.mark.parametrize("h", h_generator())
@pytest.mark.parametrize("w", w_generator())
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_until_full(device, h, w, dim, input_dtype, input_memory_config, output_memory_config):
    torch.manual_seed(0)
    # input_memory_config = output_memory_config = ttnn.DRAM_MEMORY_CONFIG
    # dim = -2
    batch_size = 1

    # h, w = hw
    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype  # torch.int for tnn.uint?
    )((1, 1, h, w))
    # torch_input_tensor = torch.randn((batch_size, h, w), dtype=torch.bfloat16)
    # golden_function = ttnn.get_golden_function(ttnn.sum)
    # torch_output_tensor = golden_function(torch_input_tensor, dim=dim,  memory_config=output_memory_config)
    # torch_output_tensor = torch.max(torch_input_tensor, dim=dim, keepdim=True).values

    # shard_config = ttnn.create_sharded_memory_config(
    #         (1,1,h,w), core_grid=ttnn.CoreGrid(y=core_grid_y, x=core_grid_x), strategy=ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=False
    #     )

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
    )
    # input_tensor = ttnn.transpose(input_tensor, 0, 1)
    output_tensor = ttnn.sum(input_tensor, dim=dim, memory_config=output_memory_config)
    # output_tensor = ttnn.max(input_tensor, dim=dim)
    # output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    # output_tensor = ttnn.from_device(output_tensor)

    # output_tensor = ttnn.to_torch(output_tensor)
    # if dim:
    #     rank = 4
    #     if isinstance(dim, tuple):
    #         for d in dim:
    #             if d < 0:
    #                 d += rank
    #     else:
    #         if dim < 0:
    #             dim += rank
    #     output_tensor = ttnn.to_torch(output_tensor).squeeze(dim=dim)
    # else:
    #     output_tensor = ttnn.to_torch(output_tensor).squeeze()
    # assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
    ],
)
@pytest.mark.parametrize("h", [2 * 32])  # , 7*32*32, 4*64*32])
@pytest.mark.parametrize("w", [17 * 64 * 32])  # , 7*32*32, 4*64*32])
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("dim", [-2])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_optimal(device, batch_size, h, w, c, n, dim, input_dtype, input_memory_config, output_memory_config):
    torch.manual_seed(0)

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

    output_tensor = ttnn.to_torch(output_tensor).squeeze()
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


# UNTIL FULL GRID TESTS


def gen_hwd_u():
    hw_list = [3 * 32, 16 * 32, 31 * 32, 64 * 32, 3 * 32 * 32, 83 * 32, 120 * 32, 2 * 64 * 32, 5 * 32 * 32, 167 * 32]
    # num = 32*32
    ret = []
    for num in hw_list:
        ret.extend([(h * 32, num, -1) for h in range(1, 64, 3)])
        ret.extend([(num, w * 32, -2) for w in range(1, 64, 3)])
    return ret


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
    ],
)
@pytest.mark.parametrize("hwd", gen_hwd_u())
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("input_memory_config", [ttnn.L1_MEMORY_CONFIG])  # , ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_sum_hwd_u(device, batch_size, hwd, c, n, input_dtype, input_memory_config, output_memory_config):
    torch.manual_seed(0)
    h, w, dim = hwd
    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )((n, c, h, w))
    # golden_function = ttnn.get_golden_function(ttnn.sum)
    # torch_output_tensor = golden_function(torch_input_tensor, dim=dim, memory_config=output_memory_config)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
    )

    output_tensor = ttnn.sum(input_tensor, dim=dim, memory_config=output_memory_config)
    # output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    # output_tensor = ttnn.from_device(output_tensor)

    # if dim:
    #     rank = 4
    #     if isinstance(dim, tuple):
    #         for d in dim:
    #             if d < 0:
    #                 d += rank
    #     else:
    #         if dim < 0:
    #             dim += rank
    #     output_tensor = ttnn.to_torch(output_tensor).squeeze(dim=dim)
    # else:
    #     output_tensor = ttnn.to_torch(output_tensor).squeeze()
    # assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
