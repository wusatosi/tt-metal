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
@pytest.mark.parametrize("h", [2 * 32, 8 * 32, 32 * 32])
@pytest.mark.parametrize("w", [4 * 32, 16 * 32, 128 * 32])
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("dim", [-2])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("core_num", [2, 4, 8, 16, 32, 64])
def test_sum(device, batch_size, h, w, c, n, dim, input_dtype, core_num):
    torch.manual_seed(0)

    if (w // 32) < core_num:
        pytest.skip("Too many cores for the input size")

    core_grid_x = 8 if core_num % 8 == 0 else core_num
    core_grid_y = core_num // core_grid_x

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )((1, 1, h, w))

    shard_config = ttnn.create_sharded_memory_config(
        (n, c, h, w),
        core_grid=ttnn.CoreGrid(y=core_grid_y, x=core_grid_x),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=False,
    )

    golden_function = ttnn.get_golden_function(ttnn.sum)
    torch_output_tensor = golden_function(torch_input_tensor, dim=dim, memory_config=shard_config, keepdim=True)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard_config
    )
    output_tensor = ttnn.sum(input_tensor, dim=dim, memory_config=shard_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
    ],
)
@pytest.mark.parametrize("h", [2 * 32, 8 * 32, 32 * 32])
@pytest.mark.parametrize("w", [4 * 32, 16 * 32, 128 * 32])
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("dim", [-2])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("core_num", [2, 4, 8, 16, 32, 64])
def test_max(device, batch_size, h, w, c, n, dim, input_dtype, core_num):
    torch.manual_seed(0)

    if (w // 32) < core_num:
        pytest.skip("Too many cores for the input size")

    core_grid_x = 8 if core_num % 8 == 0 else core_num
    core_grid_y = core_num // core_grid_x

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )((1, 1, h, w))

    shard_config = ttnn.create_sharded_memory_config(
        (n, c, h, w),
        core_grid=ttnn.CoreGrid(y=core_grid_y, x=core_grid_x),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=False,
    )

    golden_function = ttnn.get_golden_function(ttnn.max)
    torch_output_tensor = golden_function(torch_input_tensor, dim=dim, memory_config=shard_config, keepdim=True).values

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard_config
    )
    output_tensor = ttnn.max(input_tensor, dim=dim, memory_config=shard_config)

    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
    ],
)
@pytest.mark.parametrize("h", [2 * 32, 8 * 32, 32 * 32])
@pytest.mark.parametrize("w", [4 * 32, 16 * 32, 128 * 32])
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("dim", [-2])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("core_num", [2, 4, 8, 16, 32, 64])
def test_mean(device, batch_size, h, w, c, n, dim, input_dtype, core_num):
    torch.manual_seed(0)

    if (w // 32) < core_num:
        pytest.skip("Too many cores for the input size")

    core_grid_x = 8 if core_num % 8 == 0 else core_num
    core_grid_y = core_num // core_grid_x

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )((1, 1, h, w))

    shard_config = ttnn.create_sharded_memory_config(
        (n, c, h, w),
        core_grid=ttnn.CoreGrid(y=core_grid_y, x=core_grid_x),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=False,
    )

    golden_function = ttnn.get_golden_function(ttnn.mean)
    torch_output_tensor = golden_function(torch_input_tensor, dim=dim, memory_config=shard_config, keepdim=True)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard_config
    )
    output_tensor = ttnn.mean(input_tensor, dim=dim, memory_config=shard_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
    ],
)
@pytest.mark.parametrize("h", [2 * 32, 8 * 32, 32 * 32])
@pytest.mark.parametrize("w", [13 * 32, 19 * 32, 30 * 32])
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("dim", [-2])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b])
def test_sum_uneven(device, batch_size, h, w, c, n, dim, input_dtype):
    torch.manual_seed(0)

    core_grid_y1 = (w // 32) // 8

    core_grid_x2 = (w // 32) % 8

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )((1, 1, h, w))

    shard_config = ttnn.create_sharded_memory_config(
        (n, c, h, w // (8 * core_grid_y1 + core_grid_x2)),
        core_grid=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, core_grid_y1 - 1)),  # block of k * 8 cores
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, core_grid_y1), ttnn.CoreCoord(core_grid_x2 - 1, core_grid_y1)
                ),  # leftover block
            }
        ),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )

    golden_function = ttnn.get_golden_function(ttnn.sum)
    torch_output_tensor = golden_function(torch_input_tensor, dim=dim, memory_config=shard_config, keepdim=True)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard_config
    )
    output_tensor = ttnn.sum(input_tensor, dim=dim, memory_config=shard_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
