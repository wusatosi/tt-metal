import ttnn
import torch
import pytest

import math


def create_ws_memory_config(output_core_grid, input_shape):
    if isinstance(output_core_grid, tuple):
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(output_core_grid[0] - 1, output_core_grid[1] - 1)),
            ]
        )
    else:
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in output_core_grid
            ]
        )
    padded_out_w = math.ceil(input_shape[3] / output_core_range_set.num_cores() / 32) * 32
    output_memory_config = ttnn.create_sharded_memory_config(
        shape=(
            input_shape[0] * input_shape[1] * input_shape[2],
            padded_out_w,
        ),
        core_grid=output_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    return output_memory_config


@pytest.mark.parametrize(
    "shape, core_grid",
    [
        ([1, 1, 32, 32 * 64], (8, 8)),
        ([1, 1, 5 * 1024 * 1024 // 32 // 64, 32 * 64], (8, 8)),
        ([1, 1, 10 * 1024 * 1024 // 32 // 64, 32 * 64], (8, 8)),
        ([1, 1, 20 * 1024 * 1024 // 32 // 64, 32 * 64], (8, 8)),
        ([1, 1, 5 * 1024 * 1024 // 32 // 32, 32 * 32], (4, 8)),
        ([1, 1, 10 * 1024 * 1024 // 32 // 32, 3 * 32], (4, 8)),
        ([1, 1, 5 * 1024 * 1024 // 32 // 32, 32 * 32], (8, 4)),
        ([1, 1, 10 * 1024 * 1024 // 32 // 32, 3 * 32], (8, 4)),
    ],
)
def test_dram_interleaved(
    device,
    shape,
    core_grid,
):
    grid = device.compute_with_storage_grid_size()
    assert grid.x * grid.y == 64, "Only valid on 64 core grid"

    torch_input_tensor = torch.randint(low=0, high=10, size=shape, dtype=torch.int32)
    ttnn_input_tensor_dram_interleaved = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_ws_memory_config = create_ws_memory_config(core_grid, shape)

    ttnn_input_tensor_ws = ttnn.to_memory_config(
        ttnn_input_tensor_dram_interleaved, memory_config=output_ws_memory_config
    )

    print(ttnn_input_tensor_dram_interleaved.memory_config())
    print(ttnn_input_tensor_ws.memory_config())


def create_dram_sharded_bhs_memory_config(device, input_shape):
    dram_grid_size = device.dram_grid_size()

    assert (
        input_shape[0] % (dram_grid_size.x * dram_grid_size.y) == 0
    ), "Input shape must be divisible by dram grid size"

    dram_shard_spec = ttnn.ShardSpec(
        grid=ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        shard_shape=[
            input_shape[1] * input_shape[2],
            input_shape[3],
        ],
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_dram_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    return sharded_dram_mem_config


def create_bhs_memory_config(output_core_grid, input_shape):
    if isinstance(output_core_grid, tuple):
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(output_core_grid[0] - 1, output_core_grid[1] - 1)),
            ]
        )
    elif isinstance(output_core_grid, list) and isinstance(output_core_grid[0], ttnn.CoreCoord):
        output_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(core_coord, core_coord) for core_coord in output_core_grid]
        )
    else:
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in output_core_grid
            ]
        )

    assert input_shape[0] % 12 == 0, "Input shape must be divisible by dram grid size"

    output_memory_config = ttnn.create_sharded_memory_config(
        shape=(
            input_shape[1] * input_shape[2],
            input_shape[3],
        ),
        core_grid=output_core_range_set,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    return output_memory_config


@pytest.mark.parametrize(
    "shape",
    [
        [12, 1, 64, 1024],
        [12, 1, 128, 1024],
        [12, 1, 256, 1024],
    ],
)
def test_dram_sharded(
    device,
    shape,
):
    grid = device.compute_with_storage_grid_size()
    assert grid.x * grid.y == 64, "Only valid on 64 core grid"

    torch_input_tensor = torch.randint(low=0, high=10, size=shape, dtype=torch.int32)

    dram_sharded_bhs_memory_config = create_dram_sharded_bhs_memory_config(device, shape)

    ttnn_input_tensor_dram_interleaved = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_sharded_bhs_memory_config,
    )

    core_grid = device.get_optimal_dram_bank_to_logical_worker_assignment()
    output_bhs_memory_config = create_bhs_memory_config(core_grid, shape)

    ttnn_input_tensor_bhs = ttnn.to_memory_config(
        ttnn_input_tensor_dram_interleaved, memory_config=output_bhs_memory_config
    )

    print(ttnn_input_tensor_dram_interleaved.memory_config())
    print(ttnn_input_tensor_bhs.memory_config())
