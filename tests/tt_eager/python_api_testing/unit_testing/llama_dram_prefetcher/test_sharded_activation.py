# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
import ttnn
from models.utility_functions import skip_for_grayskull, skip_for_blackhole
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc


def check_valid_subcoregrids(num_cores, subcoregrids=[((1, 0), (3, 9)), ((5, 0), (6, 9))]):
    core_grid_size = 0
    for subcoregrid in subcoregrids:
        subcoregrid_size = (subcoregrid[1][0] - subcoregrid[0][0] + 1) * (subcoregrid[1][1] - subcoregrid[0][1] + 1)
        core_grid_size += subcoregrid_size
    assert (
        num_cores > 0 and num_cores <= core_grid_size
    ), f"Invalid number of cores {num_cores} for core grid size {core_grid_size}"


def get_individual_core_range_set(num_cores, subcoregrids=[((1, 0), (3, 9)), ((5, 0), (6, 9))], row_wise=True):
    check_valid_subcoregrids(num_cores, subcoregrids)
    core_ranges = []
    for subcoregrid in subcoregrids:
        if row_wise:
            for y in range(subcoregrid[0][1], subcoregrid[1][1] + 1):
                for x in range(subcoregrid[0][0], subcoregrid[1][0] + 1):
                    core_range = ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
                    core_ranges.append(core_range)
                    num_cores -= 1
                    if num_cores == 0:
                        return ttnn.CoreRangeSet(core_ranges)
        else:
            for x in range(subcoregrid[0][0], subcoregrid[1][0] + 1):
                for y in range(subcoregrid[0][1], subcoregrid[1][1] + 1):
                    core_range = ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
                    core_ranges.append(core_range)
                    num_cores -= 1
                    if num_cores == 0:
                        return ttnn.CoreRangeSet(core_ranges)


def get_compact_core_range_set(num_cores, subcoregrids=[((1, 0), (3, 9)), ((5, 0), (6, 9))], row_wise=True):
    check_valid_subcoregrids(num_cores, subcoregrids)
    core_ranges = []
    for subcoregrid in subcoregrids:
        subcoregrid_size = (subcoregrid[1][0] - subcoregrid[0][0] + 1) * (subcoregrid[1][1] - subcoregrid[0][1] + 1)
        if num_cores >= subcoregrid_size:
            core_range = ttnn.CoreRange(
                ttnn.CoreCoord(subcoregrid[0][0], subcoregrid[0][1]),
                ttnn.CoreCoord(subcoregrid[1][0], subcoregrid[1][1]),
            )
            core_ranges.append(core_range)
            num_cores -= subcoregrid_size

        else:
            subcoregrid_width = subcoregrid[1][0] - subcoregrid[0][0] + 1
            subcoregrid_height = subcoregrid[1][1] - subcoregrid[0][1] + 1
            start_x = subcoregrid[0][0]
            start_y = subcoregrid[0][1]
            if row_wise:
                if num_cores // subcoregrid_width > 0:
                    end_x = subcoregrid[1][0]
                    end_y = start_y + (num_cores // subcoregrid_width) - 1
                    core_range = ttnn.CoreRange(ttnn.CoreCoord(start_x, start_y), ttnn.CoreCoord(end_x, end_y))
                    core_ranges.append(core_range)
                    num_cores -= (end_y - start_y + 1) * subcoregrid_width
                    start_y = end_y + 1
                if num_cores % subcoregrid_width > 0:
                    end_x = start_x + num_cores - 1
                    end_y = start_y
                    core_range = ttnn.CoreRange(ttnn.CoreCoord(start_x, start_y), ttnn.CoreCoord(end_x, end_y))
                    core_ranges.append(core_range)
                    num_cores = 0
            else:
                if num_cores // subcoregrid_height > 0:
                    end_y = subcoregrid[1][1]
                    end_x = start_x + (num_cores // subcoregrid_height) - 1
                    core_range = ttnn.CoreRange(ttnn.CoreCoord(start_x, start_y), ttnn.CoreCoord(end_x, end_y))
                    core_ranges.append(core_range)
                    num_cores -= (end_x - start_x + 1) * subcoregrid_height
                    start_x = end_x + 1
                if num_cores % subcoregrid_height > 0:
                    end_y = start_y + num_cores - 1
                    end_x = start_x
                    core_range = ttnn.CoreRange(ttnn.CoreCoord(start_x, start_y), ttnn.CoreCoord(end_x, end_y))
                    core_ranges.append(core_range)
                    num_cores = 0
        if num_cores == 0:
            return ttnn.CoreRangeSet(core_ranges)


@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
@pytest.mark.parametrize("row_wise", [True, False])
@pytest.mark.parametrize("pcc", [0.9999])
@pytest.mark.parametrize("activation_shape, shard_cores", [[(1, 1, 32, 2048), 32], [(1, 1, 8, 1280), 40]])
def test_sharded_activation(device, row_wise, pcc, activation_shape, shard_cores):
    compute_grid_size = device.compute_with_storage_grid_size()
    print(f"Device Core Grid {compute_grid_size}")

    torch.manual_seed(10)
    activation = torch.randn(activation_shape, dtype=torch.bfloat16)

    core_range_set = get_compact_core_range_set(shard_cores, row_wise=row_wise)
    print(f"Core Set {core_range_set}")

    act_mem_config = ttnn.create_sharded_memory_config(
        shape=(32, activation_shape[-1] // shard_cores),
        core_grid=core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_activation_sharded = ttnn.as_tensor(
        activation, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=act_mem_config
    )
    tt_activation_torch = ttnn.to_torch(tt_activation_sharded)
    ttnn.deallocate(tt_activation_sharded)
    passing, pcc_mesg = comp_pcc(activation, tt_activation_torch, pcc)
    assert passing, pcc_mesg
