# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
import ttnn
from models.utility_functions import skip_for_grayskull, skip_for_blackhole
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from tests.tt_eager.python_api_testing.unit_testing.llama_dram_prefetcher.test_sharded_activation import (
    check_valid_subcoregrids,
    get_individual_core_range_set,
    get_compact_core_range_set,
)


@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
@pytest.mark.parametrize("pcc", [0.9999])
def test_sharded_slice_matmul(device, pcc):
    compute_grid_size = device.compute_with_storage_grid_size()
    print(f"Device Core Grid {compute_grid_size}")

    activation_shape = [1, 1, 32, 1280]
    slice_matrix_shape = [1, 1, 8, 32]
    torch.manual_seed(10)
    activation = torch.randn(activation_shape, dtype=torch.bfloat16)
    slice_matrix = torch.randn(slice_matrix_shape, dtype=torch.bfloat16)

    activation_core_range_set = get_compact_core_range_set(20, row_wise=True)
    print(f"Core Set {activation_core_range_set}")

    act_mem_config = ttnn.create_sharded_memory_config(
        shape=(32, 64),
        core_grid=activation_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_activation_sharded = ttnn.as_tensor(
        activation, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=act_mem_config
    )

    slicemat_core_range_set = get_compact_core_range_set(1, row_wise=True)
    print(f"Core Set {slicemat_core_range_set}")
    slicemat_mem_config = ttnn.create_sharded_memory_config(
        shape=(32, 32),
        core_grid=slicemat_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_slicemat_sharded = ttnn.as_tensor(
        slice_matrix, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=slicemat_mem_config
    )
    tt_output_sharded = ttnn.matmul(
        tt_slicemat_sharded, tt_activation_sharded, memory_config=act_mem_config, dtype=ttnn.bfloat16
    )

    # passing, pcc_mesg = comp_pcc(activation, tt_activation_torch, pcc)
    # assert passing, pcc_mesg
