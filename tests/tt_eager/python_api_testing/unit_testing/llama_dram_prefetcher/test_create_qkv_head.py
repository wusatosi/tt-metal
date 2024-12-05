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
@pytest.mark.parametrize("activation_shape, shard_cores", [[(1, 1, 8, 1280), 40]])
def test_create_qkv_head(device, pcc, activation_shape, shard_cores):
    compute_grid_size = device.compute_with_storage_grid_size()
    print(f"Device Core Grid {compute_grid_size}")

    torch.manual_seed(10)
    activation = torch.randn(activation_shape, dtype=torch.bfloat16)

    core_range_set = get_compact_core_range_set(shard_cores, row_wise=True)
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
    # tt operation
    (
        q_heads_tt,  # [seqlen, n_local_heads, bsz, head_dim]
        k_heads_tt,  # [seqlen, n_local_kv_heads, bsz, head_dim]
        v_heads_tt,  # [seqlen, n_local_kv_heads, bsz, head_dim]
    ) = ttnn.experimental.nlp_create_qkv_heads_decode(
        tt_activation_sharded,
        num_heads=8,
        num_kv_heads=1,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        overlap_qk_coregrid=False,
    )
    breakpoint()
    assert q_heads_tt.shape == ttnn.Shape((1, 8, 8, 128), (1, 8, 32, 128))
    assert k_heads_tt.shape == ttnn.Shape((1, 8, 1, 128), (1, 8, 32, 128))
    assert v_heads_tt.shape == ttnn.Shape((1, 8, 1, 128), (1, 8, 32, 128))
