# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttnn
import pytest
import torch
import math
import os


from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)


def rms_norm(x, dim, gamma, beta, eps):
    return x * torch.rsqrt(x.pow(2).mean([-i for i in range(1, len(dim) + 1)], keepdim=True) + eps) * gamma + beta


@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_layernorm_perf(mesh_device):
    torch.manual_seed(1234)

    num_devices_fractured = 4
    core_grid = ttnn.CoreGrid(x=2, y=8)

    num_cores = core_grid.num_cores
    dim = int(
        math.ceil(8192 / num_devices_fractured / num_cores / 32) * num_devices_fractured * num_cores * 32
    )  # padded
    print(f"dim: {dim}")

    input_shape = (1, 1, 32, dim)

    input_core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1)),
            # ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 7)),
        ]
    )

    size_per_device = dim // num_devices_fractured

    # Input memory config
    input_memory_config = ttnn.create_sharded_memory_config(
        shape=(
            input_shape[0] * input_shape[1] * input_shape[2],
            input_shape[3] // num_devices_fractured // input_core_range_set.num_cores(),
        ),
        core_grid=input_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    print(input_memory_config)

    # Create input tensor with input memory config
    input_tensor_torch = torch.randn(input_shape)
    gamma_torch = torch.randn((1, 1, 1, input_shape[3]))

    input_tensor = ttnn.as_tensor(
        input_tensor_torch,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device=mesh_device, dims=(None, 3), mesh_shape=list(mesh_device.shape)),
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_memory_config,
    )

    gamma_tensor = ttnn.as_tensor(
        gamma_torch.reshape([1, 1, dim // 32, 32]),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=list(mesh_device.shape)),
    )

    ln_prg_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(core_grid.x, core_grid.y),
        subblock_w=(size_per_device // num_cores) // 32,
        block_h=1,
        block_w=(size_per_device // num_cores) // 32,
        inplace=False,
    )

    ln_sharded_stats_memcfg = ttnn.create_sharded_memory_config(
        shape=[1, 1, 32, 32 * num_devices_fractured],
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.WIDTH,
    )

    # Run distributed rmsnorm part 1
    tt_stats = ttnn.rms_norm_pre_all_gather(input_tensor, program_config=ln_prg_cfg)

    # All gather stats
    tt_stats = ttnn.all_gather(
        tt_stats,
        3,
        num_links=1,
        cluster_axis=1,
        mesh_device=mesh_device,
        memory_config=ln_sharded_stats_memcfg,
        topology=ttnn.Topology.Linear,
    )

    # Run distributed rmsnorm part 2
    tt_out = ttnn.rms_norm_post_all_gather(
        input_tensor,
        epsilon=1e-05,
        weight=gamma_tensor,
        program_config=ln_prg_cfg,
        stats=tt_stats,
    )
    tt_stats.deallocate(True)

    tt_out_torch = ttnn.to_torch(
        tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=(8, 4))
    )[0].unsqueeze(0)

    ref_lnorm = rms_norm(input_tensor_torch, [3], gamma_torch, torch.zeros_like(gamma_torch), 1e-5)

    passing, output = comp_pcc(tt_out_torch, ref_lnorm, 0.999)
    logger.info(output)
    assert passing
