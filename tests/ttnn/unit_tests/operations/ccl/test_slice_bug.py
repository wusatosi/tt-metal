# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import pytest
from loguru import logger
import ttnn
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)


@pytest.mark.parametrize(
    "N, N_slice",
    [
        [3584, 2048],
    ],
)
def test_slice_tg(
    mesh_device,
    N,
    N_slice,
    function_level_defaults,
):
    num_cores = 16
    M = 32
    shape = (1, 1, M, N)

    ##### Input #####
    pt_input = torch.randn(shape, dtype=torch.float32)

    ##### TT Input #####
    storage_grid = (8, 8)
    CORE_RANGE = [(x, y) for y in range(storage_grid[1]) for x in range(storage_grid[0])][:num_cores]
    core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(x, y),
                ttnn.CoreCoord(x, y),
            )
            for x, y in CORE_RANGE
        ]
    )
    tt_input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_range_set,
            [32, shape[3] // num_cores],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    tt_input = ttnn.from_torch(
        pt_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=tt_input_mem_config,
    )
    logger.info(f"pt_input shape: {pt_input.shape}")
    logger.info(f"tt_input shape: {tt_input.shape}")

    ##### Operation ####
    pt_out = pt_input[..., :N_slice]
    tt_out = ttnn.slice(tt_input, [0, 0, 0, 0], [1, 1, M, N_slice])

    logger.info(f"tt_input shard spec: {tt_input.memory_config().shard_spec.shape}")
    logger.info(f"tt_out shard spec: {tt_out.memory_config().shard_spec.shape}")

    ##### Validation #####
    tt_out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))[..., :N_slice]
    logger.info(f"pt_out shape: {pt_out.shape}")
    logger.info(f"tt_out shape: {tt_out.shape}")
    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing
