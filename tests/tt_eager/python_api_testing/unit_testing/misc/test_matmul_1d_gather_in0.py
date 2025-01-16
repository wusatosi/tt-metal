# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn
from models.utility_functions import is_wormhole_b0, is_grayskull, skip_for_wormhole_b0
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero, roundup32
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)
import random
import math
from models.utility_functions import is_wormhole_b0, is_grayskull, is_wormhole_b0, is_blackhole


random.seed(10)


def num_cores_to_rectangle_grid(num_cores, device):
    """
    Find a rectangular core grid size, given an number of cores.

    Return None if rectangle grid is not possible.
    """
    x = device.compute_with_storage_grid_size().x
    while x > 0 and num_cores % x != 0:
        x -= 1

    if x == 0:
        return None

    y = num_cores // x
    return (x, y)


def get_physical_to_logical_core_mapping(device):
    """
    Get a mapping from physical core coords to logical core coords

    Returns a dictionary.
    """
    mapping = {}
    grid = device.compute_with_storage_grid_size()
    for x in range(grid.x):
        for y in range(grid.y):
            physical_core = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            mapping[(physical_core.x, physical_core.y)] = (x, y)
    return mapping


PREFETCHER_GRID = [
    (8, 11),
    (8, 9),
    (8, 8),
    (8, 7),
    (8, 5),
    (8, 3),
    (8, 2),
    (8, 1),
    (7, 1),
    (7, 2),
    (7, 3),
    (7, 5),
    (7, 7),
    (7, 8),
    (7, 9),
    (7, 11),
    (3, 11),
    (3, 7),
    (3, 5),
    (3, 1),
    (2, 1),
    (2, 5),
    (2, 7),
    (2, 11),
]


def run_multi_core_matmul_1d(
    device,
    in0_dtype,
    in1_dtype,
    fidelity,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    B,
    M,
    K,
    N,
    activation,
    grid,
    use_arbitrary_cores,
    num_iters,
    max_dst_tiles=8,
    pcc_threshold=0.98,
    num_reducer_partials=1,
    n_chunks=1,
):
    assert not has_bias, "Bias not supported for gather_in0 mode."
    if not isinstance(grid, tuple) and not use_arbitrary_cores:
        pytest.skip("Grid is not a tuple and not using arbitrary cores")

    N = N // n_chunks

    in0_shape = [1, B, M, K]
    in1_shape = [1, n_chunks, K, N]
    num_cores = grid[0] * grid[1] if isinstance(grid, tuple) else len(grid)
    ring_size = num_cores // num_reducer_partials

    storage_grid = num_cores_to_rectangle_grid(num_cores, device)
    if storage_grid is None:
        pytest.skip(f"Could not find a rectangle grid for num_cores: {num_cores}")

    M *= B  # Fuse batch always enabled

    in0_block_h = M // ttnn.TILE_SIZE
    in0_block_w = K // num_cores // ttnn.TILE_SIZE
    out_block_h = M // ttnn.TILE_SIZE
    out_block_w = N // ring_size // ttnn.TILE_SIZE

    num_blocks_y = (M // ttnn.TILE_SIZE - 1) // out_block_h + 1
    num_blocks_x = (N // ttnn.TILE_SIZE - 1) // out_block_w + 1
    num_blocks_total = num_blocks_y * num_blocks_x

    if num_blocks_total != ring_size:
        pytest.skip(f"num_blocks_total {num_blocks_total} != num_cores {ring_size}")

    out_subblock_h = 1
    out_subblock_w = max_dst_tiles if (out_block_h == 1 and out_block_w <= max_dst_tiles) else 8
    while out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1

    logger.debug("in0 block h w " + str(in0_block_h) + " " + str(in0_block_w))
    logger.debug("in1 block h w " + str(in0_block_w) + " " + str(out_block_w))
    logger.debug("out block h w " + str(out_block_h) + " " + str(out_block_w))
    logger.debug("out subblock h w " + str(out_subblock_h) + " " + str(out_subblock_w))

    output_partial_idx = (
        num_reducer_partials // 2,
        num_reducer_partials // 2 + 1,
    )  # idx range for the master reducer cores
    if use_arbitrary_cores:
        # x, y
        if isinstance(grid, tuple):  # Generate random grid
            CORE_RANGE = [(x, y) for y in range(storage_grid[1]) for x in range(storage_grid[0])]
            random.shuffle(CORE_RANGE)
        else:  # Use custom grid
            mapping = get_physical_to_logical_core_mapping(device)
            CORE_RANGE = [mapping[physical_coord] for physical_coord in grid]

        core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in CORE_RANGE
            ]
        )

        output_grid = CORE_RANGE[ring_size * output_partial_idx[0] : ring_size * output_partial_idx[1]]
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in output_grid
            ]
        )
    else:
        core_range_set = ttnn.num_cores_to_corerangeset(num_cores, storage_grid, row_wise=True)

        output_grid = [(x, y) for y in range(storage_grid[1]) for x in range(storage_grid[0])]
        output_grid = output_grid[ring_size * output_partial_idx[0] : ring_size * output_partial_idx[1]]
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in output_grid
            ]
        )

    N_shard = N // ring_size * n_chunks

    in0_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_range_set,
            [M, K // num_cores],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    in1_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED if num_reducer_partials > 1 else ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_range_set,
            [K // num_reducer_partials, N_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    output_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            output_core_range_set,
            [M, N_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    in0 = torch.randn(in0_shape)
    in1 = torch.randn(in1_shape).reshape(1, 1, K, N * n_chunks)
    # in1[:, :, :, N:] = 0.0  # Pad with zeros

    in0_t = ttnn.from_torch(
        in0,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=in0_dtype,
        memory_config=in0_sharded_mem_config,
    )
    in1_t = ttnn.from_torch(
        in1,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=in1_dtype,
        memory_config=in1_sharded_mem_config,
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=storage_grid,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=activation,
        mcast_in0=False,
        gather_in0=True,
        num_reducer_partials=num_reducer_partials,
        n_chunks=n_chunks,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=fp32_acc_mode,
        packer_l1_acc=packer_l1_acc,
        dst_full_sync_en=True,
    )

    for _ in range(num_iters):
        output_t = ttnn.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=output_sharded_mem_config,
            compute_kernel_config=compute_kernel_config,
        )
    test = output_t[:, :, :, :3584]
    logger.info(test)

    tt_out = ttnn.to_torch(output_t)

    pt_out = in0 @ in1
    if activation:
        act_fnc = torch.nn.functional.silu if activation == ttnn.UnaryOpType.SILU else torch.nn.functional.relu
        pt_out = act_fnc(pt_out)

    passing, output = comp_pcc(pt_out, tt_out, pcc_threshold)
    logger.info(output)

    # breakpoint()
    assert passing

    # Check program cache
    assert device.num_program_cache_entries() == 1  # Only 1 op


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "B, M, K, N, in0_dtype, in1_dtype, fidelity, packer_l1_acc, fp32_acc_mode, grid, num_reducer_partials, n_chunks",
    [
        # 6 partials on 24 cores
        # (1, 32, 2304 // 6, 3584 // 4, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (4, 1), 1, 1),
        # (
        #     1,
        #     32,
        #     2304,
        #     3584 // 4,
        #     ttnn.bfloat16,
        #     ttnn.bfloat4_b,
        #     ttnn.MathFidelity.LoFi,
        #     True,
        #     True,
        #     (8, 3),
        #     6,
        #     1,
        # ),
        # # 3 partials on 24 cores
        # (
        #     1,
        #     32,
        #     2304 // 3,
        #     (3584 + 512) // 4,
        #     ttnn.bfloat16,
        #     ttnn.bfloat4_b,
        #     ttnn.MathFidelity.LoFi,
        #     True,
        #     True,
        #     (8, 1),
        #     1,
        #     1,
        # ),
        # (
        #     1,
        #     32,
        #     2304,
        #     (3584 + 512) // 4,
        #     ttnn.bfloat16,
        #     ttnn.bfloat4_b,
        #     ttnn.MathFidelity.LoFi,
        #     True,
        #     True,
        #     (8, 3),
        #     3,
        #     1,
        # ),
        # # TG prefetch case
        # (1, 32, 2304, (3584) * 2, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 3), 3, 4), # 22 us
        # (1, 32, 2304, (3840) * 2, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 3), 1, 2), # 22 us
        (1, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 3), 1, 1),
        # (1, 32, 3840, 2304, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, True, (8, 3), 1, 1), # 15 us
        # (1, 32, 3840, 3072, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, True, (8, 3), 6, 1), # 48 us
        # (1, 32, 3840, 3072, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, True, (8, 3), 3, 4), # 22 us
        # 1D weight fracturing
        # ff1/3
        # (1, 32, 8192 + 256, 1536, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 3), 1, 1),
        # (1, 32, 8192 + 256, 1024, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 3), 3, 1),
        # ff2
        # (1, 32, 1536, 8192 + 256, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.HiFi2, True, False, (8, 3), 1, 1),
        # (1, 32, 1536, 8192, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, False, (8, 3), 3, 1),
        # (1, 32, 1536, 8192, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, False, (8, 3), 3, 8),
        # do
        # (1, 32, 768, 8192 + 256, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 3), 1, 1),
        # (1, 32, 768, 8192, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 3), 3, 1),
        # do/qkv (fixed)
        # (1, 32, 8192 + 256, 768, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, True, (8, 3), 1, 1),
        # (1, 32, 8192 + 256, 512, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, True, (8, 3), 3, 1),
        # (
        #     1,
        #     32,
        #     2304,
        #     (3584 + 256),
        #     ttnn.bfloat16,
        #     ttnn.bfloat4_b,
        #     ttnn.MathFidelity.LoFi,
        #     True,
        #     True,
        #     (8, 3),
        #     4,
        #     4,
        # ),  # fails (might be an issue with a bunch of partials, need to investigate)
        # (
        #     1,
        #     32,
        #     2304,
        #     (3584 + 256),
        #     ttnn.bfloat16,
        #     ttnn.bfloat4_b,
        #     ttnn.MathFidelity.LoFi,
        #     True,
        #     True,
        #     (8, 3),
        #     4,
        #     1,
        # ),
        # (
        #     1,
        #     32,
        #     (2048 + 128 + 64),
        #     (3584),
        #     ttnn.bfloat16,
        #     ttnn.bfloat4_b,
        #     ttnn.MathFidelity.LoFi,
        #     True,
        #     True,
        #     (7, 2),
        #     1,
        #     4,
        # ),
        # (
        #     1,
        #     32,
        #     2304,
        #     (3584 + 256),
        #     ttnn.bfloat16,
        #     ttnn.bfloat4_b,
        #     ttnn.MathFidelity.LoFi,
        #     True,
        #     True,
        #     (8, 3),
        #     3,
        #     1,
        # ),
        # (
        #     1,
        #     32,
        #     32 * 2,
        #     5 * 32 * 2,
        #     ttnn.bfloat16,
        #     ttnn.bfloat4_b,
        #     ttnn.MathFidelity.LoFi,
        #     True,
        #     True,
        #     (2, 1),
        #     2,
        #     2,
        # ),
        # (
        #     1,
        #     32,
        #     32 * 3,
        #     5 * 32,
        #     ttnn.bfloat16,
        #     ttnn.bfloat4_b,
        #     ttnn.MathFidelity.LoFi,
        #     True,
        #     True,
        #     (3, 1),
        #     3,
        #     1,
        # ),
        # (
        #     1,
        #     32,
        #     32 * 4,
        #     3 * 32 * 2,
        #     ttnn.bfloat16,
        #     ttnn.bfloat4_b,
        #     ttnn.MathFidelity.LoFi,
        #     True,
        #     True,
        #     (4, 1),
        #     4,
        #     2,
        # ),  # fails
        # # Check if multi-batch works
        # (
        #     1,
        #     32,
        #     2304,
        #     (3584 + 2048 + 512),
        #     ttnn.bfloat16,
        #     ttnn.bfloat4_b,
        #     ttnn.MathFidelity.LoFi,
        #     True,
        #     True,
        #     (8, 3),
        #     1,
        #     4,
        # ),
        # Showcasing speedups using inner-dim paralellism
        # (1, 32, (2048+512+128), (3584), ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (7, 4), 1, 4),
        # (1, 32, (2048+512+128), (3584) // 4, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (7, 4), 1, 1),
        # (1, 32, (2048+512+128), (3584) // 4, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (7, 4), 4, 1),
        # (1, 32, 2048, 512, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 1), 1, 1),
        # (1, 32, 2048, 512, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 2), 2, 1),
        # (1, 32, 2048, 512, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 4), 4, 1),
        # (1, 32, 2048, 512, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 8), 8, 1),
        # 32, 4096, 2048
        # (1, 32, 4096, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 1), 1, 1),
        # (1, 32, 4096, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 1), 2, 1),
        # (1, 32, 4096, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 1), 4, 1),
        # (1, 32, 4096, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 2), 1, 1),
        # (1, 32, 4096, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 2), 2, 1),
        # (1, 32, 4096, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 2), 4, 1),
        # (1, 32, 4096, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 4), 1, 1),
        # (1, 32, 4096, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 4), 2, 1),
        # (1, 32, 4096, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 4), 4, 1),
        # (1, 32, 4096, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 8), 1, 1),
        # (1, 32, 4096, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 8), 2, 1),
        # (1, 32, 4096, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 8), 4, 1),
        # # 64, 1024, 1024
        # (1, 64, 2048, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 1), 1, 1),
        # (1, 64, 2048, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 1), 2, 1),
        # (1, 64, 2048, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 1), 4, 1),
        # (1, 64, 2048, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 2), 1, 1),
        # (1, 64, 2048, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 2), 2, 1),
        # (1, 64, 2048, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 2), 4, 1),
        # (1, 64, 2048, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 4), 1, 1),
        # (1, 64, 2048, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 4), 2, 1),
        # (1, 64, 2048, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 4), 4, 1),
        # (1, 64, 2048, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 8), 1, 1),
        # (1, 64, 2048, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 8), 2, 1),
        # (1, 64, 2048, 2048, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 8), 4, 1),
        # # 23, 2048, 1024
        # (1, 32, 2048, 1024, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 1), 1, 1),
        # (1, 32, 2048, 1024, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 1), 2, 1),
        # (1, 32, 2048, 1024, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 1), 4, 1),
        # (1, 32, 2048, 1024, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 2), 1, 1),
        # (1, 32, 2048, 1024, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 2), 2, 1),
        # (1, 32, 2048, 1024, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 2), 4, 1),
        # (1, 32, 2048, 1024, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 4), 1, 1),
        # (1, 32, 2048, 1024, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 4), 2, 1),
        # (1, 32, 2048, 1024, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 4), 4, 1),
        # (1, 32, 2048, 1024, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 8), 1, 1),
        # (1, 32, 2048, 1024, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 8), 2, 1),
        # (1, 32, 2048, 1024, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 8), 4, 1),
        # # 32, 4096, 2048
        # (1, 32, 4096, 512, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (4, 1), 1, 1),
        # (1, 32, 4096, 512, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (4, 1), 2, 1),
        # (1, 32, 4096, 512, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (4, 1), 4, 1),
        # (1, 32, 4096, 512, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 1), 1, 1),
        # (1, 32, 4096, 512, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 1), 2, 1),
        # (1, 32, 4096, 512, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 1), 4, 1),
        # (1, 32, 4096, 512, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 2), 1, 1),
        # (1, 32, 4096, 512, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 2), 2, 1),
        # (1, 32, 4096, 512, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 2), 4, 1),
        # (1, 32, 4096, 512, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 4), 1, 1),
        # (1, 32, 4096, 512, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 4), 2, 1),
        # (1, 32, 4096, 512, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 4), 4, 1),
        # # 32, 1024, 2048
        # (1, 32, 1024, 4096, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (4, 1), 1, 1),
        # (1, 32, 1024, 4096, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (4, 1), 2, 1),
        # (1, 32, 1024, 4096, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (4, 1), 4, 1),
        # (1, 32, 1024, 4096, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 1), 1, 1),
        # (1, 32, 1024, 4096, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 1), 2, 1),
        # (1, 32, 1024, 4096, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 1), 4, 1),
        # (1, 32, 1024, 4096, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 2), 1, 1),
        # (1, 32, 1024, 4096, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 2), 2, 1),
        # (1, 32, 1024, 4096, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 2), 4, 1),
        # (1, 32, 1024, 4096, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 4), 1, 1),
        # (1, 32, 1024, 4096, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 4), 2, 1),
        # (1, 32, 1024, 4096, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 4), 4, 1),
        # # FF1 shapes (ish)
        # (1, 32, 2304, 896, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (4, 1), 1, 1),
        # (1, 32, 2304, 896, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (4, 1), 2, 1),
        # (1, 32, 2304, 896, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (4, 1), 4, 1),
        # (1, 32, 2304, 896 + 64, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (6, 1), 1, 1),
        # (1, 32, 2304, 896 + 64, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (6, 1), 2, 1),
        # (1, 32, 2304, 896 + 64, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (6, 1), 3, 1),
        # (1, 32, 2048 + 128 + 64, 896, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (7, 2), 1, 1),
        # (1, 32, 2048 + 128 + 64, 896, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (7, 2), 2, 1),
        # (1, 32, 2048 + 128 + 64, 896, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (7, 2), 7, 1),
        # (1, 32, 2304, 896 + 512 + 128, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 3), 1, 1),
        # (1, 32, 2304, 896 + 512 + 128, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 3), 2, 1),
        # (1, 32, 2304, 896 + 512 + 128, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 3), 3, 1),
        # (1, 32, (2304), (3584 + 256), ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (6, 1), 1, 4),
        # (1, 32, (2304), (3584 + 256), ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (6, 3), 3, 4),
        # (1, 32, 2304, (3584 + 1024), ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 3), 2, 4),
    ],
)
@pytest.mark.parametrize(
    "activation",
    [
        None,
        # ttnn.UnaryOpType.SILU,
    ],
)
@pytest.mark.parametrize(
    "use_arbitrary_cores",
    [False],
)
def test_multi_core_matmul_1d_reduce_wh(
    device,
    in0_dtype,
    in1_dtype,
    fidelity,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    B,
    M,
    K,
    N,
    activation,
    grid,
    num_reducer_partials,
    n_chunks,
    use_arbitrary_cores,
    use_program_cache,
    function_level_defaults,
):
    run_multi_core_matmul_1d(
        device,
        in0_dtype,
        in1_dtype,
        fidelity,
        has_bias,
        fp32_acc_mode,
        packer_l1_acc,
        B,
        M,
        K,
        N,
        activation,
        grid,
        use_arbitrary_cores,
        num_iters=1,
        num_reducer_partials=num_reducer_partials,
        n_chunks=n_chunks,
    )


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.skipif(is_blackhole(), reason="Test suite for GS only")
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "B, M, K, N, in0_dtype, in1_dtype, fidelity, packer_l1_acc, fp32_acc_mode, grid",
    [
        # # 32, 2304, 3840 (PREFETCHER), only works on TG
        # (1, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, PREFETCHER_GRID),
        # 32, 2304, 3840
        (1, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi, False, False, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, False, True, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, False, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, False, False, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, True, False, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, False, True, (8, 3)),
        # 256, 1024, 8192
        (1, 256, 1024, 8192, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.HiFi4, True, True, (8, 4)),
        # 256, 1024, 8192
        (1, 256, 1024, 8192, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.HiFi4, True, True, (8, 4)),
        # # 128, 8192, 2048
        # (1, 128, 8192, 2048, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.MathFidelity.HiFi2, True, True, (8, 8)),
        # # 128, 8192, 2048
        # (1, 128, 8192, 2048, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.MathFidelity.HiFi2, True, False, (8, 8)),
        # # 128, 8192, 2048
        # (1, 128, 8192, 2048, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.MathFidelity.HiFi2, False, True, (8, 8)), # Fails with 0.98 PCC
        # 32, 64, 64
        (1, 32, 64, 64, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, True, True, (2, 1)),
        # 32, 64, 64
        (11, 32, 64, 64, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, True, True, (2, 1)),
    ],
)
@pytest.mark.parametrize(
    "activation",
    [
        None,
        ttnn.UnaryOpType.SILU,
        ttnn.UnaryOpType.RELU,
    ],
)
@pytest.mark.parametrize(
    "use_arbitrary_cores",
    [False, True],
)
@pytest.mark.parametrize(
    "num_iters",
    [1, 3],
)
def test_multi_core_matmul_1d_wh(
    device,
    in0_dtype,
    in1_dtype,
    fidelity,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    B,
    M,
    K,
    N,
    activation,
    grid,
    use_arbitrary_cores,
    num_iters,
    use_program_cache,
    function_level_defaults,
):
    run_multi_core_matmul_1d(
        device,
        in0_dtype,
        in1_dtype,
        fidelity,
        has_bias,
        fp32_acc_mode,
        packer_l1_acc,
        B,
        M,
        K,
        N,
        activation,
        grid,
        use_arbitrary_cores,
        num_iters,
    )


@pytest.mark.skipif(is_wormhole_b0(), reason="Test suite for GS only")
@pytest.mark.skipif(is_blackhole(), reason="Test suite for GS only")
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "B, M, K, N, in0_dtype, in1_dtype, fidelity, packer_l1_acc, fp32_acc_mode, grid",
    [
        # 32, 2304, 3840
        (1, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, False, False, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi, False, False, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, False, False, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, False, False, (8, 3)),
        # 256, 1024, 8192
        (1, 256, 1024, 8192, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.HiFi4, False, False, (8, 4)),
        # 128, 8192, 2048
        (1, 128, 4096, 2048, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.MathFidelity.HiFi2, False, False, (8, 8)),
        # 32, 64, 64
        (1, 32, 64, 64, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, False, False, (2, 1)),
        # 32, 64, 64
        (11, 32, 64, 64, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, False, False, (2, 1)),
    ],
)
@pytest.mark.parametrize(
    "activation",
    [
        None,
        ttnn.UnaryOpType.SILU,
        ttnn.UnaryOpType.RELU,
    ],
)
@pytest.mark.parametrize(
    "use_arbitrary_cores",
    [False, True],
)
@pytest.mark.parametrize(
    "num_iters",
    [1, 3],
)
def test_multi_core_matmul_1d_gs(
    device,
    in0_dtype,
    in1_dtype,
    fidelity,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    B,
    M,
    K,
    N,
    activation,
    grid,
    use_arbitrary_cores,
    num_iters,
    use_program_cache,
    function_level_defaults,
):
    run_multi_core_matmul_1d(
        device,
        in0_dtype,
        in1_dtype,
        fidelity,
        has_bias,
        fp32_acc_mode,
        packer_l1_acc,
        B,
        M,
        K,
        N,
        activation,
        grid,
        use_arbitrary_cores,
        num_iters,
        pcc_threshold=0.96,
    )
