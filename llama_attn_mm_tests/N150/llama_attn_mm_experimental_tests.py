import torch
import ttnn
from models.utility_functions import (
    comp_pcc,
)
import math
import pytest
import tracy

TILE_SIZE = 32
DRAM_WEIGHT_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))})

SEQ_LENS = [
    128,
    256,
    512,
    1024,
    2048,
    4096,
    6144,
    8192,
    10240,
    12288,
    14336,
    16384,
    24576,
    32768,
    51200,
    65536,
    86016,
    131072,
]


def generate_kqv_projection_program_config(seq_len):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(7, 10),
        in0_block_w=8,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=2,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=max(
            1, 8 if seq_len >= 2048 else seq_len // TILE_SIZE // 8  # 8 rows
        ),  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
        per_core_N=math.ceil(1280 / 32 / 7),  # N / TILE_WIDTH / grid width
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=seq_len <= 2048,
    )


def generate_kqv_projection_program_config_new(seq_len):
    if seq_len < 4096:
        return generate_kqv_projection_program_config(seq_len)

    # if seq_len == 4096:
    #     # breakpoint()
    #     return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    #         compute_with_storage_grid_size=(7, 9),
    #         in0_block_w=8,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
    #         out_subblock_h=1,  # Must be divisible by per_core_M
    #         out_subblock_w=3,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
    #         out_block_h=19,
    #         out_block_w=6,
    #         per_core_M=19,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
    #         per_core_N=6,  # N / TILE_WIDTH / grid width
    #         transpose_mcast=False,
    #         fused_activation=None,
    #         fuse_batch=False,
    #     )

    def largest_divisor_less_than_k(n, k):
        # Start from k-1 and work down
        for i in range(k, 0, -1):
            if n % i == 0:
                return i

        # If no divisor is found (other than 1), return None
        # return None
        assert False, "No divisor fount"

    def count_prime_factors(n):
        if n <= 1:
            return 0

        count = 0

        # Check divisibility by 2
        while n % 2 == 0:
            count += 1
            n //= 2

        # Check divisibility by odd numbers starting from 3
        i = 3
        while i * i <= n:
            while n % i == 0:
                count += 1
                n //= i
            i += 2

        # If n > 1, it is a prime factor itself
        if n > 1:
            count += 1

        return count

    out_block_w = 6
    per_core_N = 6

    per_core_M = math.ceil(seq_len / (TILE_SIZE * 9))
    if per_core_M > 20:
        while count_prime_factors(per_core_M) < 4:
            per_core_M = per_core_M + 1

    out_block_h = largest_divisor_less_than_k(per_core_M, 16)
    # breakpoint()
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(7, 9),
        in0_block_w=16,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=3,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        per_core_M=per_core_M,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
        per_core_N=per_core_N,  # N / TILE_WIDTH / grid width
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def generate_wo_program_config(seq_len):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(7, 10),
        in0_block_w=8,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=2,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=max(1, 4 if seq_len >= 1024 else seq_len // TILE_SIZE // 8),  # 8~10 rows
        per_core_N=math.ceil(2048 / 32 / 7),  # N / TILE_WIDTH / grid width
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=seq_len <= 1024,
    )


def generate_wo_program_config_new(seq_len):
    if seq_len < 4096:
        return generate_wo_program_config(seq_len)

    def largest_divisor_less_than_k(n, k):
        # Start from k-1 and work down
        for i in range(k, 0, -1):
            if n % i == 0:
                return i

        # If no divisor is found (other than 1), return None
        # return None
        assert False, "No divisor fount"

    def count_prime_factors(n):
        if n <= 1:
            return 0

        count = 0

        # Check divisibility by 2
        while n % 2 == 0:
            count += 1
            n //= 2

        # Check divisibility by odd numbers starting from 3
        i = 3
        while i * i <= n:
            while n % i == 0:
                count += 1
                n //= i
            i += 2

        # If n > 1, it is a prime factor itself
        if n > 1:
            count += 1

        return count

    out_block_w = 10
    per_core_N = 10

    per_core_M = math.ceil(seq_len / (TILE_SIZE * 9))
    if per_core_M > 20:
        while count_prime_factors(per_core_M) < 4:
            per_core_M = per_core_M + 1

    out_block_h = largest_divisor_less_than_k(per_core_M, 20)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(7, 9),
        in0_block_w=16,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
        out_subblock_h=3 if out_block_h % 3 == 0 else 1,  # Must be divisible by per_core_M
        out_subblock_w=2
        if out_block_h % 3 == 0
        else 5,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        per_core_M=per_core_M,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
        per_core_N=per_core_N,  # N / TILE_WIDTH / grid width
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def create_dram_sharded_mem_config(k, n):
    """Create DRAM-sharded memory config for width-sharded tensors"""
    dram_cores = 12
    padded_size = math.ceil(n / (TILE_SIZE * dram_cores)) * (TILE_SIZE * dram_cores)
    shard_spec = ttnn.ShardSpec(DRAM_WEIGHT_GRID, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    SEQ_LENS,
)
def test_kqv_projection(device, seq_len):
    activations = (
        torch.randn((1, seq_len // 2048, 2048, 2048)) if (seq_len == 2048) else torch.randn((1, 1, seq_len, 2048))
    )
    wkqv = torch.randn((1, 1, 2048, 1280))

    golden = activations @ wkqv.squeeze()

    activations_tt = ttnn.from_torch(
        activations,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    wkqv_tt = ttnn.from_torch(
        wkqv,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if seq_len >= 4096 else create_dram_sharded_mem_config(2048, 1536),
        layout=ttnn.TILE_LAYOUT,
    )

    out_tt = ttnn.linear(
        activations_tt,
        wkqv_tt,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            dst_full_sync_en=True,
        ),
        program_config=generate_kqv_projection_program_config_new(seq_len),
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_torch = ttnn.to_torch(out_tt)
    passed, msg = comp_pcc(golden, out_torch, 0.99)
    assert passed, msg


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    SEQ_LENS,
)
def test_wo(device, seq_len):
    activations = (
        torch.randn((1, 1, seq_len // 1024, 1024, 1024))
        if 1024 <= seq_len < 4096
        else torch.randn((1, 1, seq_len, 1024))
    )
    wo = torch.randn((1, 1, 1024, 2048))

    golden = activations @ wo.squeeze()

    activations_tt = ttnn.from_torch(
        activations,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    wo_tt = ttnn.from_torch(
        wo,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG
        if seq_len >= 4096
        else create_dram_sharded_mem_config(8192 // 8, 9216 // 4),
        layout=ttnn.TILE_LAYOUT,
    )

    out_tt = ttnn.linear(
        activations_tt,
        wo_tt,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            dst_full_sync_en=True,
        ),
        program_config=generate_wo_program_config_new(seq_len),
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_torch = ttnn.to_torch(out_tt)
    passed, msg = comp_pcc(golden, out_torch, 0.99)
    assert passed, msg
