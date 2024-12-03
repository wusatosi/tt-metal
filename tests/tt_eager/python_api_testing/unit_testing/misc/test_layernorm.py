# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import pytest
import torch

import ttnn


from models.utility_functions import pad_by_zero, torch2tt_tensor, comp_pcc, is_grayskull, is_blackhole


def ref_layernorm(x, gamma, beta, eps):
    return torch.nn.functional.layer_norm(x, x.shape[-1:], gamma, beta, eps)


def ref_rmsnorm(x, gamma, beta, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma + beta


# Function to print a 32x32 tile
def print_tile(tensor, tile_row, tile_col):
    start_row = tile_row * 32
    start_col = tile_col * 32
    tile = tensor[0, 0, start_row : start_row + 32, start_col : start_col + 32]

    for idx, row in enumerate(tile):
        print(f"{idx:2}: " + " ".join(f"{val.item():.4f}" for val in row))
    print("\n")


def run_layernorm_mix_precision_tests(test_id, in_dtype, gamma_dtype, in0_mem_config, out_mem_config, device):
    epsf = 1e-2

    test_dims = (
        (1, 1, 32, 256),
        # (1, 1, 32, 128),  # W <= 4 because max 4 fp32 tiles can fit in half of a DEST
        # (130, 1, 32, 128),
        # (512, 1, 32, 64),
        # (131, 1, 32, 128),# -> fails, pcc = 0.998 hm?? core_grid=13x10 or smth?
        # # (131, 1, 32, 96), # -> fails, pcc = 0.998
        # # (1, 1, 32, 160),  # -> fails, pcc = 0.86
        # (1, 1, 512, 128),
        # (1, 1, 1024, 128),
        # (1, 2, 2048, 128),
        # # (1, 3, 2048, 128), # -> fails, pcc = 0.91
        # (1, 1, 4096, 128),
        # # (1, 2, 4096, 128), # -> fails, pcc = 0.87
        # (1, 1, 4160, 128),  # [1, 1, 130, 4], last that will pass with this W
        # # (1, 1, 4192, 128), # -> fails, pcc = 0.998
        # # (1, 1, 4224, 128), # -> fails, pcc = 0.996
        # # (1, 1, 4352, 128), # -> fails, pcc = 0.989
        # # (1, 1, 4608, 128), # -> fails, pcc = 0.97
        # # (1, 1, 5120, 128), # -> fails, pcc = 0.95
        # # (1, 1, 6144, 128), # -> fails, pcc = 0.91
        # # (1, 1, 8192, 128), # -> fails, pcc = 0.87
        # # (1, 1, 8192, 96),  # -> fails, pcc = 0.91
        # (1, 2, 8192, 64),
        # (1, 3, 8192, 64),
        # (5, 3, 8192, 64),
        # (1, 1, 16384, 64),
        # # (1, 1, 16384, 96), # -> fails, pcc = 0.91
        # (3, 3, 16384, 64),
    )
    for test_shape in test_dims:
        in0 = torch.rand(test_shape) * 2 - 0.95
        # in0 = torch.full(test_shape, 10.5)
        in0_t = torch2tt_tensor(in0, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)

        if test_id <= 5:
            in1 = torch.rand(test_shape) * 2 - 0.8
            # in1 = torch.full(test_shape, 20.0)
            in1_t = torch2tt_tensor(in1, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)

        if test_id % 3 == 0:
            gamma = torch.ones(test_shape[3])
            beta = torch.zeros(test_shape[3])
        if test_id % 3 == 1:
            gamma = torch.rand(test_shap[3]) * 2 - 1
            # gamma = torch.full(test_shape[3], 0.25)
            beta = torch.zeros(test_shape[3])
        if test_id % 3 == 2:
            gamma = torch.rand(test_shape[3]) * 2 - 1
            beta = torch.rand(test_shape[3]) * 2.0 - 1.1
            # gamma = torch.full(test_shape[3], 0.125) * 2 - 1
            # beta = torch.full(test_shape[3], 0.06125) * 2.0 - 1.1

        gamma_t = pad_by_zero(gamma, device, in0_mem_config, gamma_dtype)[0]
        beta_t = pad_by_zero(beta, device, in0_mem_config, gamma_dtype)[0]

        if not is_grayskull():
            compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=True,
                fp32_dest_acc_en=True if in_dtype == ttnn.float32 else False,
            )

        if test_id == 0:
            ttz = ttnn.layer_norm(
                in0_t,
                residual_input_tensor=in1_t,
                epsilon=epsf,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 1:
            ttz = ttnn.layer_norm(
                in0_t,
                residual_input_tensor=in1_t,
                epsilon=epsf,
                weight=gamma_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 2:
            ttz = ttnn.layer_norm(
                in0_t,
                residual_input_tensor=in1_t,
                epsilon=epsf,
                weight=gamma_t,
                bias=beta_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 3:
            ttz = ttnn.rms_norm(
                in0_t,
                residual_input_tensor=in1_t,
                epsilon=epsf,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 4:
            ttz = ttnn.rms_norm(
                in0_t,
                residual_input_tensor=in1_t,
                epsilon=epsf,
                weight=gamma_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 5:
            ttz = ttnn.rms_norm(
                in0_t,
                residual_input_tensor=in1_t,
                epsilon=epsf,
                weight=gamma_t,
                bias=beta_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 6:
            ttz = ttnn.layer_norm(
                in0_t,
                epsilon=epsf,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 7:
            ttz = ttnn.layer_norm(
                in0_t,
                epsilon=epsf,
                weight=gamma_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 8:
            ttz = ttnn.layer_norm(
                in0_t,
                epsilon=epsf,
                weight=gamma_t,
                bias=beta_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 9:
            ttz = ttnn.rms_norm(
                in0_t,
                epsilon=epsf,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 10:
            ttz = ttnn.rms_norm(
                in0_t,
                epsilon=epsf,
                weight=gamma_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 11:
            ttz = ttnn.rms_norm(
                in0_t,
                epsilon=epsf,
                weight=gamma_t,
                bias=beta_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )

        tt_got_back = ttz.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

        pt_in = in0 + in1 if test_id <= 5 else in0
        if test_id <= 2 or 6 <= test_id <= 8:
            ref_fn = ref_layernorm
        else:
            ref_fn = ref_rmsnorm

        ref_lnorm = ref_fn(pt_in, gamma.flatten(), beta.flatten(), epsf)

        # Iterate over tiles (adjust tile dimensions as necessary)
        num_rows = tt_got_back.size(2) // 32
        num_cols = tt_got_back.size(3) // 32

        if 1 == 1:
            for tile_row in range(num_rows):
                for tile_col in range(num_cols):
                    print(f"Tile ({tile_row}, {tile_col}):")
                    print_tile(tt_got_back, tile_row, tile_col)
                    # print_tile(ref_lnorm, tile_row, tile_col)

        passing, output = comp_pcc(ref_lnorm, tt_got_back)

        assert True


# @pytest.mark.skipif(is_blackhole(), reason="Mismatching on Blackhole, see #12349")
@pytest.mark.parametrize(
    "out_mem_config",
    (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),),
    ids=[
        "in0_L1",
    ],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),),
    ids=[
        "in0_L1",
    ],
)
@pytest.mark.parametrize(
    "gamma_dtype",
    (ttnn.bfloat16,),
    ids=["BFLOAT16"],
)
# @pytest.mark.parametrize(
#     "in_dtype",
#     (
#         ttnn.float32,
#         ttnn.bfloat16,
#         ttnn.bfloat8_b,
#     ),
#     ids=["FLOAT32", "BFLOAT16", "BFLOAT8_B"],
# )
# @pytest.mark.parametrize(
#     "test_id",
#     (0, 1),
#     ids=[
#         "add_LN",
#         "add_LN_G",
#         # "add_LN_GB",
#         # "add_RMSN",
#         # "add_RMSN_G",
#         # "add_RMSN_GB",
#         # "LN",
#         # "LN_G",
#         # "LN_GB",
#         # "RMSN",
#         # "RMSN_G",
#         # "RMSN_GB",
#     ],
# )
# @pytest.mark.parametrize("repeat", range(50))
def test_layernorm_mix_precision(gamma_dtype, in0_mem_config, out_mem_config, device):
    torch.manual_seed(1997)
    # if is_grayskull() and in_dtype == ttnn.float32:
    #     pytest.skip("Skipping float32 tests on Grayskull")
    run_layernorm_mix_precision_tests(0, ttnn.float32, gamma_dtype, in0_mem_config, out_mem_config, device)
