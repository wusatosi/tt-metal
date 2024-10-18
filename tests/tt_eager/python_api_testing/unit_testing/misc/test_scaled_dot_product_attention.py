# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import math
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import ttnn
from loguru import logger
import pytest
from models.utility_functions import skip_for_grayskull, skip_for_wormhole_b0, skip_for_blackhole


def is_watcher_enabled():
    return os.environ.get("TT_METAL_WATCHER") is not None


def run_test_sdpa_tt_ND(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=[1, 1],
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    Q = torch.eye(s, d).expand(b, nh, s, d) * 5
    K = torch.eye(s, d).expand(b, nkv, s, d) * 7
    V = torch.eye(s, d).expand(b, nkv, s, d) * 9
    E = torch.eye(s, d).expand(b, nkv, s, d) * 316
    Bug45 = torch.eye(s, d).expand(b, nkv, s, d) * 45
    Bug0 = torch.eye(s, d).expand(b, nkv, s, d) * 0

    # Print shapes of all inputs along with input names
    logger.debug(f"Q: {Q.shape}")
    logger.debug(f"K: {K.shape}")
    logger.debug(f"V: {V.shape}")

    tt_Q = ttnn.Tensor(Q, dtype).to(ttnn.TILE_LAYOUT).to(device)
    tt_K = ttnn.Tensor(K, dtype).to(ttnn.TILE_LAYOUT).to(device)
    tt_V = ttnn.Tensor(V, dtype).to(ttnn.TILE_LAYOUT).to(device)

    torch.set_printoptions(profile="full")
    for idx in range(3):
        tt_back = ttnn.transformer.scaled_dot_product_attention(
            tt_Q, tt_K, tt_V, is_causal=True, program_config=program_config, compute_kernel_config=compute_kernel_config
        )
        tt_back = tt_back.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        if torch.all(E.eq(tt_back)):
            print("Matched Expected")
        elif torch.all(Bug45.eq(tt_back)):
            print("Matched 45 Bug")
        elif torch.all(Bug0.eq(tt_back)):
            print("Matched 0 Bug")
        else:
            print("Unknown Error")
            print(tt_back)
            # torch.save(tt_back, "buggy_tensor_tiny.pt")


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", [32], ids=["q32"])
@pytest.mark.parametrize("k_chunk_size", [32], ids=["k32"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    ([1, 1, 1, 32, 32],),
)
def test_sdpa_nd(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    torch.set_printoptions(profile="full")
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")

    for idx in range(100):
        os.environ["TT_NOP_INSERT"] = str(35 + idx)
        ttnn.device.DisablePersistentKernelCache()
        run_test_sdpa_tt_ND(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype)
