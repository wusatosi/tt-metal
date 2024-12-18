# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
import torch.nn.functional as F
from tests.ttnn.utils_for_testing import assert_with_pcc

SDPAProgramConfig = ttnn._ttnn.operations.transformer.SDPAProgramConfig


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test(device, reset_seeds):
    query = torch.randn([2, 24, 4429, 64])
    key = torch.randn([2, 24, 4429, 64])
    value = torch.randn([2, 24, 4429, 64])
    query_tt = ttnn.from_torch(
        query, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    key_tt = ttnn.from_torch(
        key, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    value_tt = ttnn.from_torch(
        value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=32,
        k_chunk_size=32,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    tt_out = ttnn.transformer.scaled_dot_product_attention(
        query_tt,
        key_tt,
        value_tt,
        is_causal=False,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )
    torch_out = F.scaled_dot_product_attention(query, key, value, is_causal=False)
    tt_out = ttnn.transformer.scaled_dot_product_attention(query_tt, key_tt, value_tt, is_causal=False)
    tt_out_torch = ttnn.to_torch(tt_out)
    assert_with_pcc(torch_out, tt_out_torch, 0.99)
