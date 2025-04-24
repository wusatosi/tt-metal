# SPDX-FileCopyrightText: 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.utility_functions import (
    comp_pcc,
)


@pytest.mark.parametrize(
    "seq_len",
    [
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
    ],
)
def test_ff1(device, seq_len):
    in0_shape = (1, 1, seq_len, 2048)
    in1_shape = (1, 1, 2048, 3584)

    in0_torch = torch.randn(in0_shape)
    in1_torch = torch.randn(in1_shape)

    golden = torch.nn.functional.linear(in0_torch, in1_torch.permute(0, 1, 3, 2).reshape(3584, 2048))

    in0_tt = ttnn.from_torch(
        in0_torch,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    out_tt = ttnn.linear(
        in0_tt,
        in1_tt,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            dst_full_sync_en=True,
        ),
    )

    out_torch = ttnn.to_torch(out_tt)
    passed, msg = comp_pcc(golden, out_torch, 0.99)
    assert passed, msg


@pytest.mark.parametrize(
    "seq_len",
    [
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
    ],
)
def test_ff2(device, seq_len):
    in0_shape = (1, 1, seq_len, 3584)
    in1_shape = (1, 1, 3584, 2048)

    in0_torch = torch.randn(in0_shape)
    in1_torch = torch.randn(in1_shape)

    golden = torch.nn.functional.linear(in0_torch, in1_torch.permute(0, 1, 3, 2).reshape(2048, 3584))

    in0_tt = ttnn.from_torch(
        in0_torch,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    out_tt = ttnn.linear(
        in0_tt,
        in1_tt,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            dst_full_sync_en=True,
        ),
    )

    out_torch = ttnn.to_torch(out_tt)
    passed, msg = comp_pcc(golden, out_torch, 0.99)
    assert passed, msg
