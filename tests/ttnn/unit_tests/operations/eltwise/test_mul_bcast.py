# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from torch.nn import functional as F


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_mul_channel_bcast_repeat(device, h, w):
    torch_input_tensor_a = torch.rand((16, 16, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((16, 1, h, w), dtype=torch.bfloat16)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.mul(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    torch_output_tensor = torch.mul(torch_input_tensor_a, torch_input_tensor_b)
    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_mul_batch_bcast_repeat(device, h, w):
    torch_input_tensor_a = torch.rand((1, 16, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((16, 16, h, w), dtype=torch.bfloat16)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.mul(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    torch_output_tensor = torch.mul(torch_input_tensor_a, torch_input_tensor_b)
    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 960])),),
)
@pytest.mark.parametrize(
    "input_range",
    [
        {"high": 100, "low": -100},
    ],
)
def test_mul_width_sharded(device, input_shapes, input_range):
    high = input_range["high"]
    low = input_range["low"]
    in_data1 = torch.rand((input_shapes), dtype=torch.bfloat16) * (high - low) + low
    in_data2 = torch.rand((input_shapes), dtype=torch.bfloat16) * (high - low + 10) + low
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange((0, 0), (7, 7)),
        }
    )
    shard_spec = ttnn.ShardSpec(shard_grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardMode.PHYSICAL)
    input_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    input_tensor1 = ttnn.from_torch(
        in_data1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    input_tensor2 = ttnn.from_torch(
        in_data2,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    output_tensor = ttnn.multiply(
        input_tensor1,
        input_tensor2,
        memory_config=input_mem_config,
        input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)],
        use_legacy=True,
    )
    output_tensor = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn.multiply)
    golden_tensor = golden_function(torch.nn.functional.silu(in_data1), in_data2)

    # print(ttnn.silu(input_tensor1))
    # print("golden_tensor", golden_tensor)
    # print("output_tensor", output_tensor)
    pcc, pcc_msg = assert_with_pcc(golden_tensor, output_tensor, 0.999)
    assert pcc
