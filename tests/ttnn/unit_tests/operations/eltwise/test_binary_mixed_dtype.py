# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import random
import ttnn


@pytest.mark.parametrize(
    "shapes",
    [
        [[1, 1, 1024, 1024], [1, 1, 1024, 1024]],  # no bcast
        [[1, 2, 4, 3, 320, 128], [1, 2, 4, 3, 320, 128]],
        [[1, 16, 8, 49, 49], [1, 16, 1, 49, 49]],  # channel bcast
        [[1, 4, 16, 49, 49], [1, 4, 1, 49, 49]],
        [[1, 64, 4, 49, 49], [1, 64, 1, 49, 49]],
        [[1, 2, 4, 1, 2, 2], [1, 2, 1, 1, 2, 2]],  # batch bcast
        [[2, 2, 2, 1, 128, 256], [2, 2, 1, 1, 128, 256]],
        [[2, 2, 2, 1, 1, 256], [2, 2, 2, 1, 128, 256]],  # row_a bcast
        [[2, 2, 2, 1, 128, 256], [2, 2, 2, 1, 1, 256]],  # row_b bcast
        [[2, 2, 2, 1, 128, 1], [2, 2, 1, 1, 128, 256]],  # col_a bcast
        [[2, 2, 2, 1, 128, 256], [2, 2, 1, 1, 128, 1]],  # col_b bcast
        [[2, 2, 2, 1, 1, 256], [2, 2, 1, 1, 128, 1]],  # row_a col_b
        [[2, 2, 2, 1, 128, 256], [2, 2, 1, 1, 128, 256]],  # row_b col_A
        [[4, 8, 64, 512], [1, 8, 64, 1]],  # col_b, N_b
        [[4, 8, 64, 512], [4, 1, 1, 512]],  # row_b, C_b
        [[4, 8, 64, 512], [4, 8, 1, 1]],  # B scalar
        [[1, 8, 64, 1], [4, 8, 64, 512]],  # col_a, N_a
        [[4, 1, 1, 512], [4, 8, 64, 512]],  # row_a, C_a
        [[4, 8, 1, 1], [4, 8, 64, 512]],  # A scalar
        [[4, 8, 1, 512], [4, 8, 64, 1]],  # row_a, col_b
        [[4, 8, 64, 1], [4, 8, 1, 512]],  # row_b, col_a
    ],
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        # ttnn.experimental.add,
        # ttnn.experimental.sub,
        # ttnn.experimental.mul,
        ttnn.experimental.div,
        # ttnn.experimental.rsub,
        # ttnn.experimental.eq, # fails
        # ttnn.experimental.ne,
        # ttnn.experimental.gt,
        # ttnn.experimental.gte,
        # ttnn.experimental.lt,
        # ttnn.experimental.lte,
        # ttnn.experimental.logical_or,
        # ttnn.experimental.logical_xor, # fails
        # ttnn.experimental.logical_and,
        # ttnn.experimental.ldexp,  # fails
        # ttnn.experimental.logaddexp,
        # ttnn.experimental.logaddexp2,
        # ttnn.experimental.squared_difference,
        # ttnn.experimental.bias_gelu,
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        {"a": ttnn.bfloat16, "b": ttnn.float32},
        {"a": ttnn.float32, "b": ttnn.bfloat16},
    ],
)
def test_binary_float_mixed_dtype(device, shapes, ttnn_fn, input_dtype):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shapes[0], dtype=torch.float32) * 100 - 50
    torch_input_tensor_b = None
    if ttnn_fn == ttnn.experimental.div:
        torch_input_tensor_b = torch.rand(shapes[1], dtype=torch.float32) * 59 + 1
    else:
        torch_input_tensor_b = torch.rand(shapes[1], dtype=torch.float32) * 80 - 40

    golden_fn = ttnn.get_golden_function(ttnn_fn)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_dtype["a"],
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_dtype["b"],
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn_fn(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.999
