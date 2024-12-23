# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


<<<<<<< HEAD
@pytest.mark.parametrize("h", [32 * 64])  # number of cores that does the work
@pytest.mark.parametrize("w", [32 * 128])  # can go up to 128 shard width, number of tiles per core
def test_multiplyadd(device, h, w):
=======

@pytest.mark.parametrize(
    "batch_size",
    [
        1,
    ],
)
@pytest.mark.parametrize("h", [2 * 32])
@pytest.mark.parametrize("w", [32, 48, 64, 80, 96, 112, 128])
@pytest.mark.parametrize("c", [9 * 64])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
def test_multiplyadd(device, batch_size, h, w, c, n, dtype):
>>>>>>> Adding multiply add operation on three tilized tensors of arbitrary number of tiles
    torch.manual_seed(0)
    compute_grid_size = device.compute_with_storage_grid_size()

    torch_input_tensor1 = torch.randn((h, w), dtype=torch.bfloat16)
    torch_input_tensor2 = torch.randn((h, w), dtype=torch.bfloat16)
    torch_input_tensor3 = torch.randn((h, w), dtype=torch.bfloat16)

<<<<<<< HEAD
    tensor_memory_config = ttnn.create_sharded_memory_config_(
        (h, w),
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(compute_grid_size.y - 1, compute_grid_size.x - 1),
                )
            }
        ),
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
=======
    torch_input_tensor1 = torch.randn(batch_size, h, w, c, n, dtype=torch.bfloat16)
    torch_input_tensor2 = torch.randn(batch_size, h, w, c, n, dtype=torch.bfloat16)
    torch_input_tensor3 = torch.randn(batch_size, h, w, c, n, dtype=torch.bfloat16)
>>>>>>> Adding multiply add operation on three tilized tensors of arbitrary number of tiles

    golden_function = ttnn.get_golden_function(ttnn.multiplyadd)
    torch_output_tensor = golden_function(torch_input_tensor1, torch_input_tensor2, torch_input_tensor3)

    input_tensor1 = ttnn.from_torch(
        torch_input_tensor1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
<<<<<<< HEAD
        memory_config=tensor_memory_config,
    )

=======
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
>>>>>>> Adding multiply add operation for two tensors.
    input_tensor2 = ttnn.from_torch(
        torch_input_tensor2,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
<<<<<<< HEAD
        memory_config=tensor_memory_config,
    )

=======
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
>>>>>>> Adding multiply add operation for two tensors.
    input_tensor3 = ttnn.from_torch(
        torch_input_tensor3,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
<<<<<<< HEAD
        memory_config=tensor_memory_config,
    )

    output_tensor = ttnn.multiplyadd(input_tensor1, input_tensor2, input_tensor3)
    output_tensor = ttnn.to_torch(output_tensor)
=======
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.multiplyadd(input_tensor1, input_tensor2, input_tensor3)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
>>>>>>> Adding multiply add operation for two tensors.

    assert_with_pcc(torch_output_tensor, ttnn.to_torch(output_tensor), pcc=0.99)
