# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def run_test(device, hw, out_channels, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((hw, out_channels), dtype=torch.bfloat16)

    num_cores = 64
    core_range_set = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size())
    shard_shape = [hw // num_cores, out_channels if out_channels >= 32 else 32]

    # memory_config = ttnn.DRAM_MEMORY_CONFIG # this makes the test pass, the one below makes it fail
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_range_set,
            shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    output_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    output_tensor = ttnn.to_layout(
        output_tensor, ttnn.ROW_MAJOR_LAYOUT
    )  # Commenting this line makes the test pass with sharded mem config
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor, pcc)


@pytest.mark.parametrize("hw", [2048])
@pytest.mark.parametrize("out_channels", [2])
def test_to_layout(device, hw, out_channels):
    torch.manual_seed(0)
    run_test(device, hw, out_channels, pcc=0.991)
