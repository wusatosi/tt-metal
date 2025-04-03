# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from ttnn.distributed.distributed import get_mesh_device_core_grid


def test_my_op(mesh_device):
    input_tensor_shape = (1, 1, 512, 512)

    core_grid = get_mesh_device_core_grid(mesh_device)

    l1_shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))}
    )
    l1_h, l1_w = input_tensor_shape[2] // core_grid.x, input_tensor_shape[3] // core_grid.y

    shard_spec = ttnn.ShardSpec(l1_shard_grid, [l1_h, l1_w], ttnn.ShardOrientation.ROW_MAJOR)

    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    torch_input_tensor = torch.randn(input_tensor_shape)
    tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tensor = ttnn.to_device(tensor, mesh_device)

    ttnn.my_new_op(tensor, sharded_mem_config, ttnn.bfloat16, keep_l1_aligned=True)
