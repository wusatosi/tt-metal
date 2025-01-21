# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def test_reshard(
    device,
    input_height,
    input_width,
    input_memory_layout,
    input_sharded_memory_config_args,
    output_sharded_memory_config_args,
):
    if isinstance(input_sharded_memory_config_args["core_grid"], (ttnn.CoreGrid)):
        if device.core_grid.y < input_sharded_memory_config_args["core_grid"].y:
            return
        if device.core_grid.y < output_sharded_memory_config_args["core_grid"].y:
            return

    input_shape = [1, 1, input_height, input_width]

    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    interleaved_input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=input_memory_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    input_shard_memory_config = ttnn.create_sharded_memory_config(input_shape, **input_sharded_memory_config_args)
    output_shard_memory_config = ttnn.create_sharded_memory_config(input_shape, **output_sharded_memory_config_args)

    # interleaved_to_sharded
    sharded_input_tensor = ttnn.to_memory_config(interleaved_input_tensor, input_shard_memory_config)
    # dump profile here to get rid of setup events
    ttnn.DumpDeviceProfiler(device)

    ## reshard
    sharded_output_tensor = ttnn.to_memory_config(sharded_input_tensor, output_shard_memory_config)
    ttnn.DumpDeviceProfiler(device)

    ## sharded_to_interleaved
    # interleaved_output_tensor = ttnn.to_memory_config(sharded_output_tensor, ttnn.DRAM_MEMORY_CONFIG)


def main():
    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    test_reshard(
        device,
        1024,
        1024,
        ttnn.TILE_LAYOUT,
        dict(core_grid=ttnn.CoreGrid(y=2, x=2), strategy=ttnn.ShardStrategy.BLOCK),
        dict(core_grid=ttnn.CoreGrid(y=8, x=8), strategy=ttnn.ShardStrategy.BLOCK),
    )
    ttnn.close_device(device)


main()
