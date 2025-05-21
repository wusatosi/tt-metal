import numpy as np
import pytest
import torch
import ttnn


def fill_tensor_blocks(shape, block_size, start_value=1):
    """
    Fill a 4D tensor of given shape (N, C, H, W) where N and C are 1,
    so that each vertical block of rows (block_size rows per block along H)
    is filled with a unique value (starting from start_value, incrementing by 1 per block).
    """
    assert len(shape) == 4 and shape[0] == 1 and shape[1] == 1, "Shape must be (1, 1, H, W)"
    tensor = torch.zeros(shape, dtype=torch.float32)
    num_blocks = (shape[2] + block_size - 1) // block_size
    for block_idx in range(num_blocks):
        start_row = block_idx * block_size
        end_row = min((block_idx + 1) * block_size, shape[2])
        tensor[0, 0, start_row:end_row, :] = start_value + block_idx
    return tensor


@pytest.mark.parametrize(
    "conv_id, num_cores, input_shape, input_shard_shape, block_size",
    [
        (1, 64, (1, 1, 585, 256), (32, 256), 9),
        (2, 64, (1, 1, 2193, 256), (64, 256), 17),
        (3, 64, (1, 1, 9065, 256), (160, 256), 35),
        (4, 64, (1, 1, 33345, 8), (544, 8), 65),
    ],
)
def test_conv_distribute(device, conv_id, num_cores, input_shape, input_shard_shape, block_size):
    # Generate torch tensor
    nhw = input_shape[2]
    c = input_shape[3]
    torch_input_tensor = fill_tensor_blocks(input_shape, block_size)

    # Setup input on device
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})  # all cores
    input_shard_spec = ttnn.ShardSpec(shard_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, input_shard_spec
    )
    ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device=device)
    ttnn_input_tensor = ttnn.to_memory_config(ttnn_input_tensor, input_mem_config)

    # Calculate output shard spec
    evenly_divisible_blocks = nhw // num_cores // block_size
    evenly_divisible_extra_rows = nhw // num_cores % block_size
    remainder_blocks = nhw % num_cores // block_size
    remainder_extra_rows = nhw % num_cores % block_size

    total_extra_blocks = (
        remainder_blocks + (evenly_divisible_extra_rows * num_cores + remainder_extra_rows) // block_size
    )

    evenly_divisible_extra_blocks = total_extra_blocks // num_cores
    remainder_extra_blocks = total_extra_blocks % num_cores

    num_blocks_per_core = evenly_divisible_blocks + evenly_divisible_extra_blocks
    num_cores_with_extra_block = remainder_extra_blocks

    distributed_shard_shape = (0, 0)
    if num_cores_with_extra_block == 0:
        distributed_shard_shape = (num_blocks_per_core * block_size, c)
    else:
        distributed_shard_shape = ((num_blocks_per_core + 1) * block_size, c)

    distributed_shard_spec = ttnn.ShardSpec(shard_grid, distributed_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    distributed_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, distributed_shard_spec
    )

    # Run conv_distribute operation
    distributed_tensor = ttnn.conv_distribute(
        ttnn_input_tensor, distributed_mem_config, block_size, num_blocks_per_core, num_cores_with_extra_block
    )

    # Get output tensor from device
    distributed_tensor = ttnn.from_device(distributed_tensor)
    distributed_tensor = ttnn.to_torch(distributed_tensor, dtype=torch.float32)

    # print both tensors for comparison in vimdiff
    input_np = torch_input_tensor.flatten(end_dim=2).numpy()
    np.savetxt(f"original_{conv_id}.csv", input_np, delimiter=",")

    distributed_np = distributed_tensor.flatten(end_dim=2).numpy()
    np.savetxt(f"distributed_{conv_id}.csv", distributed_np, delimiter=",")
