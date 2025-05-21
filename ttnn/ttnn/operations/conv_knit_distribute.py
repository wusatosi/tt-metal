import ttnn


@ttnn.register_python_operation(name="ttnn.conv_knit_distribute")
def conv_knit_distribute(
    input_tensor,
    core_grid,
    kernel_height,
    num_output_channels,
    input_width,
    num_input_channels,
    dealloc_inputs_and_move_outputs=False,
    return_original_size=False,
):
    input_shape = input_tensor.shape

    nhw = input_shape[2]
    c = input_shape[3]

    num_cores = core_grid.x * core_grid.y
    grid_coord = ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})

    evenly_divisible_blocks = nhw // num_cores // input_width
    evenly_divisible_extra_rows = nhw // num_cores % input_width
    remainder_blocks = nhw % num_cores // input_width
    remainder_extra_rows = nhw % num_cores % input_width

    total_extra_blocks = (
        remainder_blocks + (evenly_divisible_extra_rows * num_cores + remainder_extra_rows) // input_width
    )

    evenly_divisible_extra_blocks = total_extra_blocks // num_cores
    remainder_extra_blocks = total_extra_blocks % num_cores

    num_blocks_per_core = evenly_divisible_blocks + evenly_divisible_extra_blocks
    num_cores_with_extra_block = remainder_extra_blocks

    print(
        f"Num cores: {num_cores} num_blocks_per_core: {num_blocks_per_core} num_cores_with_extra_block: {num_cores_with_extra_block}"
    )

    distributed_shard_shape = (0, 0)
    if num_cores_with_extra_block == 0:
        distributed_shard_shape = (num_blocks_per_core * input_width, c)
    else:
        distributed_shard_shape = ((num_blocks_per_core + 1) * input_width, c)

    distributed_shard_spec = ttnn.ShardSpec(shard_grid, distributed_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    distributed_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, distributed_shard_spec
    )

    print(f"Distributed shard spec shape: {distributed_shard_spec}")
    print(f"Distributed mem config: {distributed_mem_config}")

    distributed_input = ttnn.conv_distribute(
        input_tensor, distributed_mem_config, input_width, num_blocks_per_core, num_cores_with_extra_block
    )
    if dealloc_inputs_and_move_outputs:
        ttnn.deallocate(input_tensor)
        distributed_input = ttnn.move(distributed_input)

    output_tensor = ttnn.conv_knit(
        distributed_input,
        kernel_height,
        num_output_channels,
        input_width,
        num_input_channels,
        num_blocks_per_core,
        num_cores_with_extra_block,
    )

    if dealloc_inputs_and_move_outputs:
        ttnn.deallocate(distributed_input)
        output_tensor = ttnn.move(output_tensor)

    if return_original_size:
        return [output_tensor, [1, 1, nhw, c]]
    else:
        return output_tensor
