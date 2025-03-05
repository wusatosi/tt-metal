# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_all_gather import (
    run_all_gather_on_n300_impl,
    run_all_gather_sharded_n300,
)
from models.demos.llama3.tt.llama_ccl import tt_all_reduce, tt_all_gather


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, output_shape, dim, layout",
    [
        # (2, 1, [1, 1, 64, 16384], 3, ttnn.TILE_LAYOUT), # mismatching!
        # (2, 1, [8, 5, 32, 768], 3, ttnn.TILE_LAYOUT), # mismatching!
        # (2, 1, [1, 1, 32, 736], 3, ttnn.TILE_LAYOUT), # HANGING
        (2, 1, [1, 1, 32, 704], 3, ttnn.TILE_LAYOUT),  # passing
        # (2, 1, [1, 1, 64, 704], 3, ttnn.TILE_LAYOUT),  # passing
        # # (2, 1, [1, 1, 32, 736], 3, ttnn.ROW_MAJOR_LAYOUT), # alignment issue!!!
        # (2, 1, [1, 1, 32, 704], 3, ttnn.ROW_MAJOR_LAYOUT),  # passing
        # (2, 1, [1, 1, 64, 704], 3, ttnn.ROW_MAJOR_LAYOUT),  # passing
        # (2, 1, [4, 1, 256, 32], 0, ttnn.ROW_MAJOR_LAYOUT),  # passing
        # (2, 1, [8, 1, 256, 32], 0, ttnn.ROW_MAJOR_LAYOUT),  # passing
        # # (2, 1, [1, 1, 32, 8192], 3, ttnn.ROW_MAJOR_LAYOUT), # PCC MISMATCH
        # # (2, 1, [8, 5, 13, 512], 3, ttnn.ROW_MAJOR_LAYOUT), # PCC MISMATCH
        # # (2, 1, [8, 5, 13, 768], 3, ttnn.ROW_MAJOR_LAYOUT), # PCC MISMATCH!
        # # (2, 1, [8, 8, 256, 384], 1, ttnn.ROW_MAJOR_LAYOUT), # PCC MISMATCH!
        # (2, 1, [1, 1, 64, 2048], 3, ttnn.TILE_LAYOUT),  # passing
        # (2, 1, [1, 1, 32, 4096], 3, ttnn.TILE_LAYOUT),  # passing
        # (2, 1, [1, 1, 32, 1024], 3, ttnn.ROW_MAJOR_LAYOUT),  # passing
        # (2, 1, [1, 2, 32, 4096], 3, ttnn.ROW_MAJOR_LAYOUT),  # passing
        # (2, 1, [1, 2, 32, 1024], 3, ttnn.TILE_LAYOUT),  # passing
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("num_iters", [1])
@pytest.mark.parametrize("enable_async", [True, False])
def test_all_gather_on_dual_p150_post_commit(
    dual_p150_mesh_device,
    num_devices,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_on_n300_impl(
        dual_p150_mesh_device,
        num_devices,
        output_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Ring,
        num_iters=num_iters,
        enable_async=enable_async,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", [2])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "tensor_mem_layout",
    [
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ],
)
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "input_shape, input_shard_shape,shard_grid",
    (
        (
            (1, 1, 128, 8192),
            (128, 256),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (
            (1, 1, 32, 1792),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}),
        ),
    ),
)
@pytest.mark.parametrize("enable_async", [True])
def test_all_gather_sharded_dual_p150_post_commit(
    dual_p150_mesh_device,
    num_devices,
    input_shape,
    input_shard_shape,
    shard_grid,
    dim,
    num_links,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    # num_cores,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_sharded_n300(
        dual_p150_mesh_device,
        num_devices,
        input_shape,
        input_shard_shape,
        shard_grid,
        dim,
        num_links,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        # num_cores,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Ring,
        enable_async=enable_async,
    )


@pytest.mark.parametrize(
    "input_shape, input_layout, cluster_axis, dim, num_links, memory_config, sharded, topology, dtype",
    (
        (
            (1, 1, 128, 2048),
            ttnn.TILE_LAYOUT,
            None,
            3,
            1,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.DRAM),
            False,
            ttnn.Topology.Linear,
            ttnn.bfloat16,
        ),
        (
            (1, 1, 32, 2048),
            ttnn.TILE_LAYOUT,
            None,
            3,
            1,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.DRAM),
            False,
            ttnn.Topology.Linear,
            ttnn.bfloat16,
        ),
        (
            (1, 1, 32, 64128),
            ttnn.TILE_LAYOUT,
            None,
            3,
            1,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.L1),
            False,
            ttnn.Topology.Linear,
            ttnn.bfloat8_b,
        ),
        (
            (1, 1, 32, 64128),
            ttnn.TILE_LAYOUT,
            None,
            3,
            1,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.L1),
            False,
            ttnn.Topology.Linear,
            ttnn.bfloat8_b,
        ),
    ),
)
@skip_for_grayskull("Requires eth connected devices to run")
def test_interleaved_all_gather_llama8b_dual_p150(
    dual_p150_mesh_device,
    input_shape,
    input_layout,
    cluster_axis,
    dim,
    num_links,
    memory_config,
    sharded,
    topology,
    dtype,
):
    input_tensor = torch.rand(input_shape).bfloat16()
    ttnn_input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=dtype,
        layout=input_layout,
        tile=ttnn.Tile((32, 32)),
        mesh_mapper=ttnn.ShardTensorToMesh(dual_p150_mesh_device, dim),
        device=dual_p150_mesh_device,
    )

    output_tensor = tt_all_gather(
        ttnn_input_tensor, dual_p150_mesh_device, cluster_axis, dim, num_links, memory_config, sharded, topology, dtype
    )


# @pytest.mark.parametrize(
#     "input_shape, input_layout, cluster_axis, dim, num_links, memory_config, sharded, topology, dtype",
#     (
#         (
#             (1, 1, 128, 2048),
#             ttnn.TILE_LAYOUT,
#             None,
#             3,
#             1,
#             ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.DRAM),
#             False,
#             ttnn.Topology.Linear,
#             ttnn.bfloat16
#         ),
#         (
#             (1, 1, 32, 2048),
#             ttnn.TILE_LAYOUT,
#             None,
#             3,
#             1,
#             ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.DRAM),
#             False,
#             ttnn.Topology.Linear,
#             ttnn.bfloat16
#         ),
#         (
#             (1, 1, 32, 64128),
#             ttnn.TILE_LAYOUT,
#             None,
#             3,
#             1,
#             ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.L1),
#             False,
#             ttnn.Topology.Linear,
#             ttnn.bfloat8_b
#         ),
#         (
#             (1, 1, 32, 64128),
#             ttnn.TILE_LAYOUT,
#             None,
#             3,
#             1,
#             ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.L1),
#             False,
#             ttnn.Topology.Linear,
#             ttnn.bfloat8_b
#         ),
#     ),
# )
@skip_for_grayskull("Requires eth connected devices to run")
def test_sharded_all_gather_llama8b_dual_p150(
    dual_p150_mesh_device,
    # input_shape,
    # input_layout,
    # cluster_axis,
    # dim,
    # num_links,
    # memory_config,
    # sharded,
    # topology,
    # dtype
):
    input_shape = (1, 1, 32, 2048)
    dtype = ttnn.bfloat16
    memory_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    buffer_type = ttnn.BufferType.L1
    # shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))})
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    # input_shard_shape = (32, 128)
    input_shard_shape = (32, 64)
    orientation = ttnn.ShardOrientation.ROW_MAJOR

    dim = 3
    num_links = 1
    cluster_axis = None
    sharded = True
    topology = ttnn.Topology.Linear
    tensor_layout = ttnn.TILE_LAYOUT

    num_devices = 2
    numel = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * num_devices
    unchunked_input_shape = list(input_shape)
    unchunked_input_shape[dim] *= num_devices

    unchunked_input_tensor = torch.rand(unchunked_input_shape).bfloat16()
    unchunked_input_tensor = unchunked_input_tensor.bfloat16()

    input_tensors = torch.chunk(unchunked_input_tensor, num_devices, dim)

    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        input_shard_shape,
        orientation,
    )
    input_mem_config = ttnn.MemoryConfig(memory_layout, buffer_type=buffer_type, shard_spec=input_shard_spec)

    output_shard_shape = list(input_shard_shape)
    if dim == len(input_shape) - 1:
        output_shard_shape[1] *= num_devices
    else:
        output_shard_shape[0] *= num_devices
    output_shard_spec = ttnn.ShardSpec(
        shard_grid,
        output_shard_shape,
        orientation,
    )
    output_mem_config = ttnn.MemoryConfig(memory_layout, buffer_type=buffer_type, shard_spec=output_shard_spec)

    if unchunked_input_shape[dim] % num_devices != 0 or (
        dim == 3 and unchunked_input_shape[dim] // num_devices % 32 != 0
    ):
        pytest.skip("Unsupported test case")

    tile = (32, 32)
    tt_input_tensors_dups = []
    tt_input_tensors = []
    for i, t in enumerate(input_tensors):
        tt_input_tensors_dups.append(
            ttnn.Tensor(t, dtype, {}, ttnn.Tile(tile))
            .to(tensor_layout)
            .to(dual_p150_mesh_device.get_devices()[i], input_mem_config)
        )
        tt_input_tensors.append(
            ttnn.Tensor(t, dtype, {}, ttnn.Tile(tile))
            .to(tensor_layout)
            .to(dual_p150_mesh_device.get_devices()[i], input_mem_config)
        )

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)

    output_tensor = tt_all_gather(
        input_tensor_mesh, dual_p150_mesh_device, cluster_axis, dim, num_links, output_mem_config, sharded, topology
    )
