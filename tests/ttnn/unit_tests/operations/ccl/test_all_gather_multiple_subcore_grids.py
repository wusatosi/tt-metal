import torch
import os
import pytest
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# logical coords
PREFETCHER_NOC1_RING = [
    (6, 6),
    (6, 7),
    (6, 9),
    (6, 0),
    (6, 1),
    (6, 2),
    (6, 4),
    (6, 5),
    (5, 5),
    (5, 6),
    (5, 7),
    (5, 9),
    (5, 0),
    (5, 1),
    (5, 2),
    (5, 4),
    (1, 4),
    (1, 5),
    (1, 9),
    (1, 0),
    (2, 0),
    (2, 4),
    (2, 5),
    (2, 9),
]


def get_simple_core_grid(num_cores):
    start_core = ttnn.CoreCoord(0, 0)
    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7)),
        ]
    )
    selected_core_range_set = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        start_core, num_cores, sub_core_grids, row_wise=True
    )
    return selected_core_range_set


def get_rectangular_llama_core_grid(num_cores):
    core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    start_core = ttnn.CoreCoord(1, 0)
    selected_core_range_set = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        start_core, num_cores, core_range_set, row_wise=True
    )
    return selected_core_range_set


def get_matmul_ring_core_range_set():
    ring_core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(x, y),
                ttnn.CoreCoord(x, y),
            )
            for x, y in PREFETCHER_NOC1_RING
        ]
    )
    return ring_core_range_set


@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim",
    [
        (2, 1, [1, 1, 32, 12288], 3),
    ],
)
@pytest.mark.parametrize(
    "input_core_range_set",
    [
        get_rectangular_llama_core_grid(32),
        # get_simple_core_grid(32),
    ],
)
@pytest.mark.parametrize(
    "output_core_range_set",
    [
        get_matmul_ring_core_range_set(),
        # get_simple_core_grid(32),
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
def test_all_gather_reshard(
    num_devices,
    num_links,
    input_shape,
    dim,
    input_core_range_set,
    output_core_range_set,
    mesh_device,
    use_program_cache,
):
    # print("Input core range set:")
    # print(input_core_range_set)

    # print("Output core range set:")
    # print(output_core_range_set)

    # Input memory config
    input_memory_config = ttnn.create_sharded_memory_config(
        shape=(
            input_shape[0] * input_shape[1] * input_shape[2],
            input_shape[3] // num_devices // input_core_range_set.num_cores(),
        ),
        core_grid=input_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Output memory config
    output_memory_config = ttnn.create_sharded_memory_config(
        shape=(input_shape[0] * input_shape[1] * input_shape[2], input_shape[3] // output_core_range_set.num_cores()),
        core_grid=output_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Create input tensor with input memory config
    input_tensor_torch = torch.randn(input_shape)
    input_tensor = ttnn.as_tensor(
        input_tensor_torch,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device=mesh_device, dims=(3, None), mesh_shape=list(mesh_device.shape)),
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_memory_config,
    )

    # All gather with conversion to output memory config
    output_ttnn = ttnn.all_gather(
        input_tensor,
        dim,
        num_links=num_links,
        cluster_axis=0,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
        memory_config=output_memory_config,
    )
    input_tensor.deallocate(True)

    output_torch = ttnn.to_torch(
        output_ttnn,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=list(mesh_device.shape)),
    )

    is_passing = comp_pcc(input_tensor_torch, output_torch[0:1, 0:1, :, :], 0.9999)

    assert is_passing
