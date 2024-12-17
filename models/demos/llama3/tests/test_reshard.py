import ttnn
import torch
import pytest

mm_core_grid = [
    ttnn.CoreCoord(6, 9),
    ttnn.CoreCoord(6, 7),
    ttnn.CoreCoord(6, 6),
    ttnn.CoreCoord(6, 5),
    ttnn.CoreCoord(6, 4),
    ttnn.CoreCoord(6, 2),
    ttnn.CoreCoord(6, 1),
    ttnn.CoreCoord(6, 0),
    ttnn.CoreCoord(5, 0),
    ttnn.CoreCoord(5, 1),
    ttnn.CoreCoord(5, 2),
    ttnn.CoreCoord(5, 4),
    ttnn.CoreCoord(5, 5),
    ttnn.CoreCoord(5, 6),
    ttnn.CoreCoord(5, 7),
    ttnn.CoreCoord(5, 9),
    ttnn.CoreCoord(2, 9),
    ttnn.CoreCoord(2, 5),
    ttnn.CoreCoord(2, 4),
    ttnn.CoreCoord(2, 0),
    ttnn.CoreCoord(1, 0),
    ttnn.CoreCoord(1, 4),
    ttnn.CoreCoord(1, 5),
    ttnn.CoreCoord(1, 9),
]

mm_range = ttnn.CoreRangeSet([ttnn.CoreRange(mm_core_grid[i], mm_core_grid[i]) for i in range(24)])
sub_core_grids = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
    ]
)
input_corerangeset = ttnn.num_cores_to_corerangeset_in_subcoregrids(
    ttnn.CoreCoord(1, 0), 8, sub_core_grids, row_wise=True
)
input_memory_config = ttnn.create_sharded_memory_config(
    shape=(32, 288),
    core_grid=input_corerangeset,
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

print(input_memory_config)

output_memory_config = ttnn.create_sharded_memory_config(
    shape=(32, 96),
    core_grid=mm_range,
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

print(output_memory_config)


@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=False)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (1, 1),
    ],
    indirect=True,
)
def test_reshard(mesh_device, use_program_cache):
    mesh_device.enable_async(True)
    input_tensor = ttnn.from_torch(
        torch.rand(1, 1, 32, 2304),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    print(input_tensor)

    output_tensor = ttnn.to_memory_config(input_tensor, output_memory_config)
    print(output_tensor.memory_config())
