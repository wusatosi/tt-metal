import torch
import os
import pytest
from loguru import logger
import ttnn

from models.demos.llama3.tt.llama_ccl import tt_all_gather

from models.utility_functions import (
    comp_pcc,
)

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


def get_physical_to_logical_core_mapping(device):
    """
    Get a mapping from physical core coords to logical core coords

    Returns a dictionary.
    """
    mapping = {}
    is_mesh_device = isinstance(device, ttnn._ttnn.multi_device.MeshDevice)
    if is_mesh_device:
        device = device.get_device(device.get_device_ids()[0])
    grid = device.compute_with_storage_grid_size()
    for x in range(grid.x):
        for y in range(grid.y):
            physical_core = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            mapping[(physical_core.x, physical_core.y)] = (x, y)
    return mapping


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
def test_all_gather_reshard(mesh_device, use_program_cache, reset_seeds, ensure_gc):
    # Init input tensor: 1,1,32,8192 on 8x4 mesh
    # Set to sharded memory config after create heads
    # All gather with output memory config being the SHARDED_WO_RING_MEMCFG

    #
    #
    #

    # FINAL CORE GRID: FAILS
    start_core = ttnn.CoreCoord(1, 0)
    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )

    # # TODO: SIMPLE CORE GRID --> PASSES
    # start_core = ttnn.CoreCoord(0, 0)
    # sub_core_grids = ttnn.CoreRangeSet(
    #     [
    #         ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7,7)),
    #     ]
    # )

    #
    #
    #

    attn_num_cores = 32
    attn_core_range_set = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        start_core, attn_num_cores, sub_core_grids, row_wise=True
    )
    print(attn_core_range_set)

    # MM ring cores
    mm_ring_size = 24

    ring_core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(x, y),
                ttnn.CoreCoord(x, y),
            )
            for x, y in PREFETCHER_NOC1_RING
        ]
    )
    print(ring_core_range_set)

    input_memory_config = ttnn.create_sharded_memory_config(
        shape=(32, 12288 // 8 // attn_num_cores),
        core_grid=attn_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    #
    #
    #

    # Custom output memory config for 24 MM cores: FAILS
    output_memory_config = ttnn.create_sharded_memory_config(
        shape=(32, 12288 // mm_ring_size),
        core_grid=ring_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # # TODO: Use output memory config = SIMPLE CORE GRID from above --> PASSES
    # output_memory_config = ttnn.create_sharded_memory_config(
    #     shape=(32, 12288 // attn_num_cores),
    #     core_grid=attn_core_range_set,
    #     strategy=ttnn.ShardStrategy.WIDTH,
    #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
    #     use_height_and_width_as_shard_shape=True,
    # )

    #
    #
    #

    # [1, 1, 32, 12288]
    input_tensor_torch = torch.randn(1, 1, 32, 12288)
    input_tensor = ttnn.as_tensor(
        input_tensor_torch,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device=mesh_device, dims=(3, None), mesh_shape=list(mesh_device.shape)),
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_memory_config,
    )

    output_ttnn = tt_all_gather(
        input_tensor,
        mesh_device,
        dim=3,
        cluster_axis=0,
        num_links=2,
        memory_config=output_memory_config,
        sharded=True,
    )

    output_torch = ttnn.to_torch(
        output_ttnn,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=(8, 4)),
    )

    is_passing = comp_pcc(input_tensor_torch, output_torch[0:1, 0:1, :, :], 0.9999)

    assert is_passing
