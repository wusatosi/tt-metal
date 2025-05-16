import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_send_receive(mesh_device):
    input_tensor_torch = torch.randn((32, 32, 32, 32))
    return_tensor_torch = torch.randn((32, 32, 32, 32))

    send_submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 1))
    input_tensor = ttnn.from_torch(input_tensor_torch).to(send_submesh)
    return_tensor = ttnn.from_torch(return_tensor_torch).to(send_submesh)

    source_coord = ttnn.MeshCoordinate((0, 0))
    dest_coord = ttnn.MeshCoordinate((0, 1))

    receive_submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 1), offset=dest_coord)

    compute_grid_size = receive_submesh.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )

    receiver_semaphore = ttnn.create_global_semaphore(receive_submesh, cores, 0)

    # !TODO implement me
    sent_tensor = ttnn.point_to_point(input_tensor, dest_coord, ttnn.Topology.Linear, mesh_device, receiver_semaphore)
    return_tensor = ttnn.point_to_point(
        sent_tensor, source_coord, ttnn.Topology.Linear, mesh_device, receiver_semaphore
    )

    assert_with_pcc(ttnn.to_torch(return_tensor), ttnn.to_torch(input_tensor))
