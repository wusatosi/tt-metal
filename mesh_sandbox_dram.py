import ttnn
import torch
from loguru import logger
from collections import defaultdict
from models.common.lightweightmodule import LightweightModule

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc

from models.utility_functions import profiler


# reference
class TorchTwoMatmuls:
    def __init__(self, w1, w2):
        self.w1 = w1
        self.w2 = w2

    def forward(self, x):
        mid = x @ self.w1
        out = mid @ self.w2
        return out


# single_chip
class SingleChipTwoMatmuls(LightweightModule):
    def __init__(self, device, w1, w2):
        super().__init__()
        self.device = device
        self.w1 = ttnn.from_torch(
            w1,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        self.w2 = ttnn.from_torch(
            w2,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        mid = x @ self.w1
        out = mid @ self.w2
        ttnn.deallocate(mid)
        return out


# multichip_fracture_dim
class MultichipFractureTwoMatmuls(LightweightModule):
    def __init__(self, device, dim, w1, w2):
        super().__init__()
        self.device = device
        self.dim = dim
        self.w1 = ttnn.from_torch(
            w1,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            layout=ttnn.TILE_LAYOUT,
        )
        self.w2 = ttnn.from_torch(
            w2,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            layout=ttnn.TILE_LAYOUT,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        mid = x @ self.w1
        out = mid @ self.w2
        ttnn.deallocate(mid)
        return out


# multichip DP w submesh
class MultichipSubmeshTwoMatmuls(LightweightModule):
    def __init__(self, mesh_device, submesh_shape, w1, w2):
        super().__init__()
        self.submeshes = mesh_device.create_submeshes(submesh_shape, ttnn.MeshType.Ring)

        self.submesh_to_metadata = defaultdict(dict)
        for submesh in self.submeshes:
            tt_model = SingleChipTwoMatmuls(submesh, w1, w2)
            # wait for all devices in submesh to complete their work
            for i in submesh.get_device_ids():
                device = submesh.get_device(i)
                ttnn.synchronize_device(device)

            # store model for future use
            self.submesh_to_metadata[submesh.get_mesh_id()] = {"tt_model": tt_model}

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        x -> multi-device-tensor to DP
        """
        ins = ttnn.get_device_tensors(x)
        out = []
        for submesh, x in zip(self.submeshes, ins):
            tt_model = self.submesh_to_metadata[submesh.get_mesh_id()]["tt_model"]
            out.append(tt_model.forward(x))

        mesh_out = ttnn.aggregate_as_tensor(out)
        return mesh_out


##################################


# reference
def torch_cpu(profiler, torch_input, torch_w1, torch_w2):
    profiler.start("TorchTwoMatmuls_setup")
    torch_two_matmuls = TorchTwoMatmuls(torch_w1, torch_w2)
    profiler.end("TorchTwoMatmuls_setup")
    profiler.start("TorchTwoMatmuls_forward")
    torch_out = torch_two_matmuls.forward(torch_input)
    profiler.end("TorchTwoMatmuls_forward")
    return torch_out


# single_chip
def test_single_chip(profiler, torch_input, torch_w1, torch_w2, torch_out, n_iters=1):
    logger.info("SingleChipTwoMatmuls")
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()

    profiler.start("SingleChipTwoMatmuls_setup")
    single_chip_two_matmuls = SingleChipTwoMatmuls(device, torch_w1, torch_w2)
    profiler.end("SingleChipTwoMatmuls_setup")

    for i in range(n_iters):
        if i != 0:
            profiler.start("SingleChipTwoMatmuls_forward")
        ttnn_input = ttnn.from_torch(
            torch_input,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn_host_out = ttnn.to_torch(single_chip_two_matmuls.forward(ttnn_input))
        if i != 0:
            profiler.end("SingleChipTwoMatmuls_forward")

    assert_with_pcc(torch_out, ttnn_host_out, 0.98)
    ttnn.close_device(device)


# mesh_device 1x2
def test_mesh_device(profiler, torch_input, torch_w1, torch_w2, torch_out, n_iters=1):
    logger.info("MultichipFractureTwoMatmuls")
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))
    mesh_device.enable_program_cache()
    mesh_device.enable_async(True)

    profiler.start("MultichipFractureTwoMatmuls_setup")
    multichip_fractured_two_matmuls = MultichipFractureTwoMatmuls(mesh_device, 0, torch_w1, torch_w2)
    profiler.end("MultichipFractureTwoMatmuls_setup")

    for i in range(n_iters):
        if i != 0:
            profiler.start("MultichipFractureTwoMatmuls_forward")
        mesh_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            layout=ttnn.TILE_LAYOUT,
        )
        mesh_out = multichip_fractured_two_matmuls.forward(mesh_input)

        with ttnn.distribute(ttnn.ConcatMeshToTensor(mesh_device, dim=0)):
            mesh_host_out = ttnn.to_torch(mesh_out)
        if i != 0:
            profiler.end("MultichipFractureTwoMatmuls_forward")

    assert_with_pcc(torch_out, mesh_host_out, 0.98)
    ttnn.visualize_mesh_device(mesh_device, tensor=mesh_out)
    ttnn.close_mesh_device(mesh_device)


# 2 x sub_mesh 1x1 DP
def test_submesh_data_parallel(profiler, torch_input, torch_w1, torch_w2, torch_out, n_iters=1):
    logger.info("DataParallelTwoMatmuls")
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))
    mesh_device.enable_program_cache()
    mesh_device.enable_async(False)
    # mesh_device.enable_async(True)

    profiler.start("MultichipSubmeshTwoMatmuls_setup")
    data_parallel_two_matmuls = MultichipSubmeshTwoMatmuls(mesh_device, ttnn.MeshShape(1, 1), torch_w1, torch_w2)
    profiler.end("MultichipSubmeshTwoMatmuls_setup")

    for i in range(n_iters):
        if i != 0:
            profiler.start("MultichipSubmeshTwoMatmuls_forward")
        mesh_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            layout=ttnn.TILE_LAYOUT,
        )
        mesh_out = data_parallel_two_matmuls.forward(mesh_input)
        with ttnn.distribute(ttnn.ConcatMeshToTensor(mesh_device, dim=0)):
            mesh_host_out = ttnn.to_torch(mesh_out)
        if i != 0:
            profiler.end("MultichipSubmeshTwoMatmuls_forward")

    assert_with_pcc(torch_out, mesh_host_out, 0.98)
    ttnn.visualize_mesh_device(mesh_device, tensor=mesh_out)
    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    # torch_input = torch.rand(64 * 32, 128 * 32)
    # torch_w1 = torch.rand(128 * 32, 64 * 32)
    # torch_w2 = torch.rand(64 * 32, 32 * 32)

    torch_input = torch.rand(64 * 32, 32 * 128 * 32)
    torch_w1 = torch.rand(32 * 128 * 32, 4 * 64 * 32)
    torch_w2 = torch.rand(4 * 64 * 32, 32 * 32)

    torch_out = torch_cpu(profiler, torch_input, torch_w1, torch_w2)
    test_single_chip(profiler, torch_input, torch_w1, torch_w2, torch_out, 10)
    test_mesh_device(profiler, torch_input, torch_w1, torch_w2, torch_out, 10)
    test_submesh_data_parallel(profiler, torch_input, torch_w1, torch_w2, torch_out, 10)
    profiler.print(units="ms", statistics=True)


# torch.float32

# torch_input = torch.rand(64 * 32, 128 * 32)
# torch_w1 = torch.rand(128 * 32, 64 * 32)
# torch_w2 = torch.rand(64 * 32, 32 * 32)
# TorchTwoMatmuls_setup: (0.0±0.0)ms
# TorchTwoMatmuls_forward: (103.1±0.0)ms
# SingleChipTwoMatmuls_setup: (38.6±0.0)ms
# SingleChipTwoMatmuls_forward: (37.6±0.7)ms
# MultichipFractureTwoMatmuls_setup: (83.5±0.0)ms
# MultichipFractureTwoMatmuls_forward: (53.4±4.0)ms
# MultichipSubmeshTwoMatmuls_setup: (92.7±0.0)ms
# MultichipSubmeshTwoMatmuls_forward: (65.4±2.5)ms

# torch_input = torch.rand(64 * 32, 32*128 * 32)
# torch_w1 = torch.rand(32*128 * 32, 64 * 32)
# torch_w2 = torch.rand(64 * 32, 32 * 32)
# TorchTwoMatmuls_setup: (0.0±0.0)ms
# TorchTwoMatmuls_forward: (1430.4±0.0)ms
# SingleChipTwoMatmuls_setup: (1013.9±0.0)ms
# SingleChipTwoMatmuls_forward: (1230.1±5.8)ms
# MultichipFractureTwoMatmuls_setup: (2479.2±0.0)ms
# MultichipFractureTwoMatmuls_forward: (1427.0±28.4)ms
# MultichipSubmeshTwoMatmuls_setup: (2337.7±0.0)ms
# MultichipSubmeshTwoMatmuls_forward: (1828.2±11.1)ms

# torch_input = torch.rand(64 * 32, 32*128 * 32)
# torch_w1 = torch.rand(32*128 * 32, 4 * 64 * 32)
# torch_w2 = torch.rand(4 * 64 * 32, 32 * 32)
# TorchTwoMatmuls_setup: (0.0±0.0)ms | N=1
# TorchTwoMatmuls_forward: (5346.2±0.0)ms | N=1
# SingleChipTwoMatmuls_setup: (3872.6±0.0)ms | N=1
# SingleChipTwoMatmuls_forward: (1889.3±9.8)ms | N=9
# MultichipFractureTwoMatmuls_setup: (9972.9±0.0)ms | N=1
# MultichipFractureTwoMatmuls_forward: (1819.8±26.3)ms | N=9
# MultichipSubmeshTwoMatmuls_setup: (9087.3±0.0)ms | N=1
# MultichipSubmeshTwoMatmuls_forward: (2276.4±24.5)ms | N=9

# todo @ => sharded 2d matmul
# todo 2xCQs with traces
