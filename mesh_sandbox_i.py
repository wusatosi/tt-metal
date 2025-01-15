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
            w1, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        self.w2 = ttnn.from_torch(
            w2, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
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
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self.w2 = ttnn.from_torch(
            w2,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
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


#################
# playground


# reference
class TorchTwoMatmuls:
    def __init__(self, w1, w2):
        self.w1 = w1
        self.w2 = w2

    def forward(self, input):
        mid = input @ self.w1
        out = mid @ self.w2
        return out


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
    device = ttnn.open_device(
        device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH, ttnn.DispatchCoreAxis.ROW)
    )
    device.enable_program_cache()

    profiler.start("SingleChipTwoMatmuls_setup")
    single_chip_two_matmuls = SingleChipTwoMatmuls(device, torch_w1, torch_w2)
    profiler.end("SingleChipTwoMatmuls_setup")

    for i in range(n_iters):
        if i == 0:
            profiler.start("SingleChipTwoMatmuls_forward0")
        if i != 0:
            profiler.start("SingleChipTwoMatmuls_forward")
        ttnn_input = ttnn.from_torch(
            torch_input,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn_host_out = ttnn.to_torch(single_chip_two_matmuls.forward(ttnn_input))
        if i == 0:
            profiler.end("SingleChipTwoMatmuls_forward0")
        if i != 0:
            profiler.end("SingleChipTwoMatmuls_forward")

    assert_with_pcc(torch_out, ttnn_host_out, 0.98)
    ttnn.close_device(device)


# mesh_device 1x2
def test_mesh_device(profiler, torch_input, torch_w1, torch_w2, torch_out, n_iters=1):
    logger.info("MultichipFractureTwoMatmuls")
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 2),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH, ttnn.DispatchCoreAxis.ROW),
    )
    mesh_device.enable_program_cache()
    mesh_device.enable_async(True)

    profiler.start("MultichipFractureTwoMatmuls_setup")
    multichip_fractured_two_matmuls = MultichipFractureTwoMatmuls(mesh_device, 0, torch_w1, torch_w2)
    profiler.end("MultichipFractureTwoMatmuls_setup")

    for i in range(n_iters):
        if i == 0:
            profiler.start("MultichipFractureTwoMatmuls_forward0")
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
        if i == 0:
            profiler.end("MultichipFractureTwoMatmuls_forward0")
        if i != 0:
            profiler.end("MultichipFractureTwoMatmuls_forward")

    assert_with_pcc(torch_out, mesh_host_out, 0.98)
    ttnn.visualize_mesh_device(mesh_device, tensor=mesh_out)
    ttnn.close_mesh_device(mesh_device)


# 2 x sub_mesh 1x1 DP
def test_submesh_data_parallel(profiler, torch_input, torch_w1, torch_w2, torch_out, n_iters=1):
    logger.info("DataParallelTwoMatmuls")
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 2),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH, ttnn.DispatchCoreAxis.ROW),
    )
    mesh_device.enable_program_cache()
    mesh_device.enable_async(False)
    # mesh_device.enable_async(True)

    profiler.start("MultichipSubmeshTwoMatmuls_setup")
    data_parallel_two_matmuls = MultichipSubmeshTwoMatmuls(mesh_device, ttnn.MeshShape(1, 1), torch_w1, torch_w2)
    profiler.end("MultichipSubmeshTwoMatmuls_setup")

    for i in range(n_iters):
        if i == 0:
            profiler.start("MultichipSubmeshTwoMatmuls_forward0")
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
        if i == 0:
            profiler.end("MultichipSubmeshTwoMatmuls_forward0")
        if i != 0:
            profiler.end("MultichipSubmeshTwoMatmuls_forward")

    assert_with_pcc(torch_out, mesh_host_out, 0.98)
    ttnn.visualize_mesh_device(mesh_device, tensor=mesh_out)
    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    # torch_input = torch.rand(64 * 32, 128 * 32, dtype=torch.float16)
    # torch_w1 = torch.rand(128 * 32, 64 * 32, dtype=torch.float16)
    # torch_w2 = torch.rand(64 * 32, 32 * 32, dtype=torch.float16)

    torch_input = torch.rand(64 * 32, 128 * 32, dtype=torch.float16)
    torch_w1 = torch.rand(128 * 32, 4 * 64 * 32, dtype=torch.float16)
    torch_w2 = torch.rand(4 * 64 * 32, 32 * 32, dtype=torch.float16)

    torch_out = torch_cpu(profiler, torch_input, torch_w1, torch_w2)
    test_single_chip(profiler, torch_input, torch_w1, torch_w2, torch_out, 10)
    test_mesh_device(profiler, torch_input, torch_w1, torch_w2, torch_out, 10)
    test_submesh_data_parallel(profiler, torch_input, torch_w1, torch_w2, torch_out, 10)
    profiler.print(units="ms", statistics=True)


# torch_input = torch.rand(64 * 32, 128 * 32)
# torch_w1 = torch.rand(128 * 32, 64 * 32)
# torch_w2 = torch.rand(64 * 32, 32 * 32)
# TorchTwoMatmuls_setup: (0.0±0.0)ms | N=1
# TorchTwoMatmuls_forward: (50.7±0.0)ms | N=1
# SingleChipTwoMatmuls_setup: (31.8±0.0)ms | N=1
# SingleChipTwoMatmuls_forward0: (595.0±0.0)ms | N=1
# SingleChipTwoMatmuls_forward: (36.7±0.5)ms | N=9
# MultichipFractureTwoMatmuls_setup: (48.6±0.0)ms | N=1
# MultichipFractureTwoMatmuls_forward0: (493.3±0.0)ms | N=1
# MultichipFractureTwoMatmuls_forward: (30.5±2.6)ms | N=9
# MultichipSubmeshTwoMatmuls_setup: (48.8±0.0)ms | N=1
# MultichipSubmeshTwoMatmuls_forward0: (41.4±0.0)ms | N=1
# MultichipSubmeshTwoMatmuls_forward: (45.5±1.5)ms | N=9


# torch_input = torch.rand(64 * 32, 128 * 32, dtype=torch.float16)
# torch_w1 = torch.rand(128 * 32, 64 * 32, dtype=torch.float16)
# torch_w2 = torch.rand(64 * 32, 32 * 32, dtype=torch.float16)
# TorchTwoMatmuls_setup: (0.0±0.0)ms | N=1
# TorchTwoMatmuls_forward: (89363.7±0.0)ms | N=1
# SingleChipTwoMatmuls_setup: (30.8±0.0)ms | N=1
# SingleChipTwoMatmuls_forward0: (552.5±0.0)ms | N=1
# SingleChipTwoMatmuls_forward: (23.9±2.2)ms | N=9
# MultichipFractureTwoMatmuls_setup: (45.0±0.0)ms | N=1
# MultichipFractureTwoMatmuls_forward0: (501.6±0.0)ms | N=1
# MultichipFractureTwoMatmuls_forward: (21.0±0.5)ms | N=9
# MultichipSubmeshTwoMatmuls_setup: (48.4±0.0)ms | N=1
# MultichipSubmeshTwoMatmuls_forward0: (27.6±0.0)ms | N=1
# MultichipSubmeshTwoMatmuls_forward: (25.8±0.2)ms | N=9

# torch_input = torch.rand(64 * 32, 32*128 * 32)
# torch_w1 = torch.rand(32*128 * 32, 64 * 32)
# torch_w2 = torch.rand(64 * 32, 32 * 32)
#  Always | FATAL    | Out of Memory: Not enough space to allocate 536870912 B L1 buffer across 64 banks, where each bank needs to store 8388608 B


# torch_input = torch.rand(64 * 32, 128 * 32, dtype=torch.float16)
# torch_w1 = torch.rand(128 * 32, 2 * 64 * 32, dtype=torch.float16)
# torch_w2 = torch.rand(2 * 64 * 32, 32 * 32, dtype=torch.float16)
# TorchTwoMatmuls_setup: (0.0±0.0)ms | N=1
# TorchTwoMatmuls_forward: (187344.7±0.0)ms | N=1
# SingleChipTwoMatmuls_setup: (80.5±0.0)ms | N=1
# SingleChipTwoMatmuls_forward0: (494.2±0.0)ms | N=1
# SingleChipTwoMatmuls_forward: (45.4±2.4)ms | N=9
# MultichipFractureTwoMatmuls_setup: (101.7±0.0)ms | N=1
# MultichipFractureTwoMatmuls_forward0: (468.4±0.0)ms | N=1
# MultichipFractureTwoMatmuls_forward: (32.7±1.3)ms | N=9
# MultichipSubmeshTwoMatmuls_setup: (124.1±0.0)ms | N=1
# MultichipSubmeshTwoMatmuls_forward0: (39.3±0.0)ms | N=1
# MultichipSubmeshTwoMatmuls_forward: (38.1±0.5)ms | N=9


# torch_input = torch.rand(64 * 32, 128 * 32, dtype=torch.float16)
# torch_w1 = torch.rand(128 * 32, 4 * 64 * 32, dtype=torch.float16)
# torch_w2 = torch.rand(4 * 64 * 32, 32 * 32, dtype=torch.float16)
# TorchTwoMatmuls_setup: (0.0±0.0)ms | N=1
# TorchTwoMatmuls_forward: (373970.7±0.0)ms | N=1
# SingleChipTwoMatmuls_setup: (120.1±0.0)ms | N=1
# SingleChipTwoMatmuls_forward0: (458.8±0.0)ms | N=1
# SingleChipTwoMatmuls_forward: (78.3±3.0)ms | N=9
# MultichipFractureTwoMatmuls_setup: (213.8±0.0)ms | N=1
# MultichipFractureTwoMatmuls_forward0: (455.0±0.0)ms | N=1
# MultichipFractureTwoMatmuls_forward: (48.5±2.5)ms | N=9
# MultichipSubmeshTwoMatmuls_setup: (242.0±0.0)ms | N=1
# MultichipSubmeshTwoMatmuls_forward0: (53.5±0.0)ms | N=1
# MultichipSubmeshTwoMatmuls_forward: (53.2±0.6)ms | N=9
