import ttnn
import torch
from loguru import logger
from collections import defaultdict
from models.common.lightweightmodule import LightweightModule

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc

from models.utility_functions import profiler
from functools import reduce
from operator import mul


def GetBsMemoryConfig(
    x_shape: list, grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
):
    w_cores = grid.bounding_box().end.x - grid.bounding_box().start.x + 1
    h_cores = grid.bounding_box().end.y - grid.bounding_box().start.y + 1
    shard_shape = [reduce(mul, x_shape[:-1]) // h_cores, x_shape[-1] // w_cores]
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            grid=grid, shard_shape=shard_shape, shard_orientation=ttnn.ShardOrientation.ROW_MAJOR, halo=False
        ),
    )


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
    def __init__(self, device, x_shape, w1, w2):
        super().__init__()
        self.device = device
        self.w1 = ttnn.from_torch(
            w1,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=torch_to_ttnn_dtype(w1.dtype),
            # memory_config=GetBsMemoryConfig(w1)
        )
        self.w2 = ttnn.from_torch(
            w2,
            device=device,
            dtype=torch_to_ttnn_dtype(w2.dtype),
            layout=ttnn.TILE_LAYOUT,
        )

        mid_shape = list(x_shape[0:-1]) + [w1.shape[-1]]

        M = reduce(mul, list(x_shape)) // x_shape[-1] // 32  # W
        N = w1.shape[-1] // 32
        self.matmul1_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=(M - 1) // (8 - 1),
            per_core_N=(N - 1) // (8 - 1),
            transpose_mcast=False,
            fused_activation=None,
        )
        self.matmul1_output_memory_config = GetBsMemoryConfig(mid_shape)

        M = reduce(mul, list(mid_shape)) // mid_shape[-1] // 32  # W
        N = w2.shape[-1] // 32
        self.matmul2_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=(M - 1) // (8 - 1),
            per_core_N=(N - 1) // (8 - 1),
            transpose_mcast=False,
            fused_activation=None,
        )
        self.matmul2_output_memory_config = ttnn.DRAM_MEMORY_CONFIG

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        mid = ttnn.matmul(
            input_tensor_a=x,
            input_tensor_b=self.w1,
            memory_config=self.matmul1_output_memory_config,
            program_config=self.matmul1_program_config,
        )
        out = ttnn.matmul(
            input_tensor_a=mid,
            input_tensor_b=self.w2,
            memory_config=self.matmul2_output_memory_config,
            program_config=self.matmul2_program_config,
        )
        ttnn.deallocate(mid)
        return out


# multichip_fracture_dim
class MultichipFractureTwoMatmuls(LightweightModule):
    def __init__(self, device, x_shape, w1, w2):
        super().__init__()
        self.device = device

        self.w1 = ttnn.from_torch(
            w1,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=torch_to_ttnn_dtype(w1.dtype),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        self.w2 = ttnn.from_torch(
            w2,
            device=device,
            dtype=torch_to_ttnn_dtype(w2.dtype),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            layout=ttnn.TILE_LAYOUT,
        )

        x_shape_half = [x_shape[0] // 2, x_shape[1]]

        mid_shape = list(x_shape_half[0:-1]) + [w1.shape[-1]]

        M = reduce(mul, list(x_shape_half)) // x_shape_half[-1] // 32  # W
        N = w1.shape[-1] // 32
        self.matmul1_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=(M - 1) // (8 - 1),
            per_core_N=(N - 1) // (8 - 1),
            transpose_mcast=False,
            fused_activation=None,
        )
        self.matmul1_output_memory_config = GetBsMemoryConfig(mid_shape)

        M = reduce(mul, list(mid_shape)) // mid_shape[-1] // 32  # W
        N = w2.shape[-1] // 32
        self.matmul2_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=(M - 1) // (8 - 1),
            per_core_N=(N - 1) // (8 - 1),
            transpose_mcast=False,
            fused_activation=None,
        )
        self.matmul2_output_memory_config = ttnn.DRAM_MEMORY_CONFIG

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        mid = ttnn.matmul(
            input_tensor_a=x,
            input_tensor_b=self.w1,
            memory_config=self.matmul1_output_memory_config,
            program_config=self.matmul1_program_config,
        )
        out = ttnn.matmul(
            input_tensor_a=mid,
            input_tensor_b=self.w2,
            memory_config=self.matmul2_output_memory_config,
            program_config=self.matmul2_program_config,
        )
        ttnn.deallocate(mid)
        return out


# multichip DP w submesh
class MultichipSubmeshTwoMatmuls(LightweightModule):
    def __init__(self, mesh_device, submesh_shape, x_shape, w1, w2):
        super().__init__()
        self.submeshes = mesh_device.create_submeshes(submesh_shape, ttnn.MeshType.Ring)

        x_shape_half = [x_shape[0] // 2, x_shape[1]]

        self.submesh_to_metadata = defaultdict(dict)
        for submesh in self.submeshes:
            tt_model = SingleChipTwoMatmuls(submesh, x_shape_half, w1, w2)
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


def torch_to_ttnn_dtype(torch_dtype):
    if torch_dtype == torch.float16:
        return ttnn.bfloat16
    elif torch_dtype == torch.float32:
        return ttnn.float32
    else:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


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
    single_chip_two_matmuls = SingleChipTwoMatmuls(device, torch_input.shape, torch_w1, torch_w2)
    profiler.end("SingleChipTwoMatmuls_setup")

    for i in range(n_iters):
        if i != 0:
            profiler.start("SingleChipTwoMatmuls_forward")
        ttnn_input = ttnn.from_torch(
            torch_input,
            device=device,
            dtype=torch_to_ttnn_dtype(torch_input.dtype),
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
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 2),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH, ttnn.DispatchCoreAxis.ROW),
    )
    mesh_device.enable_program_cache()
    mesh_device.enable_async(True)

    profiler.start("MultichipFractureTwoMatmuls_setup")
    multichip_fractured_two_matmuls = MultichipFractureTwoMatmuls(mesh_device, torch_input.shape, torch_w1, torch_w2)
    profiler.end("MultichipFractureTwoMatmuls_setup")

    for i in range(n_iters):
        if i != 0:
            profiler.start("MultichipFractureTwoMatmuls_forward")
        mesh_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            dtype=torch_to_ttnn_dtype(torch_input.dtype),
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
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 2),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH, ttnn.DispatchCoreAxis.ROW),
    )
    mesh_device.enable_program_cache()
    mesh_device.enable_async(False)
    # mesh_device.enable_async(True)

    profiler.start("MultichipSubmeshTwoMatmuls_setup")
    data_parallel_two_matmuls = MultichipSubmeshTwoMatmuls(
        mesh_device, ttnn.MeshShape(1, 1), torch_input.shape, torch_w1, torch_w2
    )
    profiler.end("MultichipSubmeshTwoMatmuls_setup")

    for i in range(n_iters):
        if i != 0:
            profiler.start("MultichipSubmeshTwoMatmuls_forward")
        mesh_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            dtype=torch_to_ttnn_dtype(torch_input.dtype),
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
    torch_input = torch.rand(64 * 32, 128 * 32)
    torch_w1 = torch.rand(128 * 32, 64 * 32)
    torch_w2 = torch.rand(64 * 32, 32 * 32)

    # torch_input = torch.rand(64 * 32, 128 * 32, dtype=torch.float16)
    # torch_w1 = torch.rand(128 * 32, 64 * 32, dtype=torch.float16)
    # torch_w2 = torch.rand(64 * 32, 32 * 32, dtype=torch.float16)

    torch_out = torch_cpu(profiler, torch_input, torch_w1, torch_w2)
    test_single_chip(profiler, torch_input, torch_w1, torch_w2, torch_out, 10)
    test_mesh_device(profiler, torch_input, torch_w1, torch_w2, torch_out, 10)
    test_submesh_data_parallel(profiler, torch_input, torch_w1, torch_w2, torch_out, 10)
    profiler.print(units="ms", statistics=True)


# torch_input = torch.rand(64 * 32, 128 * 32)
# torch_w1 = torch.rand(128 * 32, 64 * 32)
# torch_w2 = torch.rand(64 * 32, 32 * 32
# TorchTwoMatmuls_setup: (0.0±0.0)ms | N=1
# TorchTwoMatmuls_forward: (63.1±0.0)ms | N=1
# SingleChipTwoMatmuls_setup: (61.3±0.0)ms | N=1
# SingleChipTwoMatmuls_forward: (55.4±1.4)ms | N=9
# MultichipFractureTwoMatmuls_setup: (111.5±0.0)ms | N=1
# MultichipFractureTwoMatmuls_forward: (46.8±2.6)ms | N=9
# MultichipSubmeshTwoMatmuls_setup: (120.8±0.0)ms | N=1
# MultichipSubmeshTwoMatmuls_forward: (66.2±4.8)ms | N=9

# torch_input = torch.rand(64 * 32, 128 * 32, dtype=torch.float16)
# torch_w1 = torch.rand(128 * 32, 64 * 32, dtype=torch.float16)
# torch_w2 = torch.rand(64 * 32, 32 * 32, dtype=torch.float16)
# TorchTwoMatmuls_setup: (0.0±0.0)ms | N=1
# TorchTwoMatmuls_forward: (91461.3±0.0)ms | N=1
# SingleChipTwoMatmuls_setup: (35.5±0.0)ms | N=1
# SingleChipTwoMatmuls_forward: (25.3±2.9)ms | N=9
# MultichipFractureTwoMatmuls_setup: (66.9±0.0)ms | N=1
# MultichipFractureTwoMatmuls_forward: (21.3±0.3)ms | N=9
# MultichipSubmeshTwoMatmuls_setup: (56.3±0.0)ms | N=1
# MultichipSubmeshTwoMatmuls_forward: (26.0±0.5)ms | N=9
