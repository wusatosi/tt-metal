import torch
import ttnn
import pytest
from loguru import logger
from models.demos.llama3_subdevices.tt.llama_ccl import TT_CCL


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "use_small_core_grid",
    [True, False],
)
def test_hang_sharded_parallel(mesh_device, use_small_core_grid):
    w1_torch = torch.randn(8192, 28672)
    w2_torch = torch.randn(28672, 8192)

    logger.info("Pushing w1 to devices")
    w1_ttnn = ttnn.as_tensor(
        w1_torch,
        dtype=ttnn.bfloat4_b,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, -2), mesh_shape=(8, 4)),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info("Pushing w2 to devices")
    w2_ttnn = ttnn.as_tensor(
        w2_torch,
        dtype=ttnn.bfloat8_b,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=(8, 4)),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    w1_input_torch = torch.randn(128, 2048)
    logger.info("Pushing w1 input to devices")
    w1_input_ttnn = ttnn.from_torch(
        w1_input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    w2_input_torch = torch.randn(128, 3584)
    logger.info("Pushing w2 input to devices")
    w2_input_ttnn = ttnn.from_torch(
        w2_input_torch,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    core_grid = ttnn.CoreGrid(y=6, x=6) if use_small_core_grid else ttnn.CoreGrid(y=7, x=7)
    for _ in range(1000):
        w1_out = ttnn.linear(
            w1_input_ttnn,
            w1_ttnn,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
                dst_full_sync_en=True,
            ),
            dtype=ttnn.bfloat8_b,
            core_grid=core_grid,
        )

        w2_out = ttnn.linear(
            w2_input_ttnn,
            w2_ttnn,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
                dst_full_sync_en=True,
            ),
            dtype=ttnn.bfloat8_b,
            core_grid=core_grid,
        )

        w1_out = ttnn.linear(
            w1_input_ttnn,
            w1_ttnn,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
                dst_full_sync_en=True,
            ),
            dtype=ttnn.bfloat8_b,
            core_grid=core_grid,
        )


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "use_small_core_grid",
    [True, False],
)
def test_hang_per_device(mesh_device, use_small_core_grid):
    w1_torch = torch.randn(2048, 3584)
    w2_torch = torch.randn(3584, 2048)

    w1_input_torch = torch.randn(128, 2048)
    w2_input_torch = torch.randn(128, 3584)

    devices = mesh_device.get_devices()
    # breakpoint()
    logger.info("Pushing w1 to devices")
    w1_ttnn_tensors = [
        ttnn.as_tensor(
            w1_torch,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=devices[i],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for i in range(len(devices))
    ]
    logger.info("Pushing w2 to devices")
    w2_ttnn_tensors = [
        ttnn.as_tensor(
            w2_torch,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=devices[i],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for i in range(len(devices))
    ]
    logger.info("Pushing w1 input to devices")
    w1_input_ttnn = [
        ttnn.from_torch(
            w1_input_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=devices[i],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for i in range(len(devices))
    ]
    logger.info("Pushing w2 input to devices")
    w2_input_ttnn = [
        ttnn.from_torch(
            w2_input_torch,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=devices[i],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for i in range(len(devices))
    ]

    core_grid = ttnn.CoreGrid(y=6, x=6) if use_small_core_grid else ttnn.CoreGrid(y=7, x=7)
    for i in range(len(devices)):
        for _ in range(1000):
            w1_out = ttnn.linear(
                w1_input_ttnn[i],
                w1_ttnn_tensors[i],
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=True,
                    dst_full_sync_en=True,
                ),
                dtype=ttnn.bfloat8_b,
                core_grid=core_grid,
            )
            w2_out = ttnn.linear(
                w2_input_ttnn[i],
                w2_ttnn_tensors[i],
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=True,
                    dst_full_sync_en=True,
                ),
                dtype=ttnn.bfloat8_b,
                core_grid=core_grid,
            )
            w1_out = ttnn.linear(
                w1_input_ttnn[i],
                w1_ttnn_tensors[i],
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=True,
                    dst_full_sync_en=True,
                ),
                dtype=ttnn.bfloat8_b,
                core_grid=core_grid,
            )
