import pytest
import ttnn
import torch
from tests.ttnn.unit_tests.operations.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_subdevices.tt.llama_ccl import TT_CCL
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
def test_prefill_ccl_ff2(mesh_device, use_program_cache):
    prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1, mode="prefill")
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])
    tt_ccl = TT_CCL(mesh_device, crs, prefetcher_setup.worker_sub_device_id)
    w2_out = ttnn.from_torch(
        torch.zeros(1, 1, 128, 8192),
        dtype=ttnn.bfloat8_b,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device=mesh_device, dims=(None, 3), mesh_shape=list(mesh_device.shape)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    for _ in range(10):
        w2_out_reduced_tt = tt_ccl.line_all_reduce(
            w2_out, cluster_axis=0, num_links=3, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # reference
        w2_out_torch = ttnn.to_torch(
            w2_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=(8, 4)),
        )
        w2_out_torch_reduced = torch.sum(w2_out_torch, dim=0, keepdim=True)
        w2_out_reduced = ttnn.as_tensor(
            w2_out_torch_reduced,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device=mesh_device, dims=(None, 3), mesh_shape=list(mesh_device.shape)
            ),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        w2_out = w2_out_reduced

        tt_output_torch = ttnn.to_torch(
            w2_out_reduced_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=list(mesh_device.shape)),
        )
        ref_output_torch = ttnn.to_torch(
            w2_out_reduced,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=list(mesh_device.shape)),
        )
        passing, pcc_message = comp_pcc(tt_output_torch, ref_output_torch, 0.99)
        print("pcc_message", pcc_message)
    tt_ccl.close()

    assert passing


@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
def test_prefill_ccl_ff1(mesh_device, use_program_cache):
    prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1, mode="prefill")
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])
    tt_ccl = TT_CCL(mesh_device, crs, prefetcher_setup.worker_sub_device_id)
    w2_out = ttnn.from_torch(
        torch.randn(1, 1, 128, 14336),
        dtype=ttnn.bfloat8_b,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device=mesh_device, dims=(3, None), mesh_shape=list(mesh_device.shape)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    for _ in range(10):
        w2_out_reduced_tt = tt_ccl.line_all_reduce(
            w2_out, cluster_axis=1, num_links=3, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # reference
        w2_out_torch = ttnn.to_torch(
            w2_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=(8, 4)),
        )
        w2_out_torch_reduced = torch.sum(w2_out_torch, dim=0, keepdim=True)
        w2_out_reduced = ttnn.as_tensor(
            w2_out_torch_reduced,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device=mesh_device, dims=(3, None), mesh_shape=list(mesh_device.shape)
            ),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w2_out = w2_out_reduced

        tt_output_torch = ttnn.to_torch(
            w2_out_reduced_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 1), mesh_shape=list(mesh_device.shape)),
        )
        ref_output_torch = ttnn.to_torch(
            w2_out_reduced,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 1), mesh_shape=list(mesh_device.shape)),
        )
        print("tt_output_torch", tt_output_torch.shape, ref_output_torch.shape)
        passing, pcc_message = comp_pcc(tt_output_torch, ref_output_torch, 0.99)
        print("pcc_message", pcc_message)
    tt_ccl.close()

    assert passing
