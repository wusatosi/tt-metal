import torch
import pytest
from loguru import logger
import os
import ttnn
from models.experimental.mochi.tt.dit.mlp import FeedForward as TtFeedForward
from models.experimental.mochi.tt.common import (
    get_mochi_dir,
    get_cache_path,
    compute_metrics,
    pad_vision_seq_parallel,
)
from models.experimental.mochi.tt.dit.config import MochiConfig
from models.utility_functions import (
    comp_allclose,
)

from genmo.mochi_preview.dit.joint_model.layers import FeedForward as RefFeedForward

# Get model configuration
CONFIG = MochiConfig()


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize(
    "ff_path, in_feat, seq_len, seq_shard",
    [
        ("blocks.0.mlp_x", CONFIG.hidden_size_x, 44520, True),
        ("blocks.0.mlp_y", CONFIG.hidden_size_y, 118, False),
    ],
)
@pytest.mark.parametrize("real_weights", [True, False], ids=["real_weights", "random_weights"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
def test_tt_feedforward_inference(mesh_device, seq_len, use_program_cache, ff_path, in_feat, seq_shard, real_weights):
    # Fabric setup
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    # create global semaphore handles
    if seq_shard:
        ccl_semaphore_handles = {
            s: ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for s in ["w1", "w2", "w3"]
        }
    else:
        ccl_semaphore_handles = {s: ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for s in ["w2_in"]}

    dtype = ttnn.bfloat16

    multiple_of = 256
    mlp_ratio = CONFIG.mlp_ratio_x if "mlp_x" in ff_path else CONFIG.mlp_ratio_y
    mlp_hidden_dim = int(in_feat * mlp_ratio)

    reference_model = RefFeedForward(
        in_features=in_feat,
        hidden_size=mlp_hidden_dim,
        multiple_of=multiple_of,
        ffn_dim_multiplier=None,
    )
    if real_weights:
        from safetensors.torch import load_file

        weights_path = os.path.join(get_mochi_dir(), "dit.safetensors")
        state_dict = load_file(weights_path)
        partial_state_dict = {k[len(ff_path) + 1 :]: v for k, v in state_dict.items() if k.startswith(ff_path)}
        reference_model.load_state_dict(partial_state_dict)
        weight_cache_path = get_cache_path(os.environ.get("FAKE_DEVICE"))
    else:
        state_dict = reference_model.state_dict()
        weight_cache_path = None
        state_dict = {f"{ff_path}.{k}": v for k, v in state_dict.items()}

    tt_model = TtFeedForward(
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        layer_num=0,
        in_features=in_feat,
        hidden_size=mlp_hidden_dim,
        multiple_of=multiple_of,
        ffn_dim_multiplier=None,
        state_dict_prefix=ff_path,
        seq_shard=seq_shard,
    )
    torch_input = torch.randn(1, 1, seq_len, in_feat)
    if seq_shard:
        mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-2)
        tt_input = pad_vision_seq_parallel(torch_input, mesh_device.get_num_devices())
    else:
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        tt_input = torch_input
    tt_input = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        mesh_mapper=mapper,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Run TtFeedForward")
    tt_output = tt_model(tt_input, ccl_semaphore_handles, worker_sub_device_id, ttnn.Topology.Ring)

    if seq_shard:
        tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-2))[
            :, :, :seq_len, :
        ]
    else:
        tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))

    # Tear down what we created for fabric
    mesh_device.clear_loaded_sub_device_manager()

    # Get reference output from the reference model
    reference_output = reference_model(torch_input)

    # Compute metrics
    pcc, mse, mae = compute_metrics(reference_output, tt_output_torch)
    # Check if model meets requirements
    pcc_required = 0.999
    passing = pcc >= pcc_required

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    if passing:
        logger.info("TtFeedForward Passed!")
    else:
        logger.warning("TtFeedForward Failed!")

    assert passing, f"TtFeedForward output does not meet PCC requirement {pcc_required}: {pcc}, MSE: {mse}, MAE: {mae}."
