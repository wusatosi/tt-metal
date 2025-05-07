import os
import pytest
import torch
import ttnn
from loguru import logger

from models.utility_functions import (
    comp_allclose,
)

from genmo.mochi_preview.dit.joint_model.asymm_models_joint import AsymmetricJointBlock
from models.experimental.mochi.tt.dit.model import AsymmetricJointBlock as TtAsymmetricJointBlock
from models.experimental.mochi.tt.dit.config import MochiConfig
from models.experimental.mochi.tt.common import (
    get_cache_path,
    compute_metrics,
    to_tt_tensor,
    to_torch_tensor,
    stack_cos_sin,
    pad_vision_seq_parallel,
)
from models.tt_transformers.tt.common import get_rot_transformation_mat
from models.experimental.mochi.tests.dit.common import (
    load_model_weights,
)

# Get model configuration
CONFIG = MochiConfig()

# Common test configurations
PCC_REQUIRED = 0.99
NUM_HEADS = CONFIG.num_heads
HEAD_DIM = CONFIG.hidden_size_x // NUM_HEADS

block_kwargs = {
    "qk_norm": CONFIG.qk_norm,
    "qkv_bias": CONFIG.qkv_bias,
    "out_bias": CONFIG.out_bias,
    "attention_mode": CONFIG.attention_mode,
}


def create_models(mesh_device, block_path, update_y=True, real_weights=True):
    """Initialize both reference and TT models."""
    reference_model = AsymmetricJointBlock(
        hidden_size_x=CONFIG.hidden_size_x,
        hidden_size_y=CONFIG.hidden_size_y,
        num_heads=NUM_HEADS,
        mlp_ratio_x=CONFIG.mlp_ratio_x,
        mlp_ratio_y=CONFIG.mlp_ratio_y,
        update_y=update_y,
        device="cpu",
        **block_kwargs,
    )
    if real_weights:
        state_dict, partial_state_dict = load_model_weights(block_path)
        reference_model.load_state_dict(partial_state_dict)
        weight_cache_path = get_cache_path(os.environ.get("MESH_DEVICE"))
    else:
        state_dict = reference_model.state_dict()
        # Initialize RMSNorm weights to random values
        for k in state_dict.keys():
            if "norm" in k:
                state_dict[k] = torch.randn_like(state_dict[k])
        reference_model.load_state_dict(state_dict)
        state_dict = {f"{block_path}.{k}": v for k, v in state_dict.items()}
        weight_cache_path = None

    tt_model = TtAsymmetricJointBlock(
        mesh_device=mesh_device,
        state_dict=state_dict,
        state_dict_prefix=block_path,
        weight_cache_path=weight_cache_path,
        layer_num=0,
        dtype=ttnn.bfloat16,
        hidden_size_x=CONFIG.hidden_size_x,
        hidden_size_y=CONFIG.hidden_size_y,
        num_heads=NUM_HEADS,
        mlp_ratio_x=CONFIG.mlp_ratio_x,
        mlp_ratio_y=CONFIG.mlp_ratio_y,
        update_y=update_y,
        multiple_of=256,
        ffn_dim_multiplier=None,
        **block_kwargs,
    )
    return reference_model, tt_model


@torch.no_grad()
@pytest.mark.parametrize(
    "vision_seq_len, text_seq_len",
    [
        (22 * 256 * 8, CONFIG.t5_token_length),
        (44520, 118),
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [{"T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize(
    "block_path, update_y",
    [
        ("blocks.0", True),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("real_weights", [True, False], ids=["real_weights", "random_weights"])
def test_tt_block(
    mesh_device, vision_seq_len, text_seq_len, use_program_cache, reset_seeds, block_path, update_y, real_weights
):
    """Test TtAsymmetricJointBlock implementation by comparing with reference model."""

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
    ccl_semaphore_handles = {
        s: ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)
        for s in [
            "ff_block_y",
            "mod_x",
            "mod_y",
            "y_attn",
            "seq_to_col",
            "col_to_seq",
            "qkv_x",
            "proj_x",
            "out_joint",
            "w1",
            "w2",
            "w3",
            "w2_in",
        ]
    }

    min_pcc = 0.997
    max_mse = 0.0089

    # Create reference model
    reference_model, tt_model = create_models(mesh_device, block_path, update_y, real_weights)
    # Create input tensors
    batch_size = 1
    x_input = torch.randn(batch_size, vision_seq_len, CONFIG.hidden_size_x)
    y_input = torch.randn(batch_size, text_seq_len, CONFIG.hidden_size_y)
    c_input = torch.randn(batch_size, CONFIG.hidden_size_x)  # Conditioning tensor

    # Create RoPE tensors
    rope_cos = torch.randn(vision_seq_len, NUM_HEADS, HEAD_DIM // 2)
    rope_sin = torch.randn(vision_seq_len, NUM_HEADS, HEAD_DIM // 2)

    # Stack cos/sin for TT model
    rope_cos_stack, rope_sin_stack = stack_cos_sin(
        rope_cos.unsqueeze(0).permute(0, 2, 1, 3), rope_sin.unsqueeze(0).permute(0, 2, 1, 3)
    )

    # Create transformation matrix for RoPE
    trans_mat = get_rot_transformation_mat(None)

    # Pre-applied SILU expected in block
    c_silu = torch.nn.functional.silu(c_input).view(batch_size, 1, 1, CONFIG.hidden_size_x)

    # Create valid token indices
    total_seq_len = vision_seq_len + text_seq_len
    valid_token_indices = torch.arange(total_seq_len)
    max_seqlen_in_batch = total_seq_len

    # Convert inputs to TT tensors
    x_padded = pad_vision_seq_parallel(
        x_input.view(1, batch_size, vision_seq_len, CONFIG.hidden_size_x), mesh_device.get_num_devices()
    )
    cos_padded = pad_vision_seq_parallel(rope_cos_stack, mesh_device.get_num_devices())
    sin_padded = pad_vision_seq_parallel(rope_sin_stack, mesh_device.get_num_devices())
    tt_x = to_tt_tensor(x_padded, mesh_device, shard_dim=-2)
    tt_y = to_tt_tensor(y_input.view(1, batch_size, text_seq_len, CONFIG.hidden_size_y), mesh_device)
    tt_c = to_tt_tensor(c_silu, mesh_device)
    tt_rope_cos = to_tt_tensor(cos_padded, mesh_device, shard_dim=-3)
    tt_rope_sin = to_tt_tensor(sin_padded, mesh_device, shard_dim=-3)
    tt_trans_mat = to_tt_tensor(trans_mat, mesh_device)

    persistent_buffers = {
        "seq_to_col_intermediate": ttnn.from_torch(
            torch.zeros([1, batch_size, x_padded.shape[2], 3 * HEAD_DIM * NUM_HEADS // mesh_device.get_num_devices()]),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
        "seq_to_col_output": ttnn.from_torch(
            torch.zeros([1, batch_size, x_padded.shape[2], 3 * HEAD_DIM * NUM_HEADS // mesh_device.get_num_devices()]),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
        "col_to_seq_intermediate": ttnn.from_torch(
            torch.zeros([1, batch_size, x_padded.shape[2] // mesh_device.get_num_devices(), HEAD_DIM * NUM_HEADS]),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
        "col_to_seq_output": ttnn.from_torch(
            torch.zeros([1, batch_size, x_padded.shape[2] // mesh_device.get_num_devices(), HEAD_DIM * NUM_HEADS]),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
    }

    # Create packed indices
    packed_indices = {
        "max_seqlen_in_batch_kv": max_seqlen_in_batch,
        "valid_token_indices_kv": valid_token_indices,
        "cu_seqlens_kv": None,
    }

    logger.info("Run TtAsymmetricJointBlock forward")
    tt_x_out, tt_y_out = tt_model(
        tt_x,
        tt_c,
        tt_y,
        N=vision_seq_len,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
        trans_mat=tt_trans_mat,
        ccl_semaphore_handles=ccl_semaphore_handles,
        worker_sub_device_id=worker_sub_device_id,
        topology=ttnn.Topology.Ring,
        persistent_buffers=persistent_buffers,
    )

    # Convert TT outputs to torch tensors
    # extract from replicated tensors
    tt_x_torch = to_torch_tensor(tt_x_out, mesh_device, dim=2)[:, :, :vision_seq_len, :]
    tt_y_torch = to_torch_tensor(tt_y_out, mesh_device, dim=0)[0:1]

    mesh_device.clear_loaded_sub_device_manager()

    # Get reference outputs
    ref_x, ref_y = reference_model(
        x_input, c_input, y_input, rope_cos=rope_cos, rope_sin=rope_sin, packed_indices=packed_indices
    )

    # Validate outputs
    metrics = []
    for tt_out, ref_out, name in [(tt_x_torch, ref_x, "Visual"), (tt_y_torch, ref_y, "Text")]:
        pcc, mse, mae = compute_metrics(ref_out, tt_out)
        metrics.append((name, pcc, mse, mae))
        print(f"{name} - PCC: {pcc}, MSE: {mse}, MAE: {mae}")
        print(comp_allclose(ref_out, tt_out.view(ref_out.shape)))

    passing = all((mse <= max_mse) and (pcc >= min_pcc) for _, pcc, mse, _ in metrics)

    if passing:
        logger.info("TtAsymmetricJointBlock Passed!")
    else:
        logger.warning("TtAsymmetricJointBlock Failed!")
        for name, pcc, mse, mae in metrics:
            if pcc < PCC_REQUIRED:
                logger.error(f"{name} failed with PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    assert passing, f"TtAsymmetricJointBlock output does not meet PCC requirement {PCC_REQUIRED}"
