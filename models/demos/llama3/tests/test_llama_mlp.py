# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3.tt.llama_mlp import TtLlamaMLP
from models.demos.llama3.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import FeedForward
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from models.demos.t3000.llama2_70b.tt.llama_common import ShardTensor2dMesh, ConcatMesh2DToTensor


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "seq_len",
    (
        # 64 * 1024,
        # 32 * 1024,
        512,
        # 128,
        # 32,
    ),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_llama_mlp_inference(mesh_device, seq_len, use_program_cache, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat8_b
    mode = "decode" if seq_len <= 32 else "prefill"

    mesh_device.enable_async(True)

    model_args = TtModelArgs(mesh_device)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("TtLlamaMLP", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    model_args.WEIGHTS_DTYPE = dtype
    reference_model = FeedForward(
        dim=model_args.dim,
        hidden_dim=4 * model_args.dim,
        multiple_of=model_args.multiple_of,
        ffn_dim_multiplier=model_args.ffn_dim_multiplier,
    )
    reference_model.load_state_dict(partial_state_dict)

    tt_model = TtLlamaMLP(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=model_args.get_model_config(),
    )
    torch_input = torch.randn(1, 1, seq_len, model_args.dim)
    reference_output = reference_model(torch_input)
    TG = model_args.is_galaxy
    if TG:
        if mode == "decode":
            num_users = 32
            M, K = num_users, model_args.model_config["HIDDEN_SIZE"] // 4

            core_grid = ttnn.CoreGrid(y=1, x=8)
            act_mem_config = ttnn.create_sharded_memory_config(
                shape=(M // core_grid.y, K // core_grid.x),
                core_grid=core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            tt_input = ttnn.from_torch(
                torch_input,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=act_mem_config,
                mesh_mapper=ShardTensor2dMesh(mesh_device, dims=(3, None), cluster_shape=(4, 8)),
            )
        else:
            tt_input = ttnn.from_torch(
                torch_input,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ShardTensor2dMesh(mesh_device, dims=(3, None), cluster_shape=(4, 8)),
            )
    else:
        tt_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat8_b,
            memory_config=model_args.model_config["SHARDED_MLP_INPUT_MEMCFG"]
            if mode == "decode"
            else ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

    logger.info("Run Llama_MLP")
    tt_output = tt_model(tt_input, mode)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ConcatMesh2DToTensor(mesh_device, dims=(3, 1), cluster_shape=(4, 8))
        if TG
        else ttnn.ConcatMeshToTensor(mesh_device, dim=-1),
    )

    if TG:
        tt_output_torch = tt_output_torch[:, :1, :, :]

    print(f"{tt_output_torch.shape=}")
    print(f"{reference_output.shape=}")

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("Llama_MLP Passed!")
    else:
        logger.warning("Llama_MLP Failed!")

    assert passing, f"Llama_MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
