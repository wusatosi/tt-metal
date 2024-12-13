# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from diffusers import StableDiffusionPipeline
import ttnn

from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_basic_transformer_block import basic_transformer_block
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    pre_process_input,
    post_process_output,
)
from models.utility_functions import (
    skip_for_grayskull,
)
from models.demos.wormhole.stable_diffusion_dp.tests.custom_preprocessing import create_custom_mesh_preprocessor


def get_mesh_mappers(device):
    is_mesh_device = isinstance(device, ttnn.MeshDevice)
    if is_mesh_device:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        weights_mesh_mapper = (
            None  # ttnn.ReplicateTensorToMesh(device) causes unnecessary replication/takes more time on the first pass
        )
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
    else:
        inputs_mesh_mapper = None
        weights_mesh_mapper = None
        output_mesh_composer = None
    return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize(
    "N, C, H, W, index, attention_head_dim",
    [
        (
            1,
            2,
            4096,
            320,
            3,
            40,
        ),
        (
            1,
            2,
            1024,
            640,
            2,
            80,
        ),
        (
            1,
            2,
            256,
            1280,
            1,
            160,
        ),
        (
            1,
            2,
            64,
            1280,
            1,
            160,
        ),
    ],
)
def test_basic_transformer_block_512x512(mesh_device, model_name, N, C, H, W, index, attention_head_dim):
    torch.manual_seed(0)
    device = mesh_device
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    model = pipe.unet
    model.eval()
    config = model.config
    basic_transformer = pipe.unet.up_blocks[index].attentions[1].transformer_blocks[0]
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)
    # N = N if inputs_mesh_mapper is None else N * 2
    hidden_states_shape = torch.Size([N, C, H, W])
    hidden_states = torch.rand(hidden_states_shape) * 0.01
    encoder_hidden_states_shape = [N, 2, 77, 768]
    encoder_hidden_states = torch.rand(encoder_hidden_states_shape)

    timestep = None
    attention_mask = None
    cross_attention_kwargs = None
    class_labels = None
    print(f"hidden_states: {hidden_states.shape} encoder_hidden_states: {encoder_hidden_states.shape}")
    torch_output = basic_transformer(hidden_states.squeeze(0), encoder_hidden_states.squeeze(0))

    parameters = preprocess_model_parameters(
        model_name=model_name,
        initialize_model=lambda: basic_transformer,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )
    model = basic_transformer_block(device, parameters, seq_len=H)

    hidden_states = ttnn.from_torch(
        hidden_states,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=inputs_mesh_mapper,
    )
    encoder_hidden_states = torch.nn.functional.pad(encoder_hidden_states, (0, 0, 0, 19))
    encoder_hidden_states = ttnn.from_torch(
        encoder_hidden_states,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=inputs_mesh_mapper,
    )

    grid_sizes = {8192: (5, 8), 2048: (5, 8), 512: (8, 8), 128: (8, 4)}
    hidden_states = ttnn.reshape(
        hidden_states, [1, 1, hidden_states.shape[-3] * hidden_states.shape[-2], hidden_states.shape[-1]]
    )

    grid_size = grid_sizes[hidden_states.shape[-2]]
    hidden_states = ttnn.interleaved_to_sharded(
        hidden_states,
        grid_size,
        [hidden_states.volume() // hidden_states.shape[-1] // grid_size[1], hidden_states.shape[-1] // grid_size[0]],
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    ttnn_output = model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        class_labels=class_labels,
        config=config,
        attention_head_dim=attention_head_dim,
    )

    ttnn_output = ttnn.reshape(ttnn_output, [1, 2, ttnn_output.shape[-2] // 2, ttnn_output.shape[-1]])
    ttnn_output = ttnn.to_torch(ttnn_output, mesh_composer=output_mesh_composer)

    assert_with_pcc(torch_output.unsqueeze(0), ttnn_output, pcc=0.98)
