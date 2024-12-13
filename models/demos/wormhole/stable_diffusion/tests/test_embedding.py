# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import StableDiffusionPipeline

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import (
    skip_for_grayskull,
)

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_embeddings import TtTimestepEmbedding
import pytest
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import get_mesh_mappers


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_embeddings(
    device,
):
    # device = mesh_device
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)

    torch.manual_seed(0)
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)

    model = pipe.unet
    model.eval()
    time_embedding = model.time_embedding

    parameters = preprocess_model_parameters(initialize_model=lambda: time_embedding, device=device)
    model = TtTimestepEmbedding(parameters=parameters)
    N = 1
    input = torch.randn([N * 2 if inputs_mesh_mapper else N, 1, 2, 320])
    torch_output = time_embedding(input)

    input = ttnn.from_torch(
        input,
        ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=inputs_mesh_mapper,
    )

    ttnn_output = model(input)
    ttnn_output = ttnn.to_torch(ttnn_output, mesh_composer=output_mesh_composer)
    print(f"torch_output: {torch_output.shape} ttnn_output: {ttnn_output.shape}")
    ttnn_output = ttnn_output.reshape(torch_output.shape)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
