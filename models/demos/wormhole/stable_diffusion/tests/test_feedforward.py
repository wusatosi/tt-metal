# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
from diffusers import UNet2DConditionModel
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_feedforward import feedforward
from models.utility_functions import torch_random

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import (
    skip_for_grayskull,
)
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import conv_cache, get_mesh_mappers
from models.demos.wormhole.stable_diffusion_dp.tests.custom_preprocessing import create_custom_mesh_preprocessor


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize(
    "N, C, H, W, index",
    [
        (
            1,
            2,
            4096,
            320,
            3,
        ),
        (
            1,
            2,
            1024,
            640,
            2,
        ),
        (
            1,
            2,
            256,
            1280,
            1,
        ),
        (
            1,
            2,
            64,
            1280,
            1,
        ),
    ],
)
def test_feedforward_512x512(device, model_name, N, C, H, W, index, reset_seeds):
    # device = mesh_device
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)
    input_shapes = (N * 2 if inputs_mesh_mapper else N, C, H, W)

    model = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").eval()
    ref_model = model.up_blocks[index].attentions[0].transformer_blocks[0].ff
    config = model.config
    torch_hidden_states = torch_random(input_shapes, -0.1, 0.1, dtype=torch.float32)
    torch_output = ref_model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        device=device,
        custom_preprocessor=(
            create_custom_mesh_preprocessor(weights_mesh_mapper) if weights_mesh_mapper else custom_preprocessor
        ),
    )
    model = feedforward(device, parameters)

    ttnn_hidden_state = torch_hidden_states.reshape(
        (
            torch_hidden_states.shape[0],
            1,
            torch_hidden_states.shape[-3] * torch_hidden_states.shape[-2],
            torch_hidden_states.shape[-1],
        )
    )
    ttnn_hidden_state = ttnn.from_torch(
        ttnn_hidden_state, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=inputs_mesh_mapper
    )

    output = model(config, ttnn_hidden_state)

    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)
    output = output.reshape(torch_output.shape)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.99)
