# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from diffusers import AutoPipelineForText2Image
from models.demos.stable_diffusion_xl_turbo.tt.tt_downblock_2d import down_block_2d
from models.demos.stable_diffusion_xl_turbo.tt.tt_resnet_block_2d import update_params
from models.demos.stable_diffusion_xl_turbo.custom_preprocessing import custom_preprocessor_resnet
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width",
    [
        (1, 320, 128, 128),
    ],
)
def test_down_block_2d_1024x1024(
    device,
    batch_size,
    in_channels,
    input_height,
    input_width,
    reset_seeds,
):
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
    model = pipe.unet
    model.eval()
    config = model.config

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor_resnet,
        device=device,
    )
    parameters = update_params(parameters)

    parameters = parameters.down_blocks[0]
    resnet = model.down_blocks[0]

    temb_channels = 1280

    hidden_states_shape = [batch_size, in_channels, input_height, input_width]
    temb_shape = [1, temb_channels]

    input = torch.randn(hidden_states_shape)
    temb = torch.randn(temb_shape)
    torch_output = resnet(input, temb)

    input = ttnn.from_torch(
        input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        # memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    temb = ttnn.from_torch(
        temb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    resnet_block = down_block_2d(
        device, parameters, config, input, temb, in_channels=320, input_height=128, input_width=128, num_layers=2
    )
    resnet_block = ttnn.to_torch(resnet_block)
    assert_with_pcc(torch_output[0], resnet_block, 0.98)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width",
    [
        (1, 320, 64, 64),
    ],
)
def test_down_block_2d_512x512(
    device,
    batch_size,
    in_channels,
    input_height,
    input_width,
    reset_seeds,
):
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
    model = pipe.unet
    model.eval()
    config = model.config

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor_resnet,
        device=device,
    )
    parameters = update_params(parameters)

    parameters = parameters.down_blocks[0]
    resnet = model.down_blocks[0]

    temb_channels = 1280

    hidden_states_shape = [batch_size, in_channels, input_height, input_width]
    temb_shape = [1, temb_channels]

    input = torch.randn(hidden_states_shape)
    temb = torch.randn(temb_shape)
    torch_output = resnet(input, temb)

    input = ttnn.from_torch(
        input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    temb = ttnn.from_torch(
        temb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    resnet_block = down_block_2d(
        device, parameters, config, input, temb, in_channels=320, input_height=128, input_width=128, num_layers=2
    )
    resnet_block = ttnn.to_torch(resnet_block)
    assert_with_pcc(torch_output[0], resnet_block, 0.98)
