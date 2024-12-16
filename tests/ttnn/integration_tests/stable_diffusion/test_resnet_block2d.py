# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from diffusers import AutoPipelineForText2Image
from models.demos.stable_diffusion.tt.resnetblock2d import ResnetBlock2D
from models.demos.stable_diffusion.tt.utils import custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width, index1, index2, block_name",
    [
        (1, 320, 64, 64, 1, 0, "down"),
    ],
)
def test_resnet_block_2d_512x512(
    device, batch_size, in_channels, input_height, input_width, index1, index2, block_name
):
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
    model = pipe.unet
    model.eval()
    config = model.config

    parameters = preprocess_model_parameters(
        # model_name=tt_model_name,
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    parameters = parameters.down_blocks[index1].resnets[index2]
    print(parameters.conv1.weight.get_dtype())
    print(parameters.conv1.bias.get_dtype())

    # print(model)
    resnet = model.down_blocks[index1].resnets[index2]

    temb_channels = 1280
    groups = 32
    time_embedding_norm = "default"
    output_scale_factor = 1

    hidden_states_shape = [batch_size, in_channels, input_height, input_width]
    temb_shape = [1, 1280]

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
    temb = ttnn.from_torch(temb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    resnet_block = ResnetBlock2D(
        config,
        input,
        temb,
        in_channels,
        input_height,
        input_width,
        parameters,
        device,
        conv_shortcut=True,
    )
    resnet_block = ttnn.to_torch(resnet_block)
    assert_with_pcc(torch_output, resnet_block, 0.99)
