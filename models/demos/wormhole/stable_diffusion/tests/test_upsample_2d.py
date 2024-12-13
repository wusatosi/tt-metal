# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers import StableDiffusionPipeline
import pytest
import ttnn

from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_upsample_2d_new_conv import upsample2d
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import (
    skip_for_grayskull,
)
from ttnn.model_preprocessing import preprocess_model_parameters

from models.utility_functions import torch_random
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    pre_process_input,
    post_process_output,
    get_mesh_mappers,
)
from models.demos.wormhole.stable_diffusion_dp.tests.custom_preprocessing import create_custom_mesh_preprocessor


@skip_for_grayskull()
@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width, index",
    [
        (2, 1280, 8, 8, 0),
        (2, 1280, 16, 16, 1),
        (2, 640, 32, 32, 2),
    ],
)
@pytest.mark.parametrize("scale_factor", [2])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_upsample2d_512x512(device, scale_factor, batch_size, in_channels, input_height, input_width, index):
    # device = mesh_device
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)

    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)

    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    unet_upblock = pipe.unet.up_blocks[index]
    resnet_upsampler = unet_upblock.upsamplers[0]
    reader_patterns_cache = {}

    batch_size = batch_size * 2 if inputs_mesh_mapper else batch_size

    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet,
        custom_preprocessor=(
            create_custom_mesh_preprocessor(weights_mesh_mapper) if weights_mesh_mapper else custom_preprocessor
        ),
        device=device,
    )
    parameters = parameters.up_blocks[index].upsamplers[0]

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    model = upsample2d(
        device, parameters, reader_patterns_cache, batch_size, input_height, input_width, compute_kernel_config
    )

    input_shape = batch_size, in_channels, input_height, input_width
    out_channels = in_channels

    input = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output = resnet_upsampler(input)

    input = input.permute(0, 2, 3, 1)
    input = input.reshape(1, 1, batch_size * input_height * input_width, in_channels)
    tt_input_tensor = ttnn.from_torch(
        input, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper
    )
    tt_up = model(
        tt_input_tensor,
        in_channels,
        out_channels,
    )

    torch_up = ttnn.to_torch(tt_up, mesh_composer=output_mesh_composer)
    torch_up = torch_up.reshape(batch_size, input_height * 2, input_width * 2, in_channels)
    torch_up = torch_up.permute(0, 3, 1, 2)
    assert_with_pcc(torch_output, torch_up, 0.99)
