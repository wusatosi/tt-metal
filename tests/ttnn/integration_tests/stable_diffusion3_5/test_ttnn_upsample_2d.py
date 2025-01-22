# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest


from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_stable_diffusion3_5.reference.upsample_2d import Upsample2D
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_upsample_2d import ttnnUpsample2D
from diffusers import StableDiffusion3Pipeline


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, Upsample2D):
            parameters["conv"] = {}
            parameters["conv"]["weight"] = ttnn.from_torch(model.conv.weight, dtype=ttnn.bfloat16)
            parameters["conv"]["bias"] = ttnn.from_torch(
                torch.reshape(model.conv.bias, (1, 1, 1, -1)),
                dtype=ttnn.bfloat16,
            )

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull()
@pytest.mark.parametrize(
    "x_shape,i",
    [
        ([1, 512, 64, 64], 0),
        # ([1, 512, 128, 128],1),#OOM
        # ([1, 256, 256, 256],2),#OOM
    ],
)
def test_upsample_2d(device, x_shape, i, reset_seeds):
    # Note:This test takes 10 minutes to run in cpu(the torch model is taking a long time)
    reference_model = Upsample2D(channels=x_shape[1], use_conv=True, out_channels=x_shape[1]).to(dtype=torch.bfloat16)
    torch_input = torch.randn(x_shape, dtype=torch.bfloat16)
    reference_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )

    torch_output = reference_model(torch_input)

    torch_input_permuted = torch_input.permute(0, 2, 3, 1)
    ttnn_input = ttnn.from_torch(torch_input_permuted, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_model = ttnnUpsample2D(channels=x_shape[1], use_conv=True, out_channels=x_shape[1], parameters=parameters)

    ttnn_output = ttnn_model(ttnn_input, None)
    ttnn_output = ttnn.to_torch(ttnn_output)

    ttnn_output = ttnn_output.permute(0, 3, 1, 2)

    assert_with_pcc(ttnn_output, torch_output, 0.99)
