# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters, ParameterDict, ParameterList
from torch import nn
from diffusers import StableDiffusion3Pipeline
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_stable_diffusion3_5.reference.t5_encoder_model import T5EncoderModel
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_t5_encoder_model import (
    ttnn_T5EncoderModel,
)


def move_to_TILE(object):
    if isinstance(object, ParameterDict):
        for name, value in list(object.items()):
            object[name] = move_to_TILE(value)
        return object
    elif isinstance(object, ParameterList):
        for index, element in enumerate(object):
            object[index] = move_to_TILE(element)
        return object
    elif isinstance(object, ttnn.Tensor):
        return ttnn.to_layout(object, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    else:
        return object


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull()
def test_t5_encoder_model(device, reset_seeds):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
    )
    config = pipe.text_encoder_3.config

    reference_model = T5EncoderModel(config).to(dtype=torch.bfloat16)

    reference_model.eval()

    input_ids = torch.normal(0.0, 30.0, size=(1, 256))
    input_ids = input_ids.abs()
    input_ids = input_ids.to(torch.int64)

    # torch_output = reference_model(
    #     input_ids,
    # )

    parameters = preprocess_model_parameters(initialize_model=lambda: reference_model, device=device)
    parameters = move_to_TILE(parameters)

    ttnn_input_ids = ttnn.from_torch(
        input_ids,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_model = ttnn_T5EncoderModel(config, parameters=parameters)

    ttnn_output = ttnn_model(ttnn_input_ids, parameters=parameters)
