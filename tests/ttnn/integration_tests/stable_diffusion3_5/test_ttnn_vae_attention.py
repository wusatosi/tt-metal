# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_stable_diffusion3_5.reference.attention_vae import Attention
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_attention_vae import ttnn_Attention


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name):
        parameters = {}
        if isinstance(model, Attention):
            parameters["group_norm"] = {}
            parameters["group_norm"]["weight"] = ttnn.from_torch(model.group_norm.weight, dtype=ttnn.bfloat16)
            parameters["group_norm"]["bias"] = ttnn.from_torch(model.group_norm.bias, dtype=ttnn.bfloat16)
            parameters["to_q"] = {}
            parameters["to_q"]["weight"] = preprocess_linear_weight(model.to_q.weight, dtype=ttnn.bfloat8_b)
            parameters["to_q"]["bias"] = preprocess_linear_bias(model.to_q.bias, dtype=ttnn.bfloat8_b)
            parameters["to_k"] = {}
            parameters["to_k"]["weight"] = preprocess_linear_weight(model.to_k.weight, dtype=ttnn.bfloat8_b)
            parameters["to_k"]["bias"] = preprocess_linear_bias(model.to_k.bias, dtype=ttnn.bfloat8_b)
            parameters["to_v"] = {}
            parameters["to_v"]["weight"] = preprocess_linear_weight(model.to_v.weight, dtype=ttnn.bfloat8_b)
            parameters["to_v"]["bias"] = preprocess_linear_bias(model.to_v.bias, dtype=ttnn.bfloat8_b)

            parameters["to_out"] = {}
            parameters["to_out"][0] = {}
            parameters["to_out"][0]["weight"] = preprocess_linear_weight(model.to_out[0].weight, dtype=ttnn.bfloat8_b)
            parameters["to_out"][0]["bias"] = preprocess_linear_bias(model.to_out[0].bias, dtype=ttnn.bfloat8_b)

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull()
def test_vae_attention(device, reset_seeds):
    reference_model = Attention(
        query_dim=512,
        heads=1,
        dim_head=512,
        rescale_output_factor=1,
        eps=1e-06,
        norm_num_groups=32,
        spatial_norm_dim=None,
        residual_connection=True,
        bias=True,
        upcast_softmax=True,
        _from_deprecated_attn_block=True,
        processor=None,
    ).to(dtype=torch.bfloat16)

    reference_model.eval()

    hidden_states = torch.randn(1, 512, 64, 64, dtype=torch.bfloat16)

    torch_output = reference_model(hidden_states=hidden_states, temb=None)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, device=device, custom_preprocessor=create_custom_preprocessor(device)
    )

    ttnn_model = ttnn_Attention(
        query_dim=512,
        heads=1,
        dim_head=512,
        rescale_output_factor=1,
        eps=1e-06,
        norm_num_groups=32,
        spatial_norm_dim=None,
        residual_connection=True,
        bias=True,
        upcast_softmax=True,
        _from_deprecated_attn_block=True,
        parameters=parameters,
        device=device,
    )

    ttnn_hidden_states = ttnn.from_torch(hidden_states, device=device, layout=ttnn.TILE_LAYOUT)

    ttnn_output = ttnn_model(
        hidden_states=ttnn_hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None
    )

    assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output), pcc=0.99)
