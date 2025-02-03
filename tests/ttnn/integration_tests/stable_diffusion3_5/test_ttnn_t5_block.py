# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from diffusers import StableDiffusion3Pipeline
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_stable_diffusion3_5.reference.t5_block import T5Block
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_t5_block import (
    ttnn_T5Block,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull()
@pytest.mark.parametrize(
    "idx",
    [
        (0),
        (1),
    ],
)
def test_t5_block(device, reset_seeds, idx):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
    )
    config = pipe.text_encoder_3.config

    if idx == 0:
        has_relative_attention_bias = True
    else:
        has_relative_attention_bias = False

    reference_model = T5Block(config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=idx).to(
        dtype=torch.bfloat16
    )

    reference_model.eval()

    hidden_states = torch.randn(1, 256, 4096, dtype=torch.bfloat16)
    attention_mask = torch.randn(1, 1, 1, 256, dtype=torch.bfloat16)
    if idx == 0:
        position_bias = None
    else:
        position_bias = torch.randn(1, 64, 256, 256, dtype=torch.bfloat16)
    encoder_hidden_states = None
    encoder_attention_mask = None
    encoder_decoder_position_bias = None
    layer_head_mask = None
    cross_attn_layer_head_mask = None
    past_key_value = None
    use_cache = False
    output_attentions = False
    return_dict = True

    cache_position = torch.normal(0.0, 30.0, size=(1, 256))
    cache_position = cache_position.abs()
    cache_position = cache_position.to(torch.int64)
    cache_position = cache_position[0]

    torch_output = reference_model(
        hidden_states,
        attention_mask=attention_mask,
        position_bias=position_bias,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        encoder_decoder_position_bias=encoder_decoder_position_bias,
        layer_head_mask=layer_head_mask,
        cross_attn_layer_head_mask=cross_attn_layer_head_mask,
        past_key_value=past_key_value,
        use_cache=use_cache,
        output_attentions=output_attentions,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    parameters = preprocess_model_parameters(initialize_model=lambda: reference_model, device=device)

    ttnn_hidden_states = ttnn.from_torch(
        hidden_states,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_attention_mask = ttnn.from_torch(
        attention_mask,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_cache_position = ttnn.from_torch(
        cache_position,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    if idx == 0:
        ttnn_position_bias = None
    else:
        ttnn_position_bias = ttnn.from_torch(
            position_bias,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    ttnn_model = ttnn_T5Block(config, has_relative_attention_bias, idx, parameters)

    ttnn_output = ttnn_model(
        ttnn_hidden_states,
        attention_mask=ttnn_attention_mask,
        position_bias=ttnn_position_bias,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        encoder_decoder_position_bias=encoder_decoder_position_bias,
        layer_head_mask=layer_head_mask,
        cross_attn_layer_head_mask=cross_attn_layer_head_mask,
        past_key_value=past_key_value,
        use_cache=use_cache,
        output_attentions=output_attentions,
        cache_position=ttnn_cache_position,
        parameters=parameters,
    )

    assert_with_pcc(torch_output[0], ttnn.to_torch(ttnn_output[0]), pcc=0.99)
