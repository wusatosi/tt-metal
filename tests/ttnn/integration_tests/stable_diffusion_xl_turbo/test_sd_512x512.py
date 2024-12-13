# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc
from diffusers import AutoPipelineForText2Image
from models.demos.stable_diffusion_xl_turbo.tt.sd_512_512 import (
    sd_geglu,
    sd_feed_forward,
    sd_attention,
    sd_basic_transformer_block,
    sd_transformer_2d,
    ResnetBlock2D,
    sd_downsample_2,
    sd_cross_attention_down_blocks2d,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.stable_diffusion_xl_turbo.tt.utils import custom_preprocessor, custom_preprocessor_resnet
from models.demos.stable_diffusion_xl_turbo.tt.resnetblock2d_utils import update_params


@pytest.mark.parametrize(
    "N, C, H, W, attention_head_dim, index",
    [
        (1, 2, 1024, 640, 40, 1),
        (1, 2, 256, 1280, 40, 2),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_basic_transformer_block_512_512(device, N, C, H, W, attention_head_dim, index, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp16"
    )
    model = pipe.unet
    model.eval()
    config = model.config

    basic_transformer = model.down_blocks[index].attentions[0].transformer_blocks[0]

    hidden_states = torch.randn((N, H, W), dtype=torch.float32)
    encoder_hidden_states = torch.randn((1, 77, 2048), dtype=torch.float32)

    torch_output = basic_transformer(hidden_states, encoder_hidden_states=encoder_hidden_states)

    timestep = None
    attention_mask = None
    cross_attention_kwargs = None
    class_labels = None

    parameters = preprocess_model_parameters(initialize_model=lambda: basic_transformer, device=device)
    ttnn_hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_encoder_hidden_states = ttnn.from_torch(
        encoder_hidden_states,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    ttnn_output = sd_basic_transformer_block(
        hidden_states=ttnn_hidden_states,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        timestep=timestep,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        class_labels=class_labels,
        config=config,
        attention_head_dim=attention_head_dim,
        parameters=parameters,
        device=device,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.996)


@pytest.mark.parametrize(
    "input_shape, index1, index2, attention_head_dim, block, transformer_layers_per_block",
    [
        ((1, 640, 32, 32), 1, 0, 10, "down", 2),
        ((1, 1280, 16, 16), 2, 0, 20, "down", 10),
        # ((1, 640, 32, 32), 1, 0, 10, "up", 2),
        # ((1, 1280, 16, 16), 0, 0, 20, "up", 10),
        # ((1, 1280, 16, 16), 0, 0, 20, "mid", 10),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_transformer_2d_model_512_512(
    input_shape, index1, index2, block, attention_head_dim, transformer_layers_per_block, device, reset_seeds
):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp16"
    )
    model = pipe.unet
    model.eval()
    config = model.config

    num_attention_heads = 8
    norm_num_groups = 32

    if block == "down":
        transformer_2d_model = model.down_blocks[index1].attentions[index2]
    elif block == "up":
        transformer_2d_model = model.up_blocks[index1].attentions[index2]
    else:
        transformer_2d_model = model.mid_block.attentions[index1]

    hidden_states = torch.randn(input_shape, dtype=torch.float32)
    encoder_hidden_states = torch.randn((1, 77, 2048), dtype=torch.float32)

    torch_output = transformer_2d_model(hidden_states, encoder_hidden_states=encoder_hidden_states)

    timestep = None
    attention_mask = None
    cross_attention_kwargs = None
    class_labels = None
    return_dict = False

    parameters = preprocess_model_parameters(initialize_model=lambda: transformer_2d_model, device=device)

    ttnn_hidden_states = ttnn.from_torch(
        hidden_states, dtype=ttnn.bfloat16, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    ttnn_encoder_hidden_states = ttnn.from_torch(
        encoder_hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_output = sd_transformer_2d(
        hidden_states=ttnn_hidden_states,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        parameters=parameters,
        device=device,
        timestep=timestep,
        class_labels=class_labels,
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=return_dict,
        attention_head_dim=attention_head_dim,
        transformer_layers_per_block=transformer_layers_per_block,
        norm_num_groups=norm_num_groups,
        attention_mask=attention_mask,
        config=config,
        eps=1e-06,
    )
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output.sample, ttnn_output, 0.97)
