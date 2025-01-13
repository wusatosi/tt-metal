# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from diffusers import AutoPipelineForText2Image
from models.demos.stable_diffusion.tt.utils import custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.demos.stable_diffusion.tt.resnetblock2d_utils import update_params
from models.demos.stable_diffusion.tt.ttnn_optimized_sdxl_turbo import stable_diffusion_xl_turbo
from models.demos.stable_diffusion.demo.ttnn_pipeline import ttnnStableDiffusionXLPipeline


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_demo(device, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp16"
    )
    model = pipe.unet
    model.eval()
    config = model.config
    config["dtype"] = torch.float32

    prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
    # prompt = "a photo of an astronaut riding a horse on mars"
    torch_image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

    ttnn_pipe = ttnnStableDiffusionXLPipeline(
        config=config,
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=stable_diffusion_xl_turbo,
        scheduler=pipe.scheduler,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    parameters = update_params(parameters)

    ttnn_image = ttnn_pipe(
        prompt=prompt,
        num_inference_steps=1,
        guidance_scale=0.0,
        parameters_ttnn=parameters,
        device_ttnn=device,
    ).images[0]

    ttnn_image.save("TTNN_output_SD_XL_512_512.png")
    torch_image.save("TORCH_output_SD_XL_512_512.png")
