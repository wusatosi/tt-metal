# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.utility_functions import skip_for_grayskull
from models.demos.wormhole.stable_diffusion.demo.demo import test_demo as demo
from models.demos.wormhole.stable_diffusion.demo.demo import test_demo_diffusiondb as demo_db


@pytest.mark.timeout(600)
@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_path",
    (("models/demos/wormhole/stable_diffusion/demo/input_data.json"),),
    ids=["default_input"],
)
@pytest.mark.parametrize(
    "num_prompts",
    ((1),),
)
@pytest.mark.parametrize(
    "num_inference_steps",
    ((50),),
)
@pytest.mark.parametrize(
    "image_size",
    ((512, 512),),
)
def test_demo_sd(device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size):
    if device.core_grid.y != 8:
        pytest.skip("Needs 8x8 Grid")
    demo(device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size)


@pytest.mark.timeout(600)
@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "num_prompts",
    ((1),),
)
@pytest.mark.parametrize(
    "num_inference_steps",
    ((50),),
)
@pytest.mark.parametrize(
    "image_size",
    ((512, 512),),
)
def test_demo_sd_db(device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size):
    if device.core_grid.y != 8:
        pytest.skip("Needs 8x8 Grid")
    demo_db(device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size)


from diffusers import StableDiffusionPipeline
import torch


def generate_image(prompt: str, output_path: str):
    # Load pre-trained Stable Diffusion model pipeline
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)

    pipe.to("cpu")

    # Generate an image based on the input prompt
    with torch.no_grad():  # No need to track gradients during inference
        image = pipe(prompt, num_inference_steps=50).images[0]

    # Save the generated image to the specified output path
    image.save(output_path)
    print(f"Image generated and saved at: {output_path}")


# Example test case
def test_generate_image():
    prompt = "A futuristic city skyline during sunset, with flying cars and neon lights"
    output_path = "generated_image.png"

    # Generate the image based on the test prompt
    generate_image(prompt, output_path)

    # Here, you could also add assertions if needed, such as checking if the image file was created
    assert output_path.endswith(".png"), f"Generated file is not a PNG: {output_path}"


if __name__ == "__main__":
    test_generate_image()
