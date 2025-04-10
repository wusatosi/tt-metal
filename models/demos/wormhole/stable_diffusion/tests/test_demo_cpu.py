from diffusers import StableDiffusionPipeline
import torch
import torch
from PIL import Image
from torchvision import transforms
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from models.demos.wormhole.stable_diffusion.sd_pndm_scheduler import TtPNDMScheduler
from diffusers import DiffusionPipeline


def preprocess_image(image: Image, height: int, width: int):
    # Resize the image to match the model's input dimensions
    transform = transforms.Compose(
        [
            transforms.Resize((height, width)),  # Resize the image to (height, width)
            transforms.ToTensor(),  # Convert the image to a tensor (scaled [0, 1])
            transforms.Normalize([0.5], [0.5]),  # Normalize the image in the range [-1, 1]
        ]
    )

    # Apply the transformation
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def generate_image(device, prompt: str, output_path: str):
    # Load pre-trained Stable Diffusion model pipeline
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    ttnn_scheduler = TtPNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        skip_prk_steps=True,
        steps_offset=1,
        device=device,
    )

    input_height = 512
    input_width = 512
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    guidance_scale = 7.5  # Scale for classifier-free guidance
    generator = torch.manual_seed(174)  # Seed for latent generation
    batch_size = 1

    latents = torch.randn(
        (batch_size, unet.config.in_channels, input_height // vae_scale_factor, input_width // vae_scale_factor),
        generator=generator,
        device=pipe.device,
    )

    latents = latents * ttnn_scheduler.init_noise_sigma

    pipe.to("cpu")

    # Generate an image based on the input prompt
    with torch.no_grad():  # No need to track gradients during inference
        image = pipe(prompt, num_inference_steps=50, latents=latents).images[0]

    # Save the generated image to the specified output path
    image.save(output_path)
    print(f"Image generated and saved at: {output_path}")


# Example test case
def test_generate_image(device):
    prompt = "a comic potrait of a female necromamcer with big and cute eyes, fine - face, realistic shaded perfect face, fine details. night setting. very anime style. realistic shaded lighting poster by ilya kuvshinov katsuhiro, magali villeneuve, artgerm, jeremy lipkin and michael garmash, rob rey and kentaro miura style, trending on art station"
    output_path = "generated_image3.png"

    # Generate the image based on the test prompt
    generate_image(device, prompt, output_path)

    # Here, you could also add assertions if needed, such as checking if the image file was created
    assert output_path.endswith(".png"), f"Generated file is not a PNG: {output_path}"


def test_sdxl():
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, variant="fp16"
    )


if __name__ == "__main__":
    test_generate_image()
