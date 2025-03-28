# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from PIL import Image
from loguru import logger
from tqdm.auto import tqdm
import os
import string
import time
import random

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from models.utility_functions import (
    disable_persistent_kernel_cache,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.stable_diffusion.sd_pndm_scheduler import TtPNDMScheduler
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_unet_2d_condition_model_new_conv import (
    UNet2DConditionModel as UNet2D,
)


def constant_prop_time_embeddings(timesteps, sample, time_proj):
    timesteps = timesteps[None]
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = time_proj(timesteps)
    return t_emb


def tt_guide(noise_pred, guidance_scale):  # will return latents
    noise_pred_uncond = noise_pred[:1, :, :, :]
    noise_pred_text = ttnn.slice(
        noise_pred,
        [1, 0, 0, 0],
        [
            noise_pred.shape[0],
            noise_pred.shape[1],
            noise_pred.shape[2],
            noise_pred.shape[3],
        ],
    )
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


# Global variables for the Stable Diffusion model pipeline
model_pipeline = None
trace_id = None
op_event = None
ttnn_latents = None
text_embeddings_tensor = None


def create_model_pipeline(device, num_inference_steps, image_size=(256, 256)):
    disable_persistent_kernel_cache()
    device.enable_program_cache()

    # Until di/dt issues are resolved
    os.environ["SLOW_MATMULS"] = "1"
    assert (
        num_inference_steps >= 4
    ), f"PNDMScheduler only supports num_inference_steps >= 4. Found num_inference_steps={num_inference_steps}"

    height, width = image_size

    torch_device = "cpu"
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.to(torch_device)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    # 4. load the K-LMS scheduler with some fitting parameters.
    ttnn_scheduler = TtPNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        skip_prk_steps=True,
        steps_offset=1,
        device=device,
    )

    text_encoder.to(torch_device)
    unet.to(torch_device)

    config = unet.config
    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    input_height = 64
    input_width = 64
    reader_patterns_cache = {} if height == 512 and width == 512 else None
    model = UNet2D(device, parameters, 2, input_height, input_width, reader_patterns_cache)

    guidance_scale = 7.5  # Scale for classifier-free guidance
    random_seed = random.randrange(200) + 2
    # generator = torch.manual_seed(174)  # 10233 Seed generator to create the inital latent noise
    generator = torch.manual_seed(random_seed)
    batch_size = 1

    # Initial random noise
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // vae_scale_factor, width // vae_scale_factor),
        generator=generator,
    )
    latents = latents.to(torch_device)

    ttnn_scheduler.set_timesteps(num_inference_steps)

    latents = latents * ttnn_scheduler.init_noise_sigma
    rand_latents = torch.tensor(latents)
    rand_latents = ttnn.from_torch(rand_latents, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_latent_model_input = ttnn.concat([rand_latents, rand_latents], dim=0)
    _tlist = []
    for t in ttnn_scheduler.timesteps:
        _t = constant_prop_time_embeddings(t, ttnn_latent_model_input, unet.time_proj)
        _t = _t.unsqueeze(0).unsqueeze(0)
        _t = _t.permute(2, 0, 1, 3)  # pre-permute temb
        _t = ttnn.from_torch(_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _tlist.append(_t)

    time_step = ttnn_scheduler.timesteps.tolist()

    def capture_unet_trace():
        text_embeddings_shape = [2, 77, 768]
        rand_text_embeddings = torch.randn(text_embeddings_shape)
        rand_text_embeddings = torch.nn.functional.pad(rand_text_embeddings, (0, 0, 0, 19))
        rand_text_embeddings = ttnn.from_torch(rand_text_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        text_embeddings_tensor = ttnn.allocate_tensor_on_device(
            rand_text_embeddings.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
        )

        def model_forward(text_embeddings_tensor):
            ttnn_latent_model_input = ttnn.concat([rand_latents, rand_latents], dim=0)
            for index in tqdm(range(len(time_step))):
                _t = _tlist[index]
                t = time_step[index]
                ttnn_output = model(
                    ttnn_latent_model_input,
                    timestep=_t,
                    encoder_hidden_states=text_embeddings_tensor,
                    class_labels=None,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                    return_dict=True,
                    config=config,
                )
                noise_pred = tt_guide(ttnn_output, guidance_scale)
                ttnn_latents_output = ttnn_scheduler.step(
                    noise_pred, t, ttnn_latents_output if index > 0 else rand_latents
                ).prev_sample
                if index < len(time_step) - 1:
                    ttnn_latent_model_input = ttnn.concat([ttnn_latents_output, ttnn_latents_output], dim=0)

            return ttnn_latent_model_input

        # COMPILE
        ttnn_scheduler.set_timesteps(num_inference_steps)
        op_event = ttnn.record_event(device, 0)
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(rand_text_embeddings, text_embeddings_tensor, cq_id=1)
        write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, write_event)
        output = model_forward(text_embeddings_tensor)
        ttnn.synchronize_device(device)

        # CAPTURE
        output.deallocate()
        ttnn_scheduler.set_timesteps(num_inference_steps)
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(rand_text_embeddings, text_embeddings_tensor, cq_id=1)
        write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, write_event)

        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        output = model_forward(text_embeddings_tensor)
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        ttnn.synchronize_device(device)

        return trace_id, op_event, output, text_embeddings_tensor

    # Function to generate an image from the given prompt
    def _model_pipeline(input_prompt, trace_id, op_event, output, text_embeddings_tensor):
        start = time.time()

        ttnn_scheduler.set_timesteps(num_inference_steps)

        experiment_name = f"interactive_{height}x{width}"
        logger.info(f"input prompt : {input_prompt}")
        batch_size = len(input_prompt)
        assert batch_size == 1

        ## First, we get the text_embeddings for the prompt. These embeddings will be used to condition the UNet model.
        # Tokenizer and Text Encoder
        text_input = tokenizer(
            input_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

        # For classifier-free guidance, we need to do two forward passes: one with the conditioned input (text_embeddings),
        # and another with the unconditional embeddings (uncond_embeddings).
        # In practice, we can concatenate both into a single batch to avoid doing two forward passes.
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        ttnn_text_embeddings = torch.nn.functional.pad(text_embeddings, (0, 0, 0, 19))
        ttnn_text_embeddings = ttnn.from_torch(ttnn_text_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        print("Executing trace...")
        trace_start = time.time()
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(ttnn_text_embeddings, text_embeddings_tensor, cq_id=1)
        write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, write_event)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        host_ttnn_latents = output.cpu(blocking=True)
        trace_duration = time.time() - trace_start
        print(f"Trace done! Time 51 iteration of Unet + scheduler on device took: {trace_duration}")

        latents = ttnn.to_torch(host_ttnn_latents).to(torch.float32)

        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images][0]
        # Generate a random file name for the image
        random_filename = "".join(random.choices(string.ascii_lowercase + string.digits, k=10)) + ".png"
        image_path = os.path.join("generated_images", random_filename)
        pil_images.save(image_path)

        total_duration = time.time() - start
        print(f"Image generated! Total time: {total_duration}")

        return image_path

    # warmup model pipeline
    global trace_id
    global ttnn_latents
    global text_embeddings_tensor
    global op_event
    trace_id, op_event, ttnn_latents, text_embeddings_tensor = capture_unet_trace()

    global model_pipeline
    model_pipeline = _model_pipeline


def warmup_model():
    # create device, these constants are specific to n150 & n300
    device_id = 0
    device_params = {"l1_small_size": 32768, "trace_region_size": 806191104, "num_command_queues": 2}
    dispatch_core_type = ttnn.device.DispatchCoreType.WORKER
    if ("WH_ARCH_YAML" in os.environ) and os.environ["WH_ARCH_YAML"] == "wormhole_b0_80_arch_eth_dispatch.yaml":
        dispatch_core_type = ttnn.device.DispatchCoreType.ETH
    dispatch_core_axis = ttnn.DispatchCoreAxis.COL if ttnn.get_arch_name() == "blackhole" else ttnn.DispatchCoreAxis.ROW
    dispatch_core_config = ttnn.DispatchCoreConfig(dispatch_core_type, dispatch_core_axis)
    device_params["dispatch_core_config"] = dispatch_core_config
    device = ttnn.CreateDevice(device_id=device_id, **device_params)
    device.enable_program_cache()
    num_inference_steps = 50
    image_size = (512, 512)
    create_model_pipeline(device, num_inference_steps, image_size)


def generate_image_from_prompt(prompt):
    return model_pipeline([prompt], trace_id, op_event, ttnn_latents, text_embeddings_tensor)
