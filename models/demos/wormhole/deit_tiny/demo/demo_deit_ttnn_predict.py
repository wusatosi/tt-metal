# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import transformers
from transformers import AutoFeatureExtractor, ViTForImageClassification
from loguru import logger
import torch
import torch.nn as nn

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from models.demos.deit_tiny.tt import ttnn_optimized_sharded_deit_wh
from models.utility_functions import torch2tt_tensor, is_blackhole
from models.demos.wormhole.deit_tiny.demo.deit_helper_funcs import get_batch_cifar, get_cifar10_label_dict, get_data_loader_cifar10

import ast
from torchvision import transforms
from PIL import Image
import os
from PIL import ImageDraw, ImageFont


@pytest.mark.skipif(is_blackhole(), reason="Unsupported on BH")
def test_deit(device):
    torch.manual_seed(0)

    model_name = "facebook/deit-tiny-patch16-224"
    batch_size = 1
    sequence_size = 224
    iterations = 100

    config = transformers.DeiTConfig.from_pretrained(model_name)
    config.num_hidden_layers = 12
    model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config)
    model.classifier = nn.Linear(192, 10, bias=True)
    model.load_state_dict(torch.load("models/demos/wormhole/deit_tiny/demo/deit_tiny_patch16_224_trained_statedict.pth"), strict=True)
    config = ttnn_optimized_sharded_deit_wh.update_model_config(config, batch_size)
    image_processor = AutoFeatureExtractor.from_pretrained(model_name)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_sharded_deit_wh.custom_preprocessor,
    )

    # cls_token & position embeddings expand to batch_size
    # TODO: pass batch_size to preprocess_model_parameters
    model_state_dict = model.state_dict()
    torch_cls_token = model_state_dict["vit.embeddings.cls_token"]
    torch_distillation_token = torch.nn.Parameter(torch.zeros(1, 1, 192))
    torch_position_embeddings = model_state_dict["vit.embeddings.position_embeddings"]
    if batch_size > 1:
        torch_cls_token = torch.nn.Parameter(torch_cls_token.expand(batch_size, -1, -1))
        torch_distillation_token = torch.nn.Parameter(torch_distillation_token.expand(batch_size, -1, -1))
        torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings.expand(batch_size, -1, -1))
    else:
        torch_cls_token = torch.nn.Parameter(torch_cls_token)
        torch_distillation_token = torch.nn.Parameter(torch_distillation_token)
        torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings)
    cls_token = ttnn.from_torch(torch_cls_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    distillation_token = ttnn.from_torch(
        torch_distillation_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    position_embeddings = ttnn.from_torch(
        torch_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )

    torch_attention_mask = torch.ones(config.num_hidden_layers, sequence_size, dtype=torch.float32)
    if torch_attention_mask is not None:
        head_masks = [
            ttnn.from_torch(
                torch_attention_mask[index].reshape(1, 1, 1, sequence_size).expand(batch_size, -1, -1, -1),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            for index in range(config.num_hidden_layers)
        ]
    else:
        head_masks = [None for _ in range(config.num_hidden_layers)]

    # IMAGENET INFERENCE
    #####################
    cifar_label_dict = get_cifar10_label_dict()
    
    def load_and_transform_images(image_dir, image_size=(224, 224)):
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        images = []
        
        original_images = []
        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            image = Image.open(image_path).convert("RGB")
            transformed_image = transform(image).unsqueeze(0)  # Add batch dimension
            images.append(transformed_image)
            original_images.append(image)
            
        return images, original_images

    # Load and transform images from images directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, "images")
    transformed_images = load_and_transform_images(image_dir)

    # Use the first transformed image as input
    data_loader = [(transformed_images[0], 0)]  # Dummy label 0 for testing


    correct = 0
    for iter in range(len(transformed_images[0])):
        torch_pixel_values = transformed_images[0][iter]
        original_image = transformed_images[1][iter]

        torch_pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
        torch_pixel_values = torch.nn.functional.pad(torch_pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
        batch_size, img_h, img_w, img_c = torch_pixel_values.shape  # permuted input NHWC
        patch_size = 16
        torch_pixel_values = torch_pixel_values.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)
        N, H, W, C = torch_pixel_values.shape
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(batch_size - 1, 3),
                ),
            }
        )
        n_cores = batch_size * 3
        shard_spec = ttnn.ShardSpec(shard_grid, [N * H * W // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR)

        output = None
        pixel_values = torch2tt_tensor(
            torch_pixel_values,
            device,
            ttnn.ROW_MAJOR_LAYOUT,
            tt_memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                shard_spec,
            ),
            tt_dtype=ttnn.bfloat16,
        )

        output = ttnn_optimized_sharded_deit_wh.deit(
            config,
            pixel_values,
            head_masks,
            cls_token,
            distillation_token,
            position_embeddings,
            parameters=parameters,
        )
        output = ttnn.to_torch(output)
        prediction = output[:, 0, :10].argmax(dim=-1)
        predict_label = cifar_label_dict[prediction[0].item()]
        logger.info(
                f"Iter: {iter} Sample: {iter+1} - Predicted Label: {predict_label}"
            )
        # Create results folder if it doesn't exist
        
        results_dir = os.path.join(script_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Draw the label on the original image
        draw = ImageDraw.Draw(original_image)
        font = ImageFont.load_default()  # Use default font
        text_position = (original_image.width - 100, 10)  # Top-right corner
        font = ImageFont.truetype(os.path.join(script_dir, "arial.ttf"), 20)  # Use a larger font size
        # Add a black rectangle as background for the text
        text_background_position = (text_position[0] - 10, text_position[1] - 5, text_position[0] + 100, text_position[1] + 25)
        draw.rectangle(text_background_position, fill="black")
        
        # Draw the text on top of the black background
        draw.text(text_position, predict_label, fill="white", font=font)

        # Save the image with the label
        output_image_path = os.path.join(results_dir, f"result_{iter+1}.png")
        logger.info(f"Saving labeled image to {output_image_path}")
        original_image.save(output_image_path)

        
            
            
        # del tt_output, tt_inputs, inputs, labels, predictions

    
    