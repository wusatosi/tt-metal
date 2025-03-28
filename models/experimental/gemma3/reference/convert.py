# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Convert Qwen2.5-VL model state dict from flat to nested format.
The nested format is used by the functional implementations in functional.py.
"""

import os
import torch
from collections import defaultdict
from transformers import Gemma3ForConditionalGeneration, Gemma3TextModel, AutoConfig, modeling_utils

MODEL_ID = "google/gemma-3-27b-it"


def print_dict_structure(d, prefix="", max_tensor_info=10, file=None):
    """Print the structure of a nested dictionary, including tensor shapes.

    Args:
        d: The dictionary to print
        prefix: Prefix for indentation
        max_tensor_info: Maximum number of tensor dimensions to show per level
    """
    tensor_count = 0
    for k, v in sorted(d.items()):
        # Skip layers other than layer 0
        if k.isdigit() and k != "0":
            continue

        def _print(out_str):
            print(out_str)
            if file is not None:
                file.write(out_str + "\n")

        if isinstance(v, (dict, defaultdict)):
            _print(f"{prefix}{k}/")
            print_dict_structure(v, prefix + "  ", max_tensor_info, file)
        elif isinstance(v, (torch.Tensor, torch.nn.Parameter)):
            tensor_count += 1
            if tensor_count <= max_tensor_info:
                dtype_str = str(v.dtype).replace("torch.", "")
                _print(f"{prefix}{k}: {tuple(v.shape)} ({dtype_str})")
            elif tensor_count == max_tensor_info + 1:
                _print(f"{prefix}... and more tensors ...")


def convert_state_dict(flat_dict):
    """Convert a flat state dict to nested format.

    Example:
        Input: {'model.layers.0.self_attn.q_proj.weight': tensor(...)}
        Output: {'model': {'layers': [{'self_attn': {'q_proj': {'weight': tensor(...)}}}]}}
    """
    nested = {}

    for key, value in flat_dict.items():
        # Skip if not a tensor or buffer
        if not isinstance(value, (torch.Tensor, torch.nn.Parameter)):
            continue

        # Split the key into parts
        parts = key.split(".")

        # Traverse the nested dict, creating the structure
        current = nested
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the leaf value
        current[parts[-1]] = value

    return nested


def load_only_attention():
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file as safe_load_file
    import json

    model_dir = snapshot_download(MODEL_ID)

    # Load the weight map from the model's safetensors index file
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_file, "r") as f:
        index_data = json.load(f)

    # Filter the weight map for self-attention layers
    filters = ["language_model", "layers.0.self_attn."]
    weight_map = {k: v for k, v in index_data["weight_map"].items() if all(f in k for f in filters)}

    # Add model_dir in front of each file path
    weight_map = {k: os.path.join(model_dir, v) for k, v in weight_map.items()}

    # Identify all safetensor files needed
    safetensor_files = set(weight_map.values())

    # Load weights from safetensor files
    loaded_tensors = {}
    for file in safetensor_files:
        loaded_tensors.update(safe_load_file(file))  # Loads all tensors in this file

    # Filter tensors related only to self-attention
    attention_state_dict = {k: v for k, v in loaded_tensors.items() if all(f in k for f in filters)}
    # Remove "language_model." prefix from dict keys
    attention_state_dict = {
        k.replace("language_model.model.layers.0.self_attn.", ""): v for k, v in attention_state_dict.items()
    }

    # config_file = os.path.join(model_dir, "config.json")
    config = AutoConfig.from_pretrained(MODEL_ID)

    with modeling_utils.no_init_weights():
        text_model = Gemma3TextModel(config.text_config)
        attention = text_model.layers[0].self_attn
    missing_keys, unexpected_keys = attention.load_state_dict(attention_state_dict, strict=False)

    attention_flat_dict = attention.state_dict()
    attention_nested = convert_state_dict(attention_flat_dict)
    print(attention_nested)


def main():
    load_only_attention()
    return

    # Create output directory
    os.makedirs("weights", exist_ok=True)

    print("Loading model...")
    model = Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")

    print("\nConverting state dict...")
    language_flat_dict = model.language_model.state_dict()
    language_nested = convert_state_dict(language_flat_dict)

    print("\nFull model structure:")
    with open("language_model.txt", "w") as f:
        print_dict_structure(language_nested, file=f)

    print("\nExtracting vision components...")
    vision_flat_dict = model.vision_tower.state_dict()
    vision_nested = convert_state_dict(vision_flat_dict)

    print("\nVision components structure:")
    with open("vision_model.txt", "w") as f:
        print_dict_structure(vision_nested, file=f)

    print("\nSaving converted weights...")
    # torch.save(vision_nested, "weights/vision_weights.pt")
    print("Done! Saved to weights/vision_weights.pt")


if __name__ == "__main__":
    main()
