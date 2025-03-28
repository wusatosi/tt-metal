# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
"""Tests for functional implementations of Gemma3 modules."""

import os
import json
import pytest
import torch
import torch.nn.functional as torch_F
from glob import glob
from scipy.stats import pearsonr
import numpy as np
import importlib
import sys
import re

from transformers import Gemma3ForConditionalGeneration, Gemma3TextModel, AutoConfig, modeling_utils
import functools
from typing import Optional, Tuple, List, Dict

MODEL_ID = "google/gemma-3-27b-it"


def get_script_dir():
    """Get the directory where this script is located."""
    return os.path.dirname(os.path.abspath(__file__))


# @pytest.fixture
# def weights():
#     """Load the converted vision weights."""
#     script_dir = get_script_dir()
#     weights_path = os.path.join(script_dir, "weights/vision_weights.pt")
#     if not os.path.exists(weights_path):
#         pytest.skip(f"Vision weights not found at {weights_path}. Run convert.py first.")
#     return torch.load(weights_path, weights_only=False)


# @pytest.fixture(params=["functional", "functional_ttnn"], ids=["torch", "ttnn"])
# def implementation(request, mesh_device):
#     """Fixture to run tests with both PyTorch and TTNN implementations."""
#     module_name = request.param

#     if module_name == "functional_ttnn":
#         try:
#             import ttnn
#         except ImportError:
#             pytest.skip("TTNN not available")

#     try:
#         module = importlib.import_module(f"models.demos.qwen25_vl.reference.{module_name}")

#         # Set the mesh_device for the TTNN implementation
#         if module_name == "functional_ttnn" and hasattr(module, "set_mesh_device"):
#             module.set_mesh_device(mesh_device)

#         return module
#     except ImportError:
#         pytest.skip(f"{module_name} implementation not available")


def get_nested_attr(obj, attr_path):
    """Access a nested attribute dynamically using a string path.

    Args:
        obj: The base object (e.g., a model instance).
        attr_path: A string representing the nested attribute (e.g., "layers[0].self_attn").

    Returns:
        The nested attribute value.
    """
    parts = attr_path.split(".")  # Split into parts

    def _getattr(obj, attr):
        """Handles both regular attributes and list indices."""
        if "[" in attr and "]" in attr:  # Check if it's a list index
            attr_name, index = attr[:-1].split("[")  # Extract name and index
            return getattr(obj, attr_name)[int(index)]  # Access list item
        return getattr(obj, attr)  # Regular attribute access

    return functools.reduce(_getattr, parts, obj)  # Traverse attributes


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


def load_weights(filter: str):
    """Load the weights for the filtered layer."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file as safe_load_file
    import json

    model_dir = snapshot_download(MODEL_ID)

    # Load the weight map from the model's safetensors index file
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_file, "r") as f:
        index_data = json.load(f)

    # Filter the weight map and add model_dir in front of each file path
    weight_map = {k: os.path.join(model_dir, v) for k, v in index_data["weight_map"].items() if filter in k}

    # Identify all safetensor files needed, remove duplicates
    safetensor_files = set(weight_map.values())

    # Load weights from safetensor files
    loaded_tensors = {}
    for file in safetensor_files:
        loaded_tensors.update(safe_load_file(file))

    # Filter tensors related and remove filter prefix
    filter_prefix = filter + "."
    filtered_state_dict = {k.replace(filter_prefix, ""): v for k, v in loaded_tensors.items() if filter in k}
    return convert_state_dict(filtered_state_dict)


def load_latest_run(module_name, filter):
    """Load the earliest recorded run for a given module."""
    script_dir = get_script_dir()
    pattern = os.path.join(script_dir, f"module_io_data/*")
    runs = sorted(glob(pattern))  # Default sort will put earliest run first
    latest_run = runs[-1]  # Take the last run

    filter_no_lists = re.sub(r"layers\.\d+\.", "", filter)

    module_run_dir = os.path.join(latest_run, f"{module_name}.{filter_no_lists}")
    if not os.path.exists(module_run_dir):
        pytest.skip(f"No recorded runs found for {module_name}")

    # Load metadata
    with open(os.path.join(module_run_dir, "metadata.json")) as f:
        metadata = json.load(f)

    # Load tensors
    def load_tensor_data(data):
        """Recursively load tensor data from saved files."""
        if data == "None":  # Handle string "None" in metadata
            return None
        if isinstance(data, dict):
            if data.get("type") == "tensor" and "path" in data:
                return torch.load(os.path.join(script_dir, data["path"]))
            return {k: load_tensor_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            loaded = [load_tensor_data(x) for x in data]
            # If all items are tensors and it's in kwargs['position_embeddings'],
            # convert to tuple as that's what the model expects
            if all(isinstance(x, torch.Tensor) for x in loaded):
                return tuple(loaded)
            # Try to convert a list of strings to integers if they all look like integers
            if all(isinstance(x, str) and x.isdigit() for x in loaded):
                return [int(x) for x in loaded]
            return loaded
        # Try to convert string to int if it represents a number
        if isinstance(data, str) and data.isdigit():
            return int(data)
        return data

    inputs = load_tensor_data(metadata["inputs"])
    outputs = load_tensor_data(metadata["outputs"])
    settings = metadata["settings"]  # Settings usually don't contain tensors

    return inputs, outputs, settings


def pearson_correlation(x, y):
    """Calculate Pearson correlation coefficient between two tensors."""
    x_flat = x.detach().float().cpu().numpy().flatten()
    y_flat = y.detach().float().cpu().numpy().flatten()
    return pearsonr(x_flat, y_flat)[0]


def gemma3_rms_norm(x: torch.Tensor, state_dict: Dict, eps: float = 1e-6):
    """Applies RMSNorm to the input tensor x.

    Args:
        x (torch.Tensor): Input tensor.
        weight (torch.Tensor): Learnable weight parameter (should match last dim of x).
        eps (float, optional): Small constant to avoid division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: Normalized output tensor.
    """
    norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    output = norm_x * (1.0 + state_dict["weight"].float())
    return output


def test_gemma3_rms_norm():
    """Test RMSNorm functional implementation."""
    # filter = "language_model.model.norm"
    filter = "language_model.model.layers.0.self_attn.q_norm"
    inputs, outputs, _ = load_latest_run("Gemma3RMSNorm", filter)

    # Extract inputs and state dict
    hidden_states = inputs["args"][0]
    weights = load_weights(filter)

    # Run functional implementation
    result = gemma3_rms_norm(hidden_states, weights)

    # Check correlation with recorded output
    pcc = pearson_correlation(result, outputs)
    assert pcc > 0.999, f"PCC {pcc} below threshold"
