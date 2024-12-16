import os
import torch
from safetensors.torch import load_file as safetensors_load_file
from tqdm import tqdm
import json
from pathlib import Path
from loguru import logger


# TODO Update function for large models: For 1 layer tests we only want to load 1 checkpoint file, instead of all.
def load_hf_state_dict(ckpt_dir):
    # First check if index file exists
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        # Multi-file case: Read the index file and load all referenced safetensor files
        with open(index_path, "r") as f:
            index_data = json.load(f)

        # Retrieve the weight file names from the index JSON
        weight_map = index_data["weight_map"]
        safetensor_files = set(weight_map.values())

        # Read each safetensors file mentioned in the index
        loaded_weights = {}
        for file in safetensor_files:
            safetensor_path = os.path.join(ckpt_dir, file)
            weights = safetensors_load_file(safetensor_path)
            loaded_weights.update(weights)  # Merge weights into a single dictionary
    else:
        # Single-file case: Load the single model.safetensors file
        safetensor_path = os.path.join(ckpt_dir, "model.safetensors")
        if not os.path.exists(safetensor_path):
            raise FileNotFoundError(f"Neither model.safetensors.index.json nor model.safetensors found in {ckpt_dir}")
        loaded_weights = safetensors_load_file(safetensor_path)

    if not "lm_head.weight" in loaded_weights:
        # Assume tied to the embeddings if not present
        loaded_weights["lm_head.weight"] = loaded_weights["model.embed_tokens.weight"]

    return loaded_weights


def convert_hf_to_meta(state_dict):
    state_dict = convert_hf_qkv_to_meta_format(state_dict)
    state_dict = map_hf_to_meta_keys(state_dict)
    return state_dict


def map_hf_to_meta_keys(loaded_weights):
    hf_to_meta = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
        "model.layers.{layer}.input_layernorm.weight": "layers.{layer}.attention_norm.weight",
        "model.layers.{layer}.post_attention_layernorm.weight": "layers.{layer}.ffn_norm.weight",
        "model.layers.{layer}.self_attn.q_proj.weight": "layers.{layer}.attention.wq.weight",
        "model.layers.{layer}.self_attn.k_proj.weight": "layers.{layer}.attention.wk.weight",
        "model.layers.{layer}.self_attn.v_proj.weight": "layers.{layer}.attention.wv.weight",
        "model.layers.{layer}.self_attn.o_proj.weight": "layers.{layer}.attention.wo.weight",
        "model.layers.{layer}.self_attn.q_proj.bias": "layers.{layer}.attention.wq.bias",
        "model.layers.{layer}.self_attn.k_proj.bias": "layers.{layer}.attention.wk.bias",
        "model.layers.{layer}.self_attn.v_proj.bias": "layers.{layer}.attention.wv.bias",
        "model.layers.{layer}.mlp.gate_proj.weight": "layers.{layer}.feed_forward.w1.weight",
        "model.layers.{layer}.mlp.up_proj.weight": "layers.{layer}.feed_forward.w3.weight",
        "model.layers.{layer}.mlp.down_proj.weight": "layers.{layer}.feed_forward.w2.weight",
        "lm_head.weight": "output.weight",
    }

    meta_state_dict = {}
    for key, tensor in loaded_weights.items():
        if key in hf_to_meta:
            # Direct match for top-level keys
            meta_state_dict[hf_to_meta[key]] = tensor
        elif "model.layers." in key:
            # Extract layer number and form a template key
            parts = key.split(".")
            layer_num = parts[2]  # e.g. "0" in "model.layers.0.input_layernorm.weight"
            template_key = "model.layers.{layer}." + ".".join(parts[3:])
            if template_key in hf_to_meta:
                meta_state_dict[hf_to_meta[template_key].format(layer=layer_num)] = tensor

    return meta_state_dict


def load_meta_state_dict(ckpt_dir, n_layers=None, start_layer_idx=0):
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
    is_chunked = "layers_" in str(checkpoints[0])
    if is_chunked:
        checkpoint = load_chunked_checkpoints(checkpoints, n_layers, start_layer_idx)
    else:
        checkpoint = load_sharded_checkpoints(checkpoints, n_layers)

    return checkpoint


def load_chunked_checkpoints(checkpoints, n_layers, start_layer_idx):
    checkpoint = {}

    (f"Loading {len(checkpoints)} checkpoint files")
    for ckpt in tqdm(checkpoints):
        if n_layers:
            # Layer range is in the file name, like layers_start-end.pth
            layer_range = ckpt.stem.split("_")[1]
            start_layer, end_layer = map(int, layer_range.split("-"))
            if start_layer > n_layers + start_layer_idx:
                continue
            if end_layer < start_layer_idx:
                continue

        loaded_ckpt = torch.load(ckpt, map_location="cpu")
        checkpoint.update(loaded_ckpt)
    return checkpoint


def load_sharded_checkpoints(checkpoints, n_layers):
    checkpoint = {}
    logger.info(f"Loading {len(checkpoints)} checkpoint files")
    for ckpt in tqdm(checkpoints):
        loaded_ckpt = torch.load(ckpt, map_location="cpu")
        for (
            key,
            value,
        ) in loaded_ckpt.items():
            if "layers." in key:
                layer_num = int(key.split("layers.")[1].split(".")[0])
                if n_layers and layer_num >= n_layers:
                    continue
            if key in checkpoint:
                checkpoint[key] += [value]
            else:
                checkpoint[key] = [value]
        del loaded_ckpt

    # concat checkpoint values
    for key, value in checkpoint.items():
        if len(value) == 1 or "norm" in key:
            checkpoint[key] = value[0]
        else:
            if key == "tok_embeddings.weight" or key == "output.weight":
                assert value[0].shape[1] == 8192  # FIXME: do we need this hardcoded shape?
                # Concatenate along dimension 0 for llama3 token embeddings weight and lm head
                checkpoint[key] = torch.cat(value, dim=0)
            else:
                # cat_dim is index of the smallest dimension in value[0].shape
                cat_dim = torch.argmin(torch.tensor(value[0].shape))
                checkpoint[key] = torch.cat(value, dim=cat_dim)

    return checkpoint


def convert_hf_qkv_to_meta_format(loaded_weights):
    """Convert HuggingFace QKV weights to Meta format for RoPE compatibility.
    For each attention layer's Q and K weights/biases:
    - First half and second half dimensions are interleaved
    - V weights/biases remain unchanged
    """
    converted_weights = {}
    for key, tensor in loaded_weights.items():
        if key.endswith(("q_proj.weight", "q_proj.bias", "k_proj.weight", "k_proj.bias")):
            # Get head dimension size
            head_dim = tensor.shape[-1]
            half_dim = head_dim // 2

            # Split into halves
            first_half = tensor[..., :half_dim]
            second_half = tensor[..., half_dim:]

            # Interleave the halves
            converted = torch.empty_like(tensor)
            converted[..., 0::2] = first_half
            converted[..., 1::2] = second_half

            converted_weights[key] = converted
        else:
            # Keep all other weights unchanged
            converted_weights[key] = tensor

    return converted_weights


def convert_meta_to_hf(state_dict):
    state_dict = convert_meta_qkv_to_hf_format(state_dict)
    state_dict = map_meta_to_hf_keys(state_dict)
    return state_dict


def map_meta_to_hf_keys(loaded_weights):
    # Define mappings at each level of the hierarchy
    meta_to_hf_mappings = {
        # Top level
        "tok_embeddings.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
        # Layer level
        "attention_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        # Attention module
        "attention.wq.weight": "self_attn.q_proj.weight",
        "attention.wk.weight": "self_attn.k_proj.weight",
        "attention.wv.weight": "self_attn.v_proj.weight",
        "attention.wo.weight": "self_attn.o_proj.weight",
        "attention.wq.bias": "self_attn.q_proj.bias",
        "attention.wk.bias": "self_attn.k_proj.bias",
        "attention.wv.bias": "self_attn.v_proj.bias",
        # Feed forward module
        "feed_forward.w1.weight": "mlp.gate_proj.weight",
        "feed_forward.w3.weight": "mlp.up_proj.weight",
        "feed_forward.w2.weight": "mlp.down_proj.weight",
        # Direct mappings for when we get just the final components
        "w1.weight": "gate_proj.weight",
        "w2.weight": "down_proj.weight",
        "w3.weight": "up_proj.weight",
        "wq.weight": "q_proj.weight",
        "wk.weight": "k_proj.weight",
        "wv.weight": "v_proj.weight",
        "wo.weight": "o_proj.weight",
        "wq.bias": "q_proj.bias",
        "wk.bias": "k_proj.bias",
        "wv.bias": "v_proj.bias",
        # Host embeddings
        "emb.weight": "weight",
    }

    hf_state_dict = {}
    for key, tensor in loaded_weights.items():
        # Handle full model paths with layer numbers
        if "layers." in key:
            parts = key.split(".")
            layer_num = parts[1]
            remainder = ".".join(parts[2:])
            if remainder in meta_to_hf_mappings:
                new_key = f"model.layers.{layer_num}.{meta_to_hf_mappings[remainder]}"
                hf_state_dict[new_key] = tensor
            continue

        # Try exact matches first
        if key in meta_to_hf_mappings:
            hf_state_dict[meta_to_hf_mappings[key]] = tensor
            continue

        # For submodule state dicts, try matching the end of the key
        matched = False
        for meta_pattern, hf_pattern in meta_to_hf_mappings.items():
            if key.endswith(meta_pattern):
                # Replace only the matching part at the end
                prefix = key[: -len(meta_pattern)]
                new_key = prefix + hf_pattern
                hf_state_dict[new_key] = tensor
                matched = True
                break

        # If no mapping found, keep the original key
        if not matched:
            hf_state_dict[key] = tensor

    return hf_state_dict


def convert_meta_qkv_to_hf_format(loaded_weights):
    """Convert Meta QKV weights back to HuggingFace format.
    For each attention layer's Q and K weights/biases:
    - De-interleave the dimensions back to concatenated halves
    - V weights/biases remain unchanged
    """
    converted_weights = {}
    for key, tensor in loaded_weights.items():
        if any(pattern in key for pattern in ["wq.", "wk."]):  # Matches wq.weight, wq.bias, wk.weight, wk.bias
            # Get head dimension size
            head_dim = tensor.shape[-1]

            # De-interleave the dimensions
            even_indices = tensor[..., 0::2]  # First half
            odd_indices = tensor[..., 1::2]  # Second half

            # Concatenate back to HF format
            converted = torch.cat([even_indices, odd_indices], dim=-1)
            converted_weights[key] = converted
        else:
            # Keep all other weights unchanged
            converted_weights[key] = tensor

    return converted_weights
