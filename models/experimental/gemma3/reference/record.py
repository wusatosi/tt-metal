# pip install timm
# pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

import json
import logging
import re
import os
import torch
from datetime import datetime
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import inspect

MODEL_ID = "google/gemma-3-27b-it"
RECORDS_DIR = "module_io_data"
# One timestamp per run
RECORDS_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def get_script_dir():
    """Get the directory where this script is located."""
    return os.path.dirname(os.path.abspath(__file__))


def config_to_dict(config):
    """Convert a config object to a serializable dictionary."""
    if hasattr(config, "to_dict"):
        return config.to_dict()
    elif hasattr(config, "__dict__"):
        return {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
    return str(config)


def save_tensor_data(tensor, directory, name):
    """Save a tensor to a .pt file."""
    if isinstance(tensor, torch.Tensor):
        path = os.path.join(directory, f"{name}.pt")
        torch.save(tensor.detach().cpu(), path)
        return {"type": "tensor", "shape": list(tensor.shape), "dtype": str(tensor.dtype), "path": path}
    elif isinstance(tensor, (list, tuple)):
        return [save_tensor_data(t, directory, f"{name}_{i}") for i, t in enumerate(tensor)]
    return str(tensor)


def record_hook(module, args, kwargs, output):
    """
    A hook for instrumenting a module's forward method.
    Records the module's settings, inputs, and outputs in a structured directory format:
    module_io_data/
        modulename_YYYYMMDD_HHMMSS/
            metadata.json      # Contains settings and non-tensor data
            inputs/           # Directory for input tensors
                input_0.pt    # Individual tensor files
                input_1.pt
            outputs/          # Directory for output tensors
                output_0.pt
    """

    call_functions = []

    for frame in inspect.stack():
        if frame.function == "forward":  # Only check 'forward' functions
            frame_self = frame.frame.f_locals.get("self")  # Get 'self' if it exists
            if frame_self and hasattr(frame_self, "__class__"):  # Ensure it's a class instance
                if frame.code_context:
                    line = frame.code_context[0].strip()
                    match = re.search(r"self\.(\w+)\s*\(", line)  # Match method calls on `self`
                    if match:
                        call_functions.append(match.group(1))

    module_name = type(module).__name__
    run_dir = os.path.join(
        get_script_dir(), RECORDS_DIR, f"{RECORDS_TIMESTAMP}", f"{module_name}.{'.'.join(reversed(call_functions))}"
    )
    if os.path.exists(run_dir):
        return
    os.makedirs(run_dir, exist_ok=True)

    # Create subdirectories for inputs and outputs
    inputs_dir = os.path.join(run_dir, "inputs")
    outputs_dir = os.path.join(run_dir, "outputs")
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # Record settings
    settings = {}
    if hasattr(module, "config"):
        try:
            settings = config_to_dict(module.config)
        except Exception as e:
            settings = str(e)
    else:
        for attr in dir(module):
            if not attr.startswith("_") and not callable(getattr(module, attr)):
                try:
                    val = getattr(module, attr)
                    if isinstance(val, torch.Tensor):
                        settings[attr] = save_tensor_data(val, run_dir, f"setting_{attr}")
                    else:
                        settings[attr] = str(val)
                except Exception as e:
                    settings[attr] = str(e)

    # Save inputs
    inputs = {
        "args": [save_tensor_data(a, inputs_dir, f"arg_{i}") for i, a in enumerate(args)],
        "kwargs": {k: save_tensor_data(v, inputs_dir, f"kwarg_{k}") for k, v in kwargs.items()},
    }

    # Save outputs
    if isinstance(output, torch.Tensor):
        output_data = save_tensor_data(output, outputs_dir, "output")
    elif isinstance(output, (list, tuple)):
        output_data = [save_tensor_data(o, outputs_dir, f"output_{i}") for i, o in enumerate(output)]
    else:
        output_data = str(output)

    # Save metadata
    metadata = {
        "module": module_name,
        "class": module.__class__.__name__,
        "timestamp": datetime.now().isoformat(),
        "settings": settings,
        "inputs": inputs,
        "outputs": output_data,
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def record():
    # Load model
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    # Record only 1 layer for unit tests
    model.language_model.model.layers = model.language_model.model.layers[:1]

    # Register hooks on all submodules
    model.apply(lambda m: m.register_forward_hook(record_hook, with_kwargs=True))

    processor = AutoProcessor.from_pretrained(MODEL_ID, padding_side="left")

    # Both text and image inputs
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                },
                {"type": "text", "text": "Describe this image in detail."},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Execute the model
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=1)


if __name__ == "__main__":
    # Run the main function
    record()
