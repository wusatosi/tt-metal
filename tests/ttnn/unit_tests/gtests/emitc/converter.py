# save_tensor.py
import torch
import json

from loguru import logger

# Load tensors
I_original = torch.load("tests/ttnn/unit_tests/gtests/emitc/12.pt")
weight_original = torch.load("tests/ttnn/unit_tests/gtests/emitc/9.pt")

# Print shapes
print(f"I_original shape: {I_original.shape}")
print(f"weight_original shape: {weight_original.shape}")

# Save raw binary data
I_original.to(dtype=torch.float32).numpy().astype("float32").tofile("tests/ttnn/unit_tests/gtests/emitc/I_original.bin")
weight_original.to(dtype=torch.float32).numpy().astype("float32").tofile(
    "tests/ttnn/unit_tests/gtests/emitc/weight_original.bin"
)

# Save metadata as JSON
meta_I = {"shape": list(I_original.shape), "dtype": "float32"}
meta_weight = {"shape": list(weight_original.shape), "dtype": "float32"}

with open("tests/ttnn/unit_tests/gtests/emitc/I_original.json", "w") as f:
    json.dump(meta_I, f)

with open("tests/ttnn/unit_tests/gtests/emitc/weight_original.json", "w") as f:
    json.dump(meta_weight, f)

print("Tensors saved as binary files and metadata saved as JSON files")
