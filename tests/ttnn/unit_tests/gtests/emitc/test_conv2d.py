import torch

import torch.nn.functional as F

from loguru import logger
import re

import torch
import pytest
import matplotlib.pyplot as plt


def load_ttnn_tensor_from_txt(filepath):
    """
    Parses a ttnn.Tensor(...) dump from a .txt file and returns it as a torch.Tensor.
    Assumes the text ends with a 'shape=Shape([...])' specifier.
    """
    with open(filepath, "r") as f:
        text = f.read()

    # Extract the shape
    shape_match = re.search(r"shape=Shape\(\[([\d,\s]+)\]\)", text)
    if not shape_match:
        raise ValueError("Could not find shape in the text.")
    shape = [int(dim) for dim in shape_match.group(1).split(",")]

    # Only parse the tensor values portion (cut off before 'shape=')
    data_text = text.split("shape=")[0]

    # Extract only numeric values from the tensor data portion
    numbers = re.findall(r"-?\d+\.\d+|-?\d+", data_text)
    numbers = [float(n) for n in numbers]

    # Verify total number of elements matches the shape
    expected_size = 1
    for dim in shape:
        expected_size *= dim
    if len(numbers) != expected_size:
        raise ValueError(f"Expected {expected_size} values, but got {len(numbers)}")

    # Convert and reshape
    tensor = torch.tensor(numbers, dtype=torch.float32).view(*shape)
    tensor = tensor.to(dtype=torch.bfloat16)

    return tensor


def compare_tensors(output, device_tensor, atol=1e-2):
    """
    Compare two tensors using absolute tolerance and Pearson correlation coefficient.

    Args:
        output (torch.Tensor): Reference tensor.
        device_tensor (torch.Tensor): Tensor to compare.
        atol (float): Absolute tolerance threshold.

    Returns:
        dict: Contains 'allclose', 'pcc', and 'max_abs_diff'.
    """
    if output.shape != device_tensor.shape:
        raise ValueError(f"Shape mismatch: {output.shape} vs {device_tensor.shape}")

    # Ensure both are float32 for comparison
    output = output.to(torch.float32)
    device_tensor = device_tensor.to(torch.float32)

    # Absolute tolerance check
    allclose = torch.allclose(output, device_tensor, atol=atol)

    # Pearson Correlation Coefficient
    output_flat = output.flatten()
    device_flat = device_tensor.flatten()
    pcc = torch.corrcoef(torch.stack([output_flat, device_flat]))[0, 1].item()

    # Max absolute difference
    max_abs_diff = torch.max(torch.abs(output - device_tensor)).item()

    return {"allclose": allclose, "pcc": pcc, "max_abs_diff": max_abs_diff}


def max_abs_diff_and_values(output, device_tensor):
    """
    Returns the max absolute difference, its index,
    and the tensor values at that index.

    Args:
        output (torch.Tensor)
        device_tensor (torch.Tensor)

    Returns:
        tuple: (max_diff_value (float), index (tuple), output_value (float), device_value (float))
    """
    output = output.to(torch.float32)
    device_tensor = device_tensor.to(torch.float32)
    diff = torch.abs(output - device_tensor)
    max_diff_value = diff.max().item()
    max_diff_index = torch.unravel_index(diff.argmax(), diff.shape)
    output_val = output[max_diff_index].item()
    device_val = device_tensor[max_diff_index].item()
    return max_diff_value, max_diff_index, output_val, device_val


def plot_diff_distribution(output, device_tensor, bins=100):
    """
    Plots the histogram of absolute differences between two tensors.

    Args:
        output (torch.Tensor)
        device_tensor (torch.Tensor)
        bins (int): Number of bins in the histogram
    """
    output = output.to(torch.float32)
    device_tensor = device_tensor.to(torch.float32)
    diff = torch.abs(output - device_tensor).flatten().cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.hist(diff, bins=bins, color="skyblue", edgecolor="black")
    plt.title("Distribution of Absolute Differences")
    plt.xlabel("Absolute difference")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("plot.png")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv(device):
    x = torch.load("Testing/12.pt")
    w = torch.load("Testing/9.pt")
    print(x.shape)
    print(w.shape)

    x = x.reshape(32, 224, 224, 3)  # [N, H, W, C]
    x = x.permute(0, 3, 1, 2)  # [N, C, H, W] => [32, 3, 224, 224]

    # # Perform convolution
    output = F.conv2d(x, w, stride=(4, 4), padding=(0, 0), dilation=(1, 1), groups=1)
    output = output.permute(0, 2, 3, 1)
    print(output)

    device_tensor = load_ttnn_tensor_from_txt("Testing/output.txt")

    # print(device_tensor.shape)
    # print(output.shape)  # Should be torch.Size([32, 384, 12, 12])
    # print(device_tensor)

    print(compare_tensors(device_tensor, output))
    print(max_abs_diff_and_values(device_tensor, output))
    plot_diff_distribution(device_tensor, output)
