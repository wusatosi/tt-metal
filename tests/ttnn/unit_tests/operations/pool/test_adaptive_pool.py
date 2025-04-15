import torch
import math
import pytest
import ttnn


def adaptive_to_max_pool2d(input_tensor, output_size):
    """
    Convert AdaptiveMaxPool2d to equivalent MaxPool2d operation
    Handles cases where input dimensions are not perfectly divisible by output dimensions

    Args:
        input_tensor: Input tensor of shape (N, C, H, W)
        output_size: Desired output size (tuple or int)

    Returns:
        Tuple of (kernel_size, stride, padding) for MaxPool2d and a note if exact conversion isn't possible
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    input_height, input_width = input_tensor.shape[2], input_tensor.shape[3]
    output_height, output_width = output_size

    # Check if dimensions are valid
    if input_height < output_height or input_width < output_width:
        raise ValueError("Output size cannot be larger than input size for max pooling")

    # Calculate stride (might be floating point)
    stride_h_float = input_height / output_height
    stride_w_float = input_width / output_width

    # Round down stride to integer
    stride_h = math.floor(stride_h_float)
    stride_w = math.floor(stride_w_float)

    # Ensure stride is at least 1
    stride_h = max(1, stride_h)
    stride_w = max(1, stride_w)

    # Calculate kernel size
    kernel_h = input_height - (output_height - 1) * stride_h
    kernel_w = input_width - (output_width - 1) * stride_w

    # Handle case where kernel size might be too large
    if kernel_h > input_height:
        kernel_h = input_height
    if kernel_w > input_width:
        kernel_w = input_width

    # Calculate if this is an exact conversion
    is_exact = (
        stride_h_float == stride_h
        and stride_w_float == stride_w
        and input_height == (output_height - 1) * stride_h + kernel_h
        and input_width == (output_width - 1) * stride_w + kernel_w
    )

    message = ""
    if not is_exact:
        message = (
            "Note: This is an approximation. For non-integer stride ratios, "
            "AdaptiveMaxPool2d uses a more complex logic with varying kernel sizes."
        )

    return (kernel_h, kernel_w), (stride_h, stride_w), (0, 0), message


@pytest.mark.parametrize(
    "input_height, input_width, output_height, output_width",
    [
        [20, 20, 3, 3],
        [40, 40, 3, 3],
        [80, 80, 3, 3],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_adaptive_to_max_pool2d(device, input_height, input_width, output_height, output_width):
    batch_size = 1
    input_channels = 256
    input_tensor = torch.randn(batch_size, input_channels, input_height, input_width)

    # Define output size
    output_size = (output_height, output_width)

    # Use AdaptiveMaxPool2d
    adaptive_pool = torch.nn.AdaptiveMaxPool2d(output_size)
    adaptive_output = adaptive_pool(input_tensor)

    # Convert to MaxPool2d
    kernel_size, stride, padding, message = adaptive_to_max_pool2d(input_tensor, output_size)
    print(kernel_size)
    print(stride)
    print(padding)
    # max_pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    # max_pool_output = max_pool(input_tensor)

    nhwc_input = input_tensor.permute(0, 2, 3, 1)
    nhwc_input = nhwc_input.reshape(1, 1, batch_size * input_height * input_width, input_channels)
    ttnn_input = ttnn.from_torch(nhwc_input, ttnn.bfloat16, device=device)
    max_pool_output = ttnn.max_pool2d(
        input_tensor=ttnn_input,
        batch_size=batch_size,
        input_h=input_height,
        input_w=input_width,
        channels=input_channels,
        kernel_size=[kernel_size[0], kernel_size[1]],
        stride=[stride[0], stride[1]],
        padding=[padding[0], padding[1]],
        dilation=[1, 1],
        applied_shard_scheme=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    )
    max_pool_output = ttnn.to_torch(max_pool_output)
    max_pool_output = max_pool_output.reshape(batch_size, output_height, output_width, input_channels)
    max_pool_output = max_pool_output.permute(0, 3, 1, 2)
    max_pool_output = max_pool_output.to(torch.float32)

    # Compare outputs
    difference = torch.abs(adaptive_output - max_pool_output).sum().item()
    print(f"Input shape: {input_tensor.shape}")
    print(f"Target output shape: {output_size}")
    print(f"Calculated parameters - kernel_size: {kernel_size}, stride: {stride}, padding: {padding}")
    print(f"AdaptiveMaxPool2d output shape: {adaptive_output.shape}")
    print(f"MaxPool2d output shape: {max_pool_output.shape}")
    print(f"Difference between outputs: {difference}")
    print(f"Are outputs identical? {torch.allclose(adaptive_output, max_pool_output)}")
