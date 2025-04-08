import torch
import pytest
import ttnn
import sys
from loguru import logger


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384 * 2}], indirect=True)
def test_conv_row_major_hang(device):
    batch_size = 1
    dilation = (1, 1)
    groups = 1
    in_channels = 3
    input_height = 384
    input_width = 512
    kernel_size = (32, 32)
    out_channels = 768
    padding = (0, 0)
    stride = (32, 32)

    # Create activation, weight shapes with random values.
    input_shape = (batch_size, input_height, input_width, in_channels)
    weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_weight = torch.randn(weight_shape, dtype=torch.bfloat16)

    # Move inputs to device in row-major, keep weights in system memory.
    tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(torch_weight)

    # Perform the convolution
    logger.info("Running conv2d now...")
    output = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        groups=groups,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Hang here when using ROW_MAJOR inputs.
    logger.info("Before from_device")
    output_tensor = ttnn.from_device(output)
    logger.info("Output tensor shape: ", output_tensor.shape)
