import torch
import pytest
import ttnn
import sys
from loguru import logger


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_layout",
    [pytest.param(ttnn.TILE_LAYOUT, id="TILE_LAYOUT"), pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="ROW_MAJOR_LAYOUT")],
)
def test_conv_row_major_hang(device, input_layout):
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
    # input_shape = (batch_size, input_height, input_width, in_channels)
    input_shape = (batch_size, in_channels, input_height, input_width)

    weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_weight = torch.randn(weight_shape, dtype=torch.bfloat16)

    # Move inputs to device with specified layout, keep weights in system memory.
    # Original here, but then I added datatype for weight.
    # tt_input = ttnn.from_torch(torch_input, device=device, layout=input_layout)
    # tt_weight = ttnn.from_torch(torch_weight, ttnn.DataType(ttnn.bfloat16))

    # Copied this from conv2d_common.py sweep file, didn't change much.
    torch_input = torch.reshape(torch_input, (1, 1, batch_size * input_height * input_width, in_channels))

    tt_input = ttnn.from_torch(torch_input, device=device, layout=input_layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_weight = ttnn.from_torch(torch_weight, ttnn.DataType(ttnn.bfloat16))

    # Perform the convolution
    logger.info(f"Running conv2d w/ input_layout: {input_layout}")

    # With Conv2dConfig, this the assert is hit like TTNN Sweep.
    # E       Statically allocated circular buffers in program 7 clash with L1 buffers on core range [(x=0,y=0) - (x=5,y=0)]. L1 buffer allocated at 417792 and static circular buffer region ends at 523296
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.TILE_LAYOUT,
        preprocess_weights_on_device=False,
        reallocate_halo_output=True,  # To match tt-mlir
    )

    # FIXE - Uncomment this, and see hang instead of assert.
    # conv_config = None

    print(
        f"ttnn.conv2d parameters: input_tensor={tt_input.shape}, weight_tensor={tt_weight.shape}, "
        f"in_channels={in_channels}, out_channels={out_channels}, device={device}, "
        f"bias_tensor=None, kernel_size={kernel_size}, stride={stride}, "
        f"padding={padding}, dilation={dilation}, batch_size={batch_size}, "
        f"input_height={input_height}, input_width={input_width}, groups={groups}, "
        f"conv_config={conv_config}",
        file=sys.stdout,
        flush=True,
    )

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
        conv_config=conv_config,
    )

    # Hang here when using ROW_MAJOR inputs.
    logger.info("Before from_device")
    output_tensor = ttnn.from_device(output)
    logger.info(f"Output tensor shape: {output_tensor.shape}")
