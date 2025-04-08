import torch
import typing
import pytest
import ttnn
import tempfile
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc


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

    # ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 196608 + d1 * 196608 + d2, d3), <1x1>, memref<196608x3xbf16, #dram>, <interleaved>>
    # (tensor<1x1x196608x3xbf16, #ttnn_layout8>,

    # ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 96 + d1 * 32 + d2, d3), <1x1>, memref<73728x32xbf16, #system_memory>>
    # tensor<768x3x32x32xbf16, #ttnn_layout1>, !ttnn.device)

    # ttnn_layout9 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 32 + d2, d3), <1x1>, memref<12x24x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
    # -> tensor<1x1x192x768xbf16, #ttnn_layout9> loc(#loc5)

    # E       RuntimeError: TT_FATAL @ /localdev/kmabee/metal3/ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp:408: output_channels <= b.get_padded_shape()[3]
    # E       info:
    # E       Invalid weight shape. Incorrect weight tensor.

    # Create Shapes.
    input_shape = (batch_size, input_height, input_width, in_channels)
    weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])

    # Use Random Values
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_weight = torch.randn(weight_shape, dtype=torch.bfloat16)

    # # Convert to device tensors
    # tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)
    # tt_weight = ttnn.from_torch(torch_weight, device=device, layout=ttnn.TILE_LAYOUT)

    # # Convert to row major layout as in the MLIR file
    # tt_input_cpu = ttnn.from_device(tt_input)
    # tt_input_row_major = ttnn.to_layout(tt_input_cpu, layout=ttnn.ROW_MAJOR_LAYOUT)
    # tt_input = ttnn.to_device(tt_input_row_major, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Convert to device tensors
    tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_weight = ttnn.from_torch(torch_weight, device=device, layout=ttnn.TILE_LAYOUT)  # Hmm, layout said RM<

    # Perform the convolution
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

    output_tensor = ttnn.from_device(output)
