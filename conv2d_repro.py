import torch
import ttnn
import numpy as np
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import get_atol_rtol_pcc

L1_SMALL_SIZE = 1 << 15
torch.manual_seed(0)


def ttnn_out(input_torch, weights_ttnn, device):
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        math_approx_mode=True,
        dst_full_sync_en=False,
    )

    v4 = input_torch.transpose(1, 2)
    v5 = v4.transpose(2, 3)
    v6 = v5.reshape([1, 1, 196, 256])
    v7 = ttnn.from_torch(
        v6,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    v8 = ttnn.to_layout(v7, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED))
    v9 = ttnn.to_device(
        v8,
        device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )

    # Conv2d operation
    v11 = ttnn.conv2d(
        input_tensor=v9,
        weight_tensor=weights_ttnn,
        device=device,
        out_channels=512,
        in_channels=256,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        conv_config=ttnn.Conv2dConfig(dtype=ttnn.float32, weights_dtype=ttnn.float32),
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        compute_config=compute_config,
    )

    v11 = ttnn.to_torch(v11)
    v12 = v11.reshape([1, 7, 7, 512])
    v13 = v12.transpose(2, 3)
    v14 = v13.transpose(1, 2)
    return v14


def torch_out(input_torch, weights_torch):
    # PyTorch Conv2d operation
    conv = torch.nn.Conv2d(
        in_channels=256,
        out_channels=512,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        bias=False,
    )

    # Load weights
    conv.weight.data = weights_torch

    # Forward pass
    output = conv(input_torch)

    return output


def create_inputs_for_forward(device):
    # Create input tensor
    shape = (1, 256, 14, 14)
    input_torch = torch.randn(shape, dtype=torch.float32)

    # Load weights from file
    weights_torch = torch.load("conv_weights.pt")
    weights_ttnn = ttnn.from_torch(weights_torch, device=None, layout=ttnn.ROW_MAJOR_LAYOUT)

    return input_torch, weights_torch, weights_ttnn


def test_conv2d():
    # Set device
    device_id = 0
    device = ttnn.open_device(device_id=device_id, l1_small_size=L1_SMALL_SIZE)

    # Get input tensors and weights
    input_torch, weights_torch, weights_ttnn = create_inputs_for_forward(device)

    # Compute ttnn output
    tt_out = ttnn_out(input_torch, weights_ttnn, device)

    # Compute PyTorch output
    pytorch_out = torch_out(input_torch, weights_torch)

    # Calc rtol and atol
    print(f"PyTorch output: {pytorch_out}")
    print(f"TTNN output: {tt_out}")
    _, _, device_pcc, pcc_str = get_atol_rtol_pcc(pytorch_out, tt_out)
    print(f"Output: {pcc_str}")
