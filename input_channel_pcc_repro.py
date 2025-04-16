import torch
import ttnn
import pytest

from models.demos.segformer.tt.common import Conv
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "patch_size, stride, num_channels, hidden_size, batch_size, height, width, patch_emb_i",
    [
        (7, 4, 3, 32, 1, 512, 512, 0),
        (3, 2, 32, 64, 1, 128, 128, 1),
        (3, 2, 64, 160, 1, 64, 64, 2),
        (3, 2, 160, 256, 1, 32, 32, 3),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_wtf(
    patch_size,
    stride,
    num_channels,
    hidden_size,
    batch_size,
    height,
    width,
    patch_emb_i,
    device,
):
    torch.manual_seed(20250416)
    torch_input_tensor = torch.randn(batch_size, num_channels, height, width)

    torch_weights = torch.randn((hidden_size, num_channels, patch_size, patch_size), dtype=torch.bfloat16).float()
    torch_bias = torch.randn((1, 1, 1, hidden_size), dtype=torch.bfloat16).float()

    torch_output_tensor = (
        torch.nn.functional.conv2d(
            torch_input_tensor,
            torch_weights,
            bias=torch_bias.reshape(-1),
            stride=(stride, stride),
            padding=(patch_size // 2, patch_size // 2),
            dilation=(1, 1),
            groups=1,
        )
        .flatten(start_dim=2, end_dim=3)
        .transpose(1, 2)
    )

    torch_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # ttnn_input_tensor = ttnn.pad(ttnn_input_tensor, [batch_size, height, width, 8], [0, 0, 0, 0], 0)

    ttnn_weights = ttnn.from_torch(
        torch_weights,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn_bias = ttnn.from_torch(
        torch_bias,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat16,
        activation="",
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        transpose_shards=False,
        reshard_if_not_optimal=False,
        deallocate_activation=True,
        reallocate_halo_output=True,
        enable_act_double_buffer=True,
        enable_split_reader=False,
        output_layout=ttnn.TILE_LAYOUT,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    conv_kwargs = {
        "in_channels": ttnn_input_tensor.shape[3],
        "out_channels": hidden_size,
        "batch_size": ttnn_input_tensor.shape[0],
        "input_height": ttnn_input_tensor.shape[1],
        "input_width": ttnn_input_tensor.shape[2],
        "kernel_size": (ttnn_weights.shape[2], ttnn_weights.shape[3]),
        "stride": (stride, stride),
        "padding": (patch_size // 2, patch_size // 2),
        "dilation": (1, 1),
        "groups": 1,
        "device": device,
        "conv_config": conv_config,
    }

    if not ttnn.is_tensor_storage_on_device(ttnn_weights):
        ttnn_weights = ttnn.prepare_conv_weights(
            weight_tensor=ttnn_weights,
            weights_format="OIHW",
            input_memory_config=ttnn_input_tensor.memory_config(),
            input_layout=ttnn_input_tensor.get_layout(),
            has_bias=True,
            **conv_kwargs,
        )
        ttnn_bias = ttnn.prepare_conv_bias(
            bias_tensor=ttnn_bias,
            input_memory_config=ttnn_input_tensor.memory_config(),
            input_layout=ttnn_input_tensor.get_layout(),
            **conv_kwargs,
        )
        ttnn_weights = ttnn.to_device(ttnn_weights, device)
        ttnn_bias = ttnn.to_device(ttnn_bias, device)

    [ttnn_output_tensor, [_out_height, _out_width]] = ttnn.conv2d(
        input_tensor=ttnn_input_tensor,
        weight_tensor=ttnn_weights,
        bias_tensor=ttnn_bias,
        **conv_kwargs,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=False,
    )
    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)

    _, pcc_message = assert_with_pcc(torch_output_tensor, ttnn_output_tensor[0], pcc=0.99)
    print(pcc_message)
