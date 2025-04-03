import torch
from diffusers import (
    AutoencoderKL,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
import ttnn


class UpsampleBlock:
    def __init__(
        self,
        device,
        torch_upsample,
        in_channels,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
        scale_factor=2,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.scale_factor = scale_factor

        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        self.conv_weight = ttnn.from_torch(
            torch_upsample.conv.weight,
            device=device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.conv_bias = ttnn.from_torch(
            torch_upsample.conv.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def __call__(self, hidden_states):
        if hidden_states.layout == ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)

        hidden_states = ttnn.upsample(hidden_states, self.scale_factor)

        # conv
        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat8_b,
            weights_dtype=ttnn.bfloat8_b,
            activation="",
            input_channels_alignment=32,
            transpose_shards=False,
            preprocess_weights_on_device=True,
            always_preprocess_weights=True,
        )
        conv_kwargs_1 = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "batch_size": 1,
            "input_height": self.output_height,
            "input_width": self.output_width,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "device": self.device,
            "conv_config": conv_config,
        }

        hidden_states = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=self.conv_weight,
            bias_tensor=self.conv_bias,
            **conv_kwargs_1,
            compute_config=self.compute_config,
        )

        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.reshape(hidden_states, [1, self.output_height, self.output_width, self.out_channels])
        return hidden_states


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_channels, input_height, input_width, out_channels, output_height, output_width, block_id",
    [
        (512, 64, 64, 512, 128, 128, 0),
        (512, 128, 128, 512, 256, 256, 1),
        (256, 256, 256, 256, 512, 512, 2),
    ],
)
def test_upsample(
    device,
    input_channels,
    input_height,
    input_width,
    out_channels,
    output_height,
    output_width,
    block_id,
    # use_program_cache
):
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    # vae.decode(torch.randn([1, 4, 64, 64]))

    torch_upsample = vae.decoder.up_blocks[block_id].upsamplers[0]

    torch_input = torch.randn([1, input_channels, input_height, input_width])
    torch_output = torch_upsample(torch_input)

    ttnn_model = UpsampleBlock(
        device,
        torch_upsample,
        input_channels,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
    )
    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
    )
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.permute(ttnn_output, [0, 3, 1, 2])

    result = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, result, 0.99)

    print(result.shape)
