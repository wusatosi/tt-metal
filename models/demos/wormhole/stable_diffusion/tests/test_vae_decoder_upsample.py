import torch
from diffusers import (
    AutoencoderKL,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
import ttnn


def prepare_split_conv(
    in_channels,
    out_channels,
    conv_in_channel_split_factor,
    conv_out_channel_split_factor,
    torch_weight_tensor,
    torch_bias_tensor,
):
    split_output_channels = out_channels // conv_out_channel_split_factor
    split_input_channels = in_channels // conv_in_channel_split_factor

    # weights
    if conv_out_channel_split_factor > 1:
        split_weight_tensors = list(torch.split(torch_weight_tensor, split_output_channels, 0))
    else:
        split_weight_tensors = [torch_weight_tensor]

    # bias
    if conv_in_channel_split_factor > 1:
        split_bias_tensors = list(torch.split(torch_bias_tensor, split_output_channels, 3))
    else:
        split_bias_tensors = [torch_bias_tensor]

    for i in range(len(split_weight_tensors)):
        split_weight_tensors[i] = torch.split(split_weight_tensors[i], split_input_channels, 1)

    return split_weight_tensors, split_bias_tensors


class UpsampleBlock:
    def __init__(
        self,
        torch_upsample,
        device,
        compute_config,
        in_channels,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
        conv_in_channel_split_factor,
        conv_out_channel_split_factor,
        scale_factor=2,
    ):
        self.device = device
        self.compute_config = compute_config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.conv_in_channel_split_factor = conv_in_channel_split_factor
        self.conv_out_channel_split_factor = conv_out_channel_split_factor
        self.scale_factor = scale_factor

        conv_weights, conv_biases = prepare_split_conv(
            in_channels,
            out_channels,
            conv_in_channel_split_factor,
            conv_out_channel_split_factor,
            torch_upsample.conv.weight,
            torch_upsample.conv.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        )

        self.conv_weight = [
            [
                ttnn.from_torch(
                    weight,
                    # device=device,
                    dtype=ttnn.float32,
                    # memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    # layout=ttnn.TILE_LAYOUT,
                )
                for weight in output_channel_spit_weights
            ]
            for output_channel_spit_weights in conv_weights
        ]
        self.conv_bias = [
            ttnn.from_torch(
                bias,
                # device=device,
                dtype=ttnn.float32,
                # memory_config=ttnn.DRAM_MEMORY_CONFIG,
                # layout=ttnn.TILE_LAYOUT,
            )
            for bias in conv_biases
        ]

        self.conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat8_b,
            weights_dtype=ttnn.bfloat8_b,
            activation="",
        )

    def __call__(self, hidden_states):
        if hidden_states.layout == ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)

        hidden_states = ttnn.upsample(hidden_states, self.scale_factor)

        # conv
        conv_kwargs = {
            "in_channels": in_channel_slice,
            "out_channels": out_channel_slice,
            "batch_size": 1,
            "input_height": self.output_height,
            "input_width": self.output_width,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "device": self.device,
            "conv_config": self.conv_config,
        }

        in_channel_slice = self.in_channels // self.conv_in_channel_split_factor
        out_channel_slice = self.out_channels // self.conv_out_channel_split_factor

        hidden_states_split = ttnn.split(hidden_states, in_channel_slice, 3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if len(hidden_states_split) != 1:
            hidden_states.deallocate(True)

        for i in range(len(hidden_states_split)):
            hidden_states_split[i] = ttnn.to_layout(hidden_states_split[i], ttnn.TILE_LAYOUT)
            hidden_states_split[i] = ttnn.typecast(hidden_states_split[i], ttnn.bfloat8_b)

        output = None
        for out_channel_slice_id in range(self.conv_out_channel_split_factor):
            out_channel_slice_output = None
            for in_channel_slice_id in range(self.conv_in_channel_split_factor):
                in_channel_slice_output = ttnn.conv2d(
                    input_tensor=hidden_states_split[in_channel_slice_id],
                    weight_tensor=self.conv_weight[out_channel_slice_id][in_channel_slice_id],
                    bias_tensor=self.conv_bias[out_channel_slice_id],
                    **conv_kwargs,
                    compute_config=self.compute_config,
                )

                if in_channel_slice_id == 0:
                    if in_channel_slice_output.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                        out_channel_slice_output = ttnn.to_memory_config(
                            in_channel_slice_output, ttnn.DRAM_MEMORY_CONFIG
                        )
                        in_channel_slice_output.deallocate(True)
                    else:
                        out_channel_slice_output = in_channel_slice_output
                else:
                    out_channel_slice_output = ttnn.add(
                        out_channel_slice_output, in_channel_slice_output, output_tensor=out_channel_slice_output
                    )
                    in_channel_slice_output.deallocate(True)

            if out_channel_slice_id == 0:
                if out_channel_slice_output.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                    output = ttnn.to_memory_config(out_channel_slice_output, ttnn.DRAM_MEMORY_CONFIG)
                    out_channel_slice_output.deallocate(True)
                else:
                    output = out_channel_slice_output
            else:
                output = ttnn.concat([output, out_channel_slice_output], dim=-1)
                out_channel_slice_output.deallocate(True)

        hidden_states = output
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.reshape(hidden_states, [1, self.output_height, self.output_width, self.out_channels])
        return hidden_states


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_channels, input_height, input_width, out_channels, output_height, output_width, conv_in_channel_split_factor, conv_out_channel_split_factor, block_id",
    [
        (512, 64, 64, 512, 128, 128, 1, 1, 0),
        (512, 128, 128, 512, 256, 256, 2, 2, 1),
        (256, 256, 256, 256, 512, 512, 8, 2, 2),
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
    conv_in_channel_split_factor,
    conv_out_channel_split_factor,
    block_id,
    use_program_cache,
):
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    # vae.decode(torch.randn([1, 4, 64, 64]))

    torch_upsample = vae.decoder.up_blocks[block_id].upsamplers[0]

    torch_input = torch.randn([1, input_channels, input_height, input_width])
    torch_output = torch_upsample(torch_input)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    ttnn_model = UpsampleBlock(
        torch_upsample,
        device,
        compute_config,
        input_channels,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
        conv_in_channel_split_factor,
        conv_out_channel_split_factor,
    )
    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.permute(ttnn_output, [0, 3, 1, 2])

    result = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, result, 0.97)
