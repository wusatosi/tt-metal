import torch
from diffusers import (
    AutoencoderKL,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
import ttnn


class VaeDecoder:
    def __init__(
        self,
        torch_decoder,
        device,
        compute_config,
        in_channels,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
        midblock_resnet_norm_blocks,
        upblock_resnet_norm_blocks,
    ):
        self.device = device
        self.compute_config = compute_config
        self.in_channels = in_channels
        self.input_height = input_height
        self.input_width = input_width
        self.out_channels = out_channels
        self.output_height = output_height
        self.output_width = output_width

        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        self.conv_in_out_channels = torch_decoder.conv_in.out_channels
        self.conv_in_weight = ttnn.from_torch(
            torch_decoder.conv_in.weight,
            device=device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.conv_in_bias = ttnn.from_torch(
            torch_decoder.conv_in.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        self.conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat8_b,
            weights_dtype=ttnn.bfloat8_b,
            activation="",
            input_channels_alignment=32,
            transpose_shards=False,
            preprocess_weights_on_device=True,
            always_preprocess_weights=True,
        )

        # self.midblock = MidBlock(
        #     device,
        #     torch_decoder.mid_block,
        #     in_channels,
        #     input_height,
        #     input_width,
        #     midblock_resnet_norm_blocks,
        # )

        # # move somewhere?
        # in_channels_list = [
        #     in_channels,
        #     in_channels,
        #     in_channels,
        #     in_channels // 2
        # ]

        # input_dimension_list = [
        #     input_height,
        #     input_height*2,
        #     input_height*4,
        #     input_height*8
        # ]

        # out_channels_list = [
        #     in_channels,
        #     in_channels,
        #     in_channels // 2,
        #     out_channels
        # ]

        # output_dimension_list = [
        #     input_height*2,
        #     input_height*4,
        #     input_height*8,
        #     input_height*8
        # ]

        # self.upblocks = []
        # for i in range(4):
        #     self.upblocks.append(
        #         UpDecoderBlock(
        #             device,
        #             torch_decoder.up_blocks[i],
        #             in_channels_list[i],
        #             input_dimension_list[i],
        #             input_dimension_list[i],
        #             out_channels_list[i],
        #             output_dimension_list[i],
        #             output_dimension_list[i],
        #             upblock_resnet_norm_blocks[i]
        #         )
        #     )

        # self.conv_out_weight = ttnn.from_torch(
        #     torch_decoder.conv_in.weight,
        #     device=device,
        #     dtype=ttnn.bfloat16,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     layout=ttnn.ROW_MAJOR_LAYOUT,
        # )
        # self.conv_out_bias = ttnn.from_torch(
        #     torch_decoder.conv_in.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        #     device=device,
        #     dtype=ttnn.bfloat16,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     layout=ttnn.ROW_MAJOR_LAYOUT,
        # )

    def __call__(self, hidden_states):
        # conv in
        conv_in_kwargs = {
            "in_channels": self.in_channels,
            "out_channels": self.conv_in_out_channels,
            "batch_size": 1,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "device": self.device,
            "conv_config": self.conv_config,
        }
        hidden_states = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=self.conv_in_weight,
            bias_tensor=self.conv_in_bias,
            **conv_in_kwargs,
            compute_config=self.compute_config,
        )

        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.reshape(hidden_states, [1, self.input_height, self.input_width, self.conv_in_out_channels])
        return hidden_states


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_channels, input_height, input_width, out_channels, output_height, output_width, midblock_resnet_norm_blocks, upblock_resnet_norm_blocks",
    [
        (
            4,
            64,
            64,
            3,
            512,
            512,
            [(1, 1), (1, 1)],
            [
                [(1, 1), (1, 1), (1, 1)],
                [(1, 1), (1, 1), (1, 1)],
                [(4, 4), (4, 4), (4, 16)],
                [(16, 32), (32, 32), (32, 32)],
            ],
        ),
    ],
)
def test_decoder(
    device,
    input_channels,
    input_height,
    input_width,
    out_channels,
    output_height,
    output_width,
    midblock_resnet_norm_blocks,
    upblock_resnet_norm_blocks
    # use_program_cache
):
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.decode(torch.randn([1, 4, 64, 64]))

    torch_decoder = vae.decoder

    torch_input = torch.randn([1, input_channels, input_height, input_width])
    torch_output = torch_decoder(torch_input)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    ttnn_model = VaeDecoder(
        torch_decoder,
        device,
        compute_config,
        input_channels,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
        midblock_resnet_norm_blocks,
        upblock_resnet_norm_blocks,
    )
    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.permute(ttnn_output, [0, 3, 1, 2])
    result = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, result, 0.99)

    print(result.shape)
