import torch
from diffusers import (
    AutoencoderKL,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
import ttnn


def prepare_group_norm(device, in_channels, core_grid, torch_weights, torch_bias):
    num_cores_across_channel = core_grid.y

    torch_input_mask = ttnn.create_group_norm_input_mask(in_channels, 32, num_cores_across_channel)
    input_mask = ttnn.from_torch(
        torch_input_mask,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    weights = ttnn.from_torch(
        ttnn.create_group_norm_weight_bias_rm(torch_weights, in_channels, num_cores_across_channel),
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bias = ttnn.from_torch(
        ttnn.create_group_norm_weight_bias_rm(torch_bias, in_channels, num_cores_across_channel),
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return input_mask, weights, bias


class ResnetBlock:
    def __init__(
        self,
        torch_resnet,
        device,
        compute_config,
        in_channels,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
        norm1_num_blocks,
        norm2_num_blocks,
    ):
        self.device = device
        self.compute_config = compute_config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        # groupnorm 1
        self.norm1_num_blocks = norm1_num_blocks
        self.norm1_grid_core = ttnn.CoreGrid(y=4, x=8) if in_channels == 128 else ttnn.CoreGrid(y=8, x=8)
        (
            self.norm1_input_mask,
            self.norm1_weights,
            self.norm1_bias,
        ) = prepare_group_norm(
            self.device,
            in_channels,
            self.norm1_grid_core,
            torch_resnet.norm1.weight,
            torch_resnet.norm1.bias,
        )

        # conv 1
        self.conv1_weight = ttnn.from_torch(
            torch_resnet.conv1.weight,
            device=device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.conv1_bias = ttnn.from_torch(
            torch_resnet.conv1.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
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

        # groupnorm 2
        self.norm2_num_blocks = norm2_num_blocks
        self.norm2_grid_core = ttnn.CoreGrid(y=4, x=8) if out_channels == 128 else ttnn.CoreGrid(y=8, x=8)
        (
            self.norm2_input_mask,
            self.norm2_weights,
            self.norm2_bias,
        ) = prepare_group_norm(
            self.device,
            out_channels,
            self.norm2_grid_core,
            torch_resnet.norm2.weight,
            torch_resnet.norm2.bias,
        )

        # conv 2
        self.conv2_weight = ttnn.from_torch(
            torch_resnet.conv2.weight,
            device=device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.conv2_bias = ttnn.from_torch(
            torch_resnet.conv2.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def __call__(self, input_tensor, groups=32, eps=1e-5):
        hidden_states = input_tensor
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(hidden_states, [1, 1, self.input_height * self.input_width, self.in_channels])
        hidden_states = ttnn.tilize_with_zero_padding(hidden_states, use_multicore=True)

        # groupnorm 1
        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=groups,
            input_mask=self.norm1_input_mask,
            weight=self.norm1_weights,
            bias=self.norm1_bias,
            epsilon=eps,
            core_grid=self.norm1_grid_core,
            dtype=ttnn.bfloat8_b,
            inplace=False,
            num_out_blocks=self.norm1_num_blocks,
        )

        # silu 1
        hidden_states = ttnn.silu(hidden_states)

        # conv 1
        conv1_kwargs = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
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
            weight_tensor=self.conv1_weight,
            bias_tensor=self.conv1_bias,
            **conv1_kwargs,
            compute_config=self.compute_config,
        )

        # groupnorm 2
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.typecast(hidden_states, ttnn.bfloat16)

        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=groups,
            input_mask=self.norm2_input_mask,
            weight=self.norm2_weights,
            bias=self.norm2_bias,
            epsilon=eps,
            core_grid=self.norm2_grid_core,
            dtype=ttnn.bfloat8_b,
            inplace=False,
            num_out_blocks=self.norm2_num_blocks,
        )

        # silu 2
        hidden_states = ttnn.silu(hidden_states)

        conv2_kwargs = {
            "in_channels": self.out_channels,
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
            "conv_config": self.conv_config,
        }

        # conv 2
        hidden_states = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=self.conv2_weight,
            bias_tensor=self.conv2_bias,
            **conv2_kwargs,
            compute_config=self.compute_config,
        )

        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.reshape(hidden_states, [1, self.output_height, self.output_width, self.out_channels])

        hidden_states = hidden_states + input_tensor

        return hidden_states


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_channels, input_height, input_width, out_channels, output_height, output_width, norm_num_blocks, block, block_id, resnet_block_id",
    [
        # passing
        (512, 64, 64, 512, 64, 64, (1, 1), "mid", None, 0),
        # (512, 64, 64, 512, 64, 64, (1, 1), "mid", None, 1),
        # (512, 64, 64, 512, 64, 64, (1, 1), "up", 0, 0),
        # (512, 64, 64, 512, 64, 64, (1, 1), "up", 0, 1),
        # (512, 64, 64, 512, 64, 64, (1, 1), "up", 0, 2),
        # (512, 128, 128, 512, 128, 128, (1, 1), "up", 1, 0),
        # (512, 128, 128, 512, 128, 128, (1, 1), "up", 1, 1),
        # (512, 128, 128, 512, 128, 128, (1, 1), "up", 1, 2),
        # # failing
        # (512, 256, 256, 256, 256, 256, (4, 4), "up", 2, 0),
        # (256, 256, 256, 256, 256, 256, (4, 4), "up", 2, 1),
        # (256, 256, 256, 256, 256, 256, (4, 16), "up", 2, 2),
        # (256, 512, 512, 128, 512, 512, (16, 32), "up", 3, 0),
        # (128, 512, 512, 128, 512, 512, (32, 32), "up", 3, 1),
        # (128, 512, 512, 128, 512, 512, (32, 32), "up", 3, 2),
    ],
)
def test_resnet(
    device,
    input_channels,
    input_height,
    input_width,
    out_channels,
    output_height,
    output_width,
    norm_num_blocks,
    block,
    block_id,
    resnet_block_id,
    use_program_cache,
):
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    # vae.decode(torch.randn([1, 4, 64, 64]))

    if block == "mid":
        torch_resnet = vae.decoder.mid_block.resnets[resnet_block_id]
    else:
        torch_resnet = vae.decoder.up_blocks[block_id].resnets[resnet_block_id]

    torch_input = torch.randn([1, input_channels, input_height, input_width])
    torch_output = torch_resnet(torch_input, temb=None)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    ttnn_model = ResnetBlock(
        torch_resnet,
        device,
        compute_config,
        input_channels,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
        norm_num_blocks[0],
        norm_num_blocks[1],
    )
    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.permute(ttnn_output, [0, 3, 1, 2])
    result = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, result, 0.96)
