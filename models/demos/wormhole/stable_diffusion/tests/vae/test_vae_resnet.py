import torch
from diffusers import (
    AutoencoderKL,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
import ttnn
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_resnet import ResnetBlock


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_channels, input_height, input_width, out_channels, norm_num_blocks, conv1_channel_split_factors, conv2_channel_split_factors, block, block_id, resnet_block_id",
    [
        (512, 64, 64, 512, (1, 1), (1, 1), (1, 1), "mid", None, 0),
        (512, 64, 64, 512, (1, 1), (1, 1), (1, 1), "mid", None, 1),
        (512, 64, 64, 512, (1, 1), (1, 1), (1, 1), "up", 0, 0),
        (512, 64, 64, 512, (1, 1), (1, 1), (1, 1), "up", 0, 1),
        (512, 64, 64, 512, (1, 1), (1, 1), (1, 1), "up", 0, 2),
        (512, 128, 128, 512, (1, 1), (1, 1), (1, 1), "up", 1, 0),
        (512, 128, 128, 512, (1, 1), (1, 1), (1, 1), "up", 1, 1),
        (512, 128, 128, 512, (1, 1), (1, 1), (1, 1), "up", 1, 2),
        (512, 256, 256, 256, (4, 4), (2, 1), (1, 1), "up", 2, 0),
        (256, 256, 256, 256, (4, 4), (1, 1), (1, 1), "up", 2, 1),
        (256, 256, 256, 256, (4, 16), (1, 1), (1, 1), "up", 2, 2),
        (256, 512, 512, 128, (16, 32), (8, 1), (4, 1), "up", 3, 0),
        (128, 512, 512, 128, (32, 32), (4, 1), (4, 1), "up", 3, 1),
        (128, 512, 512, 128, (32, 32), (4, 1), (4, 1), "up", 3, 2),
    ],
)
def test_vae_resnet(
    device,
    input_channels,
    input_height,
    input_width,
    out_channels,
    norm_num_blocks,
    conv1_channel_split_factors,
    conv2_channel_split_factors,
    block,
    block_id,
    resnet_block_id,
    use_program_cache,
):
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    if block == "mid":
        torch_resnet = vae.decoder.mid_block.resnets[resnet_block_id]
    else:
        torch_resnet = vae.decoder.up_blocks[block_id].resnets[resnet_block_id]

    # Run pytorch model
    torch_input = torch.randn([1, input_channels, input_height, input_width])
    torch_output = torch_resnet(torch_input, temb=None)

    # Initialize ttnn model
    ttnn_model = ResnetBlock(
        torch_resnet,
        device,
        input_channels,
        input_height,
        input_width,
        out_channels,
        norm_num_blocks[0],
        norm_num_blocks[1],
        conv1_channel_split_factors,
        conv2_channel_split_factors,
    )

    # Prepare ttnn input
    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    # Run ttnn model twice to test program cache and weights reuse
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output.deallocate(True)
    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output = ttnn.permute(ttnn_output, [0, 3, 1, 2])
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.96)
