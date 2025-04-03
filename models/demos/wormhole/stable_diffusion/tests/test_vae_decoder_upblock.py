import torch
from diffusers import (
    AutoencoderKL,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
import ttnn

from models.demos.wormhole.stable_diffusion.tests.test_vae_decoder_upsample import UpsampleBlock
from models.demos.wormhole.stable_diffusion.tests.test_vae_decoder_resnet import ResnetBlock


class UpDecoderBlock:
    def __init__(
        self,
        device,
        torch_upblock,
        in_channels,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
        num_resnet_gn_blocks,
    ):
        self.resnets = []
        for i in range(2):
            self.resnets.append(
                ResnetBlock(
                    device,
                    torch_upblock.resnets[0],
                    in_channels if i == 0 else out_channels,
                    input_height,
                    input_width,
                    out_channels,
                    input_height,
                    input_width,
                    num_resnet_gn_blocks[i],
                )
            )

        if torch_upblock.upsamplers:
            self.upsample = UpsampleBlock(
                device,
                torch_upblock.upsamplers[0],
                out_channels,
                input_height,
                input_width,
                out_channels,
                output_height,
                output_width,
            )

    def __call__(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.upsample:
            hidden_states = self.upsample(hidden_states)

        return hidden_states


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_channels, input_height, input_width, out_channels, output_height, output_width, num_resnet_gn_blocks, block_id",
    [
        (512, 64, 64, 512, 128, 128, [1, 1, 1], 0),
        # (512, 128, 128, 512, 256, 256, [1, 1, 1], 1),
        # (512, 256, 256, 256, 512, 512, [4, 4, 4], 2),
        # (256, 512, 512, 128, 512, 512, [16, 32, 32], 3),
    ],
)
def test_upblock(
    device,
    input_channels,
    input_height,
    input_width,
    out_channels,
    output_height,
    output_width,
    num_resnet_gn_blocks,
    block_id,
    # use_program_cache
):
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    # vae.decode(torch.randn([1, 4, 64, 64]))

    torch_upblock = vae.decoder.up_blocks[block_id]

    torch_input = torch.randn([1, input_channels, input_height, input_width])
    torch_output = torch_upblock(torch_input)

    ttnn_model = UpDecoderBlock(
        device,
        torch_upblock,
        input_channels,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
        num_resnet_gn_blocks,
    )
    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.permute(ttnn_output, [0, 3, 1, 2])
    result = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, result, 0.99)

    print(result.shape)
