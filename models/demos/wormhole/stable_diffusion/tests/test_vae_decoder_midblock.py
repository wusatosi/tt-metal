import torch
from diffusers import (
    AutoencoderKL,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
import ttnn

from models.demos.wormhole.stable_diffusion.tests.test_vae_decoder_resnet import ResnetBlock


class MidBlock:
    def __init__(
        self,
        device,
        torch_midblock,
        in_channels,
        input_height,
        input_width,
        num_resnet_gn_blocks,
    ):
        self.resnets = []
        self.resnets.append(
            ResnetBlock(
                device,
                torch_midblock.resnets[0],
                in_channels,
                input_height,
                input_width,
                in_channels,
                input_height,
                input_width,
                num_resnet_gn_blocks[0],
            )
        )

        self.resnets.append(
            ResnetBlock(
                device,
                torch_midblock.resnets[1],
                in_channels,
                input_height,
                input_width,
                in_channels,
                input_height,
                input_width,
                num_resnet_gn_blocks[1],
            )
        )

    def __call__(self, hidden_states):
        hidden_states = self.resnets[0](hidden_states)
        # add attention
        hidden_states = self.resnets[1](hidden_states)
        return hidden_states


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_channels, input_height, input_width, num_resnet_gn_blocks",
    [
        (512, 64, 64, [1, 1]),
    ],
)
def test_upblock(
    device,
    input_channels,
    input_height,
    input_width,
    num_resnet_gn_blocks,
    # use_program_cache
):
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.decode(torch.randn([1, 4, 64, 64]))

    torch_midblock = vae.decoder.mid_block

    torch_input = torch.randn([1, input_channels, input_height, input_width])
    torch_output = torch_midblock(torch_input)

    ttnn_model = MidBlock(device, torch_midblock, input_channels, input_height, input_width, num_resnet_gn_blocks)
    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.permute(ttnn_output, [0, 3, 1, 2])
    result = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, result, 0.99)

    print(result.shape)
