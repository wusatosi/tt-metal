import ttnn
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_midblock import MidBlock
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_upblock import UpDecoderBlock

from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_utils import (
    get_default_compute_config,
    get_default_conv_config,
    prepare_split_conv_weights_bias,
    split_conv_and_run,
    prepare_group_norm,
)

from models.utility_functions import is_wormhole_b0


class VaeDecoder:
    def __init__(
        self,
        torch_decoder,
        device,
        in_channels,
        input_height,
        input_width,
        mid_channels,
        out_channels,
        output_height,
        output_width,
        conv_in_channel_split_factor=(1, 1),
        conv_out_channel_split_factor=(2, 1),
    ):
        self.device = device
        self.in_channels = in_channels
        self.input_height = input_height
        self.input_width = input_width
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.output_height = output_height
        self.output_width = output_width
        self.conv_in_channel_split_factor = conv_in_channel_split_factor
        self.conv_out_channel_split_factor = conv_out_channel_split_factor

        self.compute_config = get_default_compute_config(device)
        self.conv_config = get_default_conv_config()

        # conv in
        self.conv_in_weights, self.conv_in_bias = prepare_split_conv_weights_bias(
            in_channels,
            mid_channels,
            conv_in_channel_split_factor[0],
            conv_in_channel_split_factor[1],
            torch_decoder.conv_in.weight,
            torch_decoder.conv_in.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        )

        in_channels = mid_channels

        # move somewhere?
        in_channels_list = [in_channels, in_channels, in_channels, in_channels // 2]

        input_dimension_list = [input_height, input_height * 2, input_height * 4, input_height * 8]

        out_channels_list = [in_channels, in_channels, in_channels // 2, in_channels // 4]

        output_dimension_list = [input_height * 2, input_height * 4, input_height * 8, input_height * 8]

        resnet_norm_blocks = [
            [(1, 1), (1, 1), (1, 1)],
            [(1, 1), (1, 1), (1, 1)],
            [(4, 4), (4, 4), (4, 16)],
            [(16, 32), (32, 32), (32, 32)],
        ]

        resnet_conv1_channel_split_factors = [
            [(1, 1), (1, 1), (1, 1)],
            [(1, 1), (1, 1), (1, 1)],
            [(2, 1), (1, 1), (1, 1)],
            [(8, 1), (4, 1), (4, 1)],
        ]

        resnet_conv2_channel_split_factors = [
            [(1, 1), (1, 1), (1, 1)],
            [(1, 1), (1, 1), (1, 1)],
            [(1, 1), (1, 1), (1, 1)],
            [(4, 1), (4, 1), (4, 1)],
        ]

        upsample_conv_channel_split_factors = [
            (1, 1),
            (8 if is_wormhole_b0() else 2, 2 if is_wormhole_b0() else 2),
            (8 if is_wormhole_b0() else 4, 2),
            (1, 1),
        ]

        self.upblocks = []
        for i in range(4):
            self.upblocks.append(
                UpDecoderBlock(
                    torch_decoder.up_blocks[i],
                    device,
                    in_channels_list[i],
                    input_dimension_list[i],
                    input_dimension_list[i],
                    out_channels_list[i],
                    output_dimension_list[i],
                    output_dimension_list[i],
                    resnet_norm_blocks[i],
                    resnet_conv1_channel_split_factors[i],
                    resnet_conv2_channel_split_factors[i],
                    upsample_conv_channel_split_factors[i],
                )
            )

        self.midblock = MidBlock(
            torch_decoder.mid_block,
            device,
            in_channels,
            input_height,
            input_width,
        )

        self.norm_num_blocks = 16
        self.norm_grid_core = ttnn.CoreGrid(y=4, x=8)
        (
            self.norm_input_mask,
            self.norm_weights,
            self.norm_bias,
        ) = prepare_group_norm(
            self.device,
            128,
            self.norm_grid_core,
            torch_decoder.conv_norm_out.weight,
            torch_decoder.conv_norm_out.bias,
        )

        # conv out
        self.conv_out_weights, self.conv_out_bias = prepare_split_conv_weights_bias(
            128,
            3,
            conv_out_channel_split_factor[0],
            conv_out_channel_split_factor[1],
            torch_decoder.conv_out.weight,
            torch_decoder.conv_out.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        )

    def __call__(self, hidden_states):
        # conv in
        hidden_states = split_conv_and_run(
            hidden_states,
            self.conv_in_weights,
            self.conv_in_bias,
            self.device,
            self.in_channels,
            self.input_height,
            self.input_width,
            self.mid_channels,
            self.conv_in_channel_split_factor[0],
            self.conv_in_channel_split_factor[1],
            self.compute_config,
            self.conv_config,
        )

        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)

        hidden_states = self.midblock(hidden_states)

        # upblocks
        for upblock in self.upblocks:
            hidden_states = upblock(hidden_states)

        hidden_states = ttnn.typecast(hidden_states, ttnn.bfloat16)

        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=32,
            input_mask=self.norm_input_mask,
            weight=self.norm_weights,
            bias=self.norm_bias,
            epsilon=1e-6,
            core_grid=self.norm_grid_core,
            dtype=ttnn.bfloat8_b,
            inplace=False,
            num_out_blocks=self.norm_num_blocks,
        )

        # hidden_states = ttnn.silu(hidden_states)
        # conv out
        # hidden_states = split_conv_and_run(
        #     hidden_states,
        #     self.conv_out_weights,
        #     self.conv_out_bias,
        #     self.device,
        #     128,
        #     self.output_height,
        #     self.output_width,
        #     3,
        #     self.conv_out_channel_split_factor[0],
        #     self.conv_out_channel_split_factor[1],
        #     self.compute_config,
        #     self.conv_config,
        # )
        hidden_states = ttnn.reshape(hidden_states, [1, self.output_height, self.output_width, 128])

        return hidden_states
