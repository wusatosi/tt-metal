import ttnn
import pytest
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc


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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_channels, input_height, input_width, out_channels, output_height, output_width, conv_in_channel_split_factor, conv_out_channel_split_factor",
    [
        (512, 64, 64, 512, 128, 128, 1, 1),
        (512, 256, 256, 512, 256, 256, 2, 2),
        (256, 256, 256, 256, 512, 512, 8, 2),
    ],
)
def test_split_conv(
    device,
    input_channels,
    input_height,
    input_width,
    out_channels,
    output_height,
    output_width,
    conv_in_channel_split_factor,
    conv_out_channel_split_factor,
    use_program_cache,
):
    torch_input = torch.randn([1, input_channels, input_height, input_width])
    torch_weights = torch.randn([out_channels, input_channels, 3, 3])
    torch_biases = torch.randn([out_channels])

    torch_output = torch.nn.functional.conv2d(
        torch_input, torch_weights, bias=torch_biases, stride=(1, 1), padding=(1, 1), groups=1
    )

    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    conv_weights, conv_biases = prepare_split_conv(
        input_channels,
        out_channels,
        conv_in_channel_split_factor,
        conv_out_channel_split_factor,
        torch_weights,
        torch_biases.unsqueeze(0).unsqueeze(0).unsqueeze(0),
    )

    conv_weight = [
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
    conv_bias = [
        ttnn.from_torch(
            bias,
            # device=device,
            dtype=ttnn.float32,
            # memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # layout=ttnn.TILE_LAYOUT,
        )
        for bias in conv_biases
    ]

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        activation="",
    )

    in_channel_slice = input_channels // conv_in_channel_split_factor
    out_channel_slice = out_channels // conv_out_channel_split_factor

    hidden_states_split = ttnn.split(ttnn_input, in_channel_slice, 3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for i in range(len(hidden_states_split)):
        hidden_states_split[i] = ttnn.to_layout(hidden_states_split[i], ttnn.TILE_LAYOUT)
        hidden_states_split[i] = ttnn.typecast(hidden_states_split[i], ttnn.bfloat8_b)

    # conv
    conv_kwargs = {
        "in_channels": in_channel_slice,
        "out_channels": out_channel_slice,
        "batch_size": 1,
        "input_height": output_height,
        "input_width": output_width,
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
        "groups": 1,
        "device": device,
        "conv_config": conv_config,
    }

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    output = None
    for out_channel_slice_id in range(conv_out_channel_split_factor):
        out_channel_slice_output = None
        for in_channel_slice_id in range(conv_in_channel_split_factor):
            in_channel_slice_output = ttnn.conv2d(
                input_tensor=hidden_states_split[in_channel_slice_id],
                weight_tensor=conv_weight[out_channel_slice_id][in_channel_slice_id],
                bias_tensor=conv_bias[out_channel_slice_id],
                **conv_kwargs,
                compute_config=compute_config,
            )

            if in_channel_slice_id == 0:
                out_channel_slice_output = ttnn.to_memory_config(in_channel_slice_output, ttnn.DRAM_MEMORY_CONFIG)
            else:
                out_channel_slice_output = ttnn.add(
                    out_channel_slice_output, in_channel_slice_output, output_tensor=out_channel_slice_output
                )

            in_channel_slice_output.deallocate(True)

        if out_channel_slice_id == 0:
            output = ttnn.to_memory_config(out_channel_slice_output, ttnn.DRAM_MEMORY_CONFIG)
        else:
            output = ttnn.concat([output, out_channel_slice_output], dim=-1)

        out_channel_slice_output.deallocate(True)

    hidden_states = output
    hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
    hidden_states = ttnn.reshape(hidden_states, [1, output_height, output_width, out_channels])

    ttnn_output = ttnn.permute(hidden_states, [0, 3, 1, 2])
    result = ttnn.to_torch(ttnn_output)
    breakpoint()
    assert_with_pcc(torch_output, result, 0.99)
