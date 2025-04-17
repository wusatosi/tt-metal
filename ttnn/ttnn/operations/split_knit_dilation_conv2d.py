# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


# Split-Knit Conv2d with high dilation
def torch_split_knit_dilation(
    torch_input_tensor_nchw,
    torch_weight_tensor_oihw,
    torch_bias_tensor,
    filter_hw,
    stride_hw,
    padding_hw,
    dilation_hw,
    groups,
):
    assert groups == 1, "groups must be 1"
    assert all(d % 2 == 0 for d in dilation_hw), "dilation must be even"
    assert all(s == 1 for s in stride_hw), "stride must be 1"
    assert padding_hw[0] == 0 or padding_hw[0] == dilation_hw[0], "padding must be 0 or equal to dilation"
    assert padding_hw[1] == 0 or padding_hw[1] == dilation_hw[1], "padding must be 0 or equal to dilation"
    assert padding_hw[0] == padding_hw[1], "padding must be equal"

    assert torch_input_tensor_nchw.shape[2] % dilation_hw[0] == 0, "input height must be divisible by dilation"
    assert torch_input_tensor_nchw.shape[3] % dilation_hw[1] == 0, "input width must be divisible by dilation"

    # split
    inputs_splited = []
    for split_h in range(dilation_hw[0]):
        for split_w in range(dilation_hw[1]):
            inputs_splited.append(
                torch_input_tensor_nchw[
                    :,
                    :,
                    split_h :: dilation_hw[0],
                    split_w :: dilation_hw[1],
                ]
            )

    # conv2d
    sk_padding_hw = (1, 1) if padding_hw[0] > 0 else (0, 0)
    sk_dilation_hw = (1, 1)
    sk_groups = groups
    """batch, input_channels, output_channels, input_height, input_width, weights_dtype, activations_dtype, groups, kernel, stride, padding, dilation")"""
    print(
        f"# torch_split_knit_dilation: {len(inputs_splited)}x \n({inputs_splited[0].shape[0]}, {torch_weight_tensor_oihw.shape[1]}, {torch_weight_tensor_oihw.shape[0]}, {inputs_splited[0].shape[2]}, {inputs_splited[0].shape[3]}, ttnn.bfloat8_b, ttnn.bfloat8_b, {sk_groups}, ({torch_weight_tensor_oihw.shape[2]}, {torch_weight_tensor_oihw.shape[3]}), (1, 1), {sk_padding_hw}, {sk_dilation_hw}, True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),"
    )

    outs_splited = []
    for input_splited in inputs_splited:
        out_splited = torch.nn.functional.conv2d(
            input_splited,
            torch_weight_tensor_oihw,
            bias=torch_bias_tensor.reshape(-1) if torch_bias_tensor is not None else None,
            stride=stride_hw,
            padding=sk_padding_hw,
            dilation=sk_dilation_hw,
            groups=sk_groups,
        )
        outs_splited.append(out_splited)

    # knit
    out_h = (
        torch_input_tensor_nchw.shape[2] + 2 * padding_hw[0] - dilation_hw[0] * (filter_hw[0] - 1) - 1
    ) // stride_hw[0] + 1
    out_w = (
        torch_input_tensor_nchw.shape[3] + 2 * padding_hw[1] - dilation_hw[1] * (filter_hw[1] - 1) - 1
    ) // stride_hw[1] + 1
    out_knitted = torch.zeros((torch_input_tensor_nchw.shape[0], torch_weight_tensor_oihw.shape[0], out_h, out_w))

    for split_h in range(dilation_hw[0]):
        for split_w in range(dilation_hw[1]):
            out_knitted[:, :, split_h :: dilation_hw[0], split_w :: dilation_hw[1]] = outs_splited[
                split_h * dilation_hw[1] + split_w
            ]

    return out_knitted


# Split-Knit Conv2d with high dilation (batched)
def torch_split_knit_batched_dilation(
    torch_input_tensor_nchw,
    torch_weight_tensor_oihw,
    torch_bias_tensor,
    filter_hw,
    stride_hw,
    padding_hw,
    dilation_hw,
    groups,
):
    assert groups == 1, "groups must be 1"
    assert all(d % 2 == 0 for d in dilation_hw), "dilation must be even"
    assert all(s == 1 for s in stride_hw), "stride must be 1"
    assert padding_hw[0] == 0 or padding_hw[0] == dilation_hw[0], "padding must be 0 or equal to dilation"
    assert padding_hw[1] == 0 or padding_hw[1] == dilation_hw[1], "padding must be 0 or equal to dilation"
    assert padding_hw[0] == padding_hw[1], "padding must be equal"

    assert torch_input_tensor_nchw.shape[2] % dilation_hw[0] == 0, "input height must be divisible by dilation"
    assert torch_input_tensor_nchw.shape[3] % dilation_hw[1] == 0, "input width must be divisible by dilation"

    assert torch_input_tensor_nchw.shape[0] == 1, f"batch size must be 1"

    # split
    sk_batch = torch_input_tensor_nchw.shape[0] * dilation_hw[0] * dilation_hw[1]
    inputs_grouped_splited = torch.zeros(
        (
            sk_batch,
            torch_input_tensor_nchw.shape[1],
            torch_input_tensor_nchw.shape[2] // dilation_hw[0],
            torch_input_tensor_nchw.shape[3] // dilation_hw[1],
        )
    )
    for split_h in range(dilation_hw[0]):
        for split_w in range(dilation_hw[1]):
            batch_idx = split_h * dilation_hw[1] + split_w
            inputs_grouped_splited[batch_idx, :, :, :] = torch_input_tensor_nchw[
                :,
                :,
                split_h :: dilation_hw[0],
                split_w :: dilation_hw[1],
            ]

    sk_padding_hw = (1, 1) if padding_hw[0] > 0 else (0, 0)
    sk_dilation_hw = (1, 1)
    sk_groups = 1

    # conv2d
    out_splited = torch.nn.functional.conv2d(
        inputs_grouped_splited,
        torch_weight_tensor_oihw,
        # bias=torch_bias_tensor.reshape(-1).repeat(dilation_hw[0]*dilation_hw[1]) if torch_bias_tensor is not None else None, # TBD if this is OK
        bias=torch_bias_tensor.reshape(-1) if torch_bias_tensor is not None else None,  # TBD if this is OK
        stride=stride_hw,
        padding=sk_padding_hw,
        dilation=sk_dilation_hw,
        groups=sk_groups,
    )

    """batch, input_channels, output_channels, input_height, input_width, weights_dtype, activations_dtype, groups, kernel, stride, padding, dilation")"""
    print(
        f"# torch_split_knit_batched_dilation: 1x \n({inputs_grouped_splited.shape[0]}, {inputs_grouped_splited.shape[1]}, {out_splited.shape[1]}, {inputs_grouped_splited.shape[2]}, {inputs_grouped_splited.shape[3]}, ttnn.bfloat8_b, ttnn.bfloat8_b, {sk_groups}, ({torch_weight_tensor_oihw.shape[2]}, {torch_weight_tensor_oihw.shape[3]}), (1, 1), {sk_padding_hw}, {sk_dilation_hw}, True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),"
    )

    # knit
    out_h = (
        torch_input_tensor_nchw.shape[2] + 2 * padding_hw[0] - dilation_hw[0] * (filter_hw[0] - 1) - 1
    ) // stride_hw[0] + 1
    out_w = (
        torch_input_tensor_nchw.shape[3] + 2 * padding_hw[1] - dilation_hw[1] * (filter_hw[1] - 1) - 1
    ) // stride_hw[1] + 1
    out_knitted = torch.zeros((torch_input_tensor_nchw.shape[0], torch_weight_tensor_oihw.shape[0], out_h, out_w))

    # for out_channel in range(out_channels):
    for split_h in range(dilation_hw[0]):
        for split_w in range(dilation_hw[1]):
            src_batch_idx = split_h * dilation_hw[1] + split_w
            out_knitted[:, :, split_h :: dilation_hw[0], split_w :: dilation_hw[1]] = out_splited[
                src_batch_idx, :, :, :
            ]

    return out_knitted


# Split-Knit Conv2d with high dilation (batched)
@ttnn.register_python_operation(name="ttnn.experimental.split_knit_batch_dilation_conv2d")
def ttnn_split_knit_batched_dilation(
    device,
    torch_input_tensor_nchw,
    torch_weight_tensor_oihw,
    torch_bias_tensor,
    filter_hw,
    stride_hw,
    padding_hw,
    dilation_hw,
    groups,
):
    assert groups == 1, "groups must be 1"
    assert all(d % 2 == 0 for d in dilation_hw), "dilation must be even"
    assert all(s == 1 for s in stride_hw), "stride must be 1"
    assert padding_hw[0] == 0 or padding_hw[0] == dilation_hw[0], "padding must be 0 or equal to dilation"
    assert padding_hw[1] == 0 or padding_hw[1] == dilation_hw[1], "padding must be 0 or equal to dilation"
    assert padding_hw[0] == padding_hw[1], "padding must be equal"

    assert torch_input_tensor_nchw.shape[2] % dilation_hw[0] == 0, "input height must be divisible by dilation"
    assert torch_input_tensor_nchw.shape[3] % dilation_hw[1] == 0, "input width must be divisible by dilation"

    assert torch_input_tensor_nchw.shape[0] == 1, f"batch size must be 1"

    weights_dtype = ttnn.bfloat8_b
    activations_dtype = ttnn.bfloat8_b
    math_fidelity = ttnn.MathFidelity.LoFi
    fp32_dest_acc_en = False
    packer_l1_acc = False

    # split (host-side)
    sk_in_channels = torch_weight_tensor_oihw.shape[1]
    sk_out_channels = torch_weight_tensor_oihw.shape[0]
    sk_batch_size = torch_input_tensor_nchw.shape[0] * dilation_hw[0] * dilation_hw[1]
    sk_input_height = torch_input_tensor_nchw.shape[2] // dilation_hw[0]
    sk_input_width = torch_input_tensor_nchw.shape[3] // dilation_hw[1]
    print(
        f"sk_batch_size {sk_batch_size}, sk_in_channels {sk_in_channels}, sk_out_channels {sk_out_channels}, sk_input_height {sk_input_height}, sk_input_width {sk_input_width}"
    )
    torch_inputs_grouped_splited = torch.zeros(
        (
            sk_batch_size,
            torch_input_tensor_nchw.shape[1],
            torch_input_tensor_nchw.shape[2] // dilation_hw[0],
            torch_input_tensor_nchw.shape[3] // dilation_hw[1],
        )
    )
    for split_h in range(dilation_hw[0]):
        for split_w in range(dilation_hw[1]):
            batch_idx = split_h * dilation_hw[1] + split_w
            torch_inputs_grouped_splited[batch_idx, :, :, :] = torch_input_tensor_nchw[
                :,
                :,
                split_h :: dilation_hw[0],
                split_w :: dilation_hw[1],
            ]

    tt_input_tensor_nchw = ttnn.from_torch(
        torch_inputs_grouped_splited,
        activations_dtype if activations_dtype == ttnn.float32 else ttnn.bfloat16,
        mesh_mapper=None,
    ).to(device)

    tt_input_tensor = ttnn.permute(tt_input_tensor_nchw, (0, 2, 3, 1))

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor_oihw, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32, mesh_mapper=None
    )

    tt_bias_tensor = ttnn.from_torch(
        torch_bias_tensor,
        weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32,
        mesh_mapper=None,
    )

    sk_padding_hw = (1, 1) if padding_hw[0] > 0 else (0, 0)
    sk_dilation_hw = (1, 1)
    sk_groups = 1

    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        input_channels_alignment=32,
        deallocate_activation=True,
        enable_act_double_buffer=True,
        enable_split_reader=True,
        enable_subblock_padding=False,
        # output_layout=ttnn.ROW_MAJOR_LAYOUT,
        output_layout=ttnn.TILE_LAYOUT,
        activation="",
        act_block_h_override=32 * 2,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )

    # conv2d
    [tt_output_splited_tensor_on_device, [out_h, out_w]] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        device=device,
        in_channels=sk_in_channels,
        out_channels=sk_out_channels,
        batch_size=sk_batch_size,
        input_height=sk_input_height,
        input_width=sk_input_width,
        kernel_size=filter_hw,
        stride=stride_hw,
        padding=sk_padding_hw,
        dilation=sk_dilation_hw,
        groups=sk_groups,
        bias_tensor=tt_bias_tensor,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=False,
    )
    # out_splited = torch.nn.functional.conv2d(
    #     torch_inputs_grouped_splited,
    #     torch_weight_tensor_oihw,
    #     # bias=torch_bias_tensor.reshape(-1).repeat(dilation_hw[0]*dilation_hw[1]) if torch_bias_tensor is not None else None, # TBD if this is OK
    #     bias=torch_bias_tensor.reshape(-1) if torch_bias_tensor is not None else None, # TBD if this is OK
    #     stride=stride_hw,
    #     padding=sk_padding_hw,
    #     dilation=sk_dilation_hw,
    #     groups=sk_groups
    # )

    print(f"tt_output_splited_tensor_on_device.shape {tt_output_splited_tensor_on_device.shape}")
    tt_output_splited_tensor_on_device = tt_output_splited_tensor_on_device.reshape(
        sk_batch_size, out_h, out_w, tt_output_splited_tensor_on_device.shape[-1]
    )
    ttnn.synchronize_device(device)

    # host fallback
    out_splited = ttnn.to_torch(tt_output_splited_tensor_on_device, mesh_composer=None)
    out_splited = torch.permute(out_splited, (0, 3, 1, 2))

    # knit (HOST)
    full_out_h = (
        torch_input_tensor_nchw.shape[2] + 2 * padding_hw[0] - dilation_hw[0] * (filter_hw[0] - 1) - 1
    ) // stride_hw[0] + 1
    full_out_w = (
        torch_input_tensor_nchw.shape[3] + 2 * padding_hw[1] - dilation_hw[1] * (filter_hw[1] - 1) - 1
    ) // stride_hw[1] + 1
    out_knitted = torch.zeros(
        (torch_input_tensor_nchw.shape[0], torch_weight_tensor_oihw.shape[0], full_out_h, full_out_w)
    )

    # for out_channel in range(out_channels):
    for split_h in range(dilation_hw[0]):
        for split_w in range(dilation_hw[1]):
            src_batch_idx = split_h * dilation_hw[1] + split_w
            out_knitted[:, :, split_h :: dilation_hw[0], split_w :: dilation_hw[1]] = out_splited[
                src_batch_idx, :, :, :
            ]

    return out_knitted


ttnn.attach_golden_function(
    ttnn.conv2d,
    golden_function=torch_split_knit_batched_dilation,
)


# Split-Knit Conv2d with high dilation, grouped conv2d approach
def torch_split_knit_grouped_dilation(
    torch_input_tensor_nchw,
    torch_weight_tensor_oihw,
    torch_bias_tensor,
    filter_hw,
    stride_hw,
    padding_hw,
    dilation_hw,
    groups,
):
    assert groups == 1, "groups must be 1"
    assert all(d % 2 == 0 for d in dilation_hw), "dilation must be even"
    assert all(s == 1 for s in stride_hw), "stride must be 1"
    assert padding_hw[0] == 0 or padding_hw[0] == dilation_hw[0], "padding must be 0 or equal to dilation"
    assert padding_hw[1] == 0 or padding_hw[1] == dilation_hw[1], "padding must be 0 or equal to dilation"
    assert padding_hw[0] == padding_hw[1], "padding must be equal"

    assert torch_input_tensor_nchw.shape[2] % dilation_hw[0] == 0, "input height must be divisible by dilation"
    assert torch_input_tensor_nchw.shape[3] % dilation_hw[1] == 0, "input width must be divisible by dilation"

    assert torch_input_tensor_nchw.shape[0] == 1, f"batch size must be 1"

    in_channels = torch_input_tensor_nchw.shape[1]
    out_channels = torch_weight_tensor_oihw.shape[0]

    # split
    inputs_grouped_splited = torch.zeros(
        (
            torch_input_tensor_nchw.shape[0],
            dilation_hw[0] * dilation_hw[1] * torch_input_tensor_nchw.shape[1],
            torch_input_tensor_nchw.shape[2] // dilation_hw[0],
            torch_input_tensor_nchw.shape[3] // dilation_hw[1],
        )
    )
    torch_weight_tensor_oihw_grouped = torch.zeros(
        (
            dilation_hw[0] * dilation_hw[1] * torch_weight_tensor_oihw.shape[0],
            torch_weight_tensor_oihw.shape[1],
            torch_weight_tensor_oihw.shape[2],
            torch_weight_tensor_oihw.shape[3],
        )
    )
    for split_h in range(dilation_hw[0]):
        for split_w in range(dilation_hw[1]):
            group_idx = split_h * dilation_hw[1] + split_w
            inputs_grouped_splited[
                :, group_idx * in_channels : (group_idx + 1) * in_channels, :, :
            ] = torch_input_tensor_nchw[
                :,
                :,
                split_h :: dilation_hw[0],
                split_w :: dilation_hw[1],
            ]
            torch_weight_tensor_oihw_grouped[
                group_idx * out_channels : (group_idx + 1) * out_channels, :, :, :
            ] = torch_weight_tensor_oihw

    sk_padding_hw = (1, 1) if padding_hw[0] > 0 else (0, 0)
    sk_dilation_hw = (1, 1)
    sk_groups = dilation_hw[0] * dilation_hw[1]

    # conv2d
    out_splited = torch.nn.functional.conv2d(
        inputs_grouped_splited,
        torch_weight_tensor_oihw_grouped,
        bias=torch_bias_tensor.reshape(-1).repeat(dilation_hw[0] * dilation_hw[1])
        if torch_bias_tensor is not None
        else None,
        stride=stride_hw,
        padding=sk_padding_hw,
        dilation=sk_dilation_hw,
        groups=sk_groups,
    )

    """batch, input_channels, output_channels, input_height, input_width, weights_dtype, activations_dtype, groups, kernel, stride, padding, dilation")"""
    print(
        f"# torch_split_knit_grouped_dilation: 1x \n({inputs_grouped_splited.shape[0]}, {inputs_grouped_splited.shape[1]}, {out_splited.shape[1]}, {inputs_grouped_splited.shape[2]}, {inputs_grouped_splited.shape[3]}, ttnn.bfloat8_b, ttnn.bfloat8_b, {sk_groups}, ({torch_weight_tensor_oihw_grouped.shape[2]}, {torch_weight_tensor_oihw_grouped.shape[3]}), (1, 1), {sk_padding_hw}, {sk_dilation_hw}, True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),"
    )

    # knit
    out_h = (
        torch_input_tensor_nchw.shape[2] + 2 * padding_hw[0] - dilation_hw[0] * (filter_hw[0] - 1) - 1
    ) // stride_hw[0] + 1
    out_w = (
        torch_input_tensor_nchw.shape[3] + 2 * padding_hw[1] - dilation_hw[1] * (filter_hw[1] - 1) - 1
    ) // stride_hw[1] + 1
    out_knitted = torch.zeros((torch_input_tensor_nchw.shape[0], torch_weight_tensor_oihw.shape[0], out_h, out_w))

    # for out_channel in range(out_channels):
    for split_h in range(dilation_hw[0]):
        for split_w in range(dilation_hw[1]):
            src_group_idx = split_h * dilation_hw[1] + split_w
            out_knitted[:, :, split_h :: dilation_hw[0], split_w :: dilation_hw[1]] = out_splited[
                :, src_group_idx * out_channels : (src_group_idx + 1) * out_channels, :, :
            ]

    return out_knitted


# Split-Knit Conv2d with high dilation, height only
def torch_split_knit_dilation_h(
    torch_input_tensor_nchw,
    torch_weight_tensor_oihw,
    torch_bias_tensor,
    filter_hw,
    stride_hw,
    padding_hw,
    dilation_hw,
    groups,
):
    assert groups == 1, "groups must be 1"
    assert all(d % 2 == 0 for d in dilation_hw), "dilation must be even"
    assert all(s == 1 for s in stride_hw), "stride must be 1"
    assert padding_hw[0] == 0 or padding_hw[0] == dilation_hw[0], "padding must be 0 or equal to dilation"
    assert padding_hw[1] == 0 or padding_hw[1] == dilation_hw[1], "padding must be 0 or equal to dilation"
    assert padding_hw[0] == padding_hw[1], "padding must be equal"

    assert torch_input_tensor_nchw.shape[2] % dilation_hw[0] == 0, "input height must be divisible by dilation"

    # split
    inputs_splited = []
    for split_h in range(dilation_hw[0]):
        inputs_splited.append(
            torch_input_tensor_nchw[
                :,
                :,
                split_h :: dilation_hw[0],
                :,
            ]
        )

    # conv2d
    sk_padding_hw = (1, padding_hw[1]) if padding_hw[0] > 0 else (0, padding_hw[1])
    sk_dilation_hw = (1, dilation_hw[1])
    sk_groups = groups
    """batch, input_channels, output_channels, input_height, input_width, weights_dtype, activations_dtype, groups, kernel, stride, padding, dilation")"""
    print(
        f"# torch_split_knit_dilation_h: {len(inputs_splited)}x \n({inputs_splited[0].shape[0]}, {torch_weight_tensor_oihw.shape[1]}, {torch_weight_tensor_oihw.shape[0]}, {inputs_splited[0].shape[2]}, {inputs_splited[0].shape[3]}, ttnn.bfloat8_b, ttnn.bfloat8_b, {sk_groups}, ({torch_weight_tensor_oihw.shape[2]}, {torch_weight_tensor_oihw.shape[3]}), (1, 1), {sk_padding_hw}, {sk_dilation_hw}, True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),"
    )

    outs_splited = []
    for input_splited in inputs_splited:
        out_splited = torch.nn.functional.conv2d(
            input_splited,
            torch_weight_tensor_oihw,
            bias=torch_bias_tensor.reshape(-1) if torch_bias_tensor is not None else None,
            stride=stride_hw,
            padding=sk_padding_hw,
            dilation=sk_dilation_hw,
            groups=sk_groups,
        )
        outs_splited.append(out_splited)

    # knit
    out_h = (
        torch_input_tensor_nchw.shape[2] + 2 * padding_hw[0] - dilation_hw[0] * (filter_hw[0] - 1) - 1
    ) // stride_hw[0] + 1
    out_w = (
        torch_input_tensor_nchw.shape[3] + 2 * padding_hw[1] - dilation_hw[1] * (filter_hw[1] - 1) - 1
    ) // stride_hw[1] + 1
    out_knitted = torch.zeros((torch_input_tensor_nchw.shape[0], torch_weight_tensor_oihw.shape[0], out_h, out_w))

    for split_h in range(dilation_hw[0]):
        out_knitted[:, :, split_h :: dilation_hw[0], :] = outs_splited[split_h]

    return out_knitted


# Split-Knit Conv2d with high dilation, grouped conv2d approach, height-only
def torch_split_knit_grouped_dilation_h(
    torch_input_tensor_nchw,
    torch_weight_tensor_oihw,
    torch_bias_tensor,
    filter_hw,
    stride_hw,
    padding_hw,
    dilation_hw,
    groups,
):
    assert groups == 1, "groups must be 1"
    assert all(d % 2 == 0 for d in dilation_hw), "dilation must be even"
    assert all(s == 1 for s in stride_hw), "stride must be 1"

    assert padding_hw[0] == 0 or padding_hw[0] == dilation_hw[0], "padding must be 0 or equal to dilation"
    assert padding_hw[1] == 0 or padding_hw[1] == dilation_hw[1], "padding must be 0 or equal to dilation"
    assert padding_hw[0] == padding_hw[1], "padding must be equal"

    assert torch_input_tensor_nchw.shape[2] % dilation_hw[0] == 0, "input height must be divisible by dilation"
    assert torch_input_tensor_nchw.shape[3] % dilation_hw[1] == 0, "input width must be divisible by dilation"

    assert torch_input_tensor_nchw.shape[0] == 1, f"batch size must be 1"

    in_channels = torch_input_tensor_nchw.shape[1]
    out_channels = torch_weight_tensor_oihw.shape[0]

    # split
    inputs_grouped_splited = torch.zeros(
        (
            torch_input_tensor_nchw.shape[0],
            dilation_hw[0] * torch_input_tensor_nchw.shape[1],
            torch_input_tensor_nchw.shape[2] // dilation_hw[0],
            torch_input_tensor_nchw.shape[3],
        )
    )
    torch_weight_tensor_oihw_grouped = torch.zeros(
        (
            dilation_hw[0] * torch_weight_tensor_oihw.shape[0],
            torch_weight_tensor_oihw.shape[1],
            torch_weight_tensor_oihw.shape[2],
            torch_weight_tensor_oihw.shape[3],
        )
    )
    for split_h in range(dilation_hw[0]):
        inputs_grouped_splited[:, split_h * in_channels : (split_h + 1) * in_channels, :, :] = torch_input_tensor_nchw[
            :,
            :,
            split_h :: dilation_hw[0],
            :,
        ]
        torch_weight_tensor_oihw_grouped[
            split_h * out_channels : (split_h + 1) * out_channels, :, :, :
        ] = torch_weight_tensor_oihw

    # conv2d
    sk_padding_hw = (1, padding_hw[1]) if padding_hw[0] > 0 else (0, padding_hw[1])
    sk_dilation_hw = (1, dilation_hw[1])
    sk_groups = dilation_hw[0]
    out_splited = torch.nn.functional.conv2d(
        inputs_grouped_splited,
        torch_weight_tensor_oihw_grouped,
        bias=torch_bias_tensor.reshape(-1).repeat(dilation_hw[0]) if torch_bias_tensor is not None else None,
        stride=stride_hw,
        padding=sk_padding_hw,
        dilation=sk_dilation_hw,
        groups=sk_groups,
    )

    """batch, input_channels, output_channels, input_height, input_width, weights_dtype, activations_dtype, groups, kernel, stride, padding, dilation")"""
    print(
        f"# torch_split_knit_grouped_dilation: 1x \n({inputs_grouped_splited.shape[0]}, {inputs_grouped_splited.shape[1]}, {out_splited.shape[1]}, {inputs_grouped_splited.shape[2]}, {inputs_grouped_splited.shape[3]}, ttnn.bfloat8_b, ttnn.bfloat8_b, {sk_groups}, ({torch_weight_tensor_oihw_grouped.shape[2]}, {torch_weight_tensor_oihw_grouped.shape[3]}), (1, 1), {sk_padding_hw}, {sk_dilation_hw}, True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),"
    )

    # knit
    out_h = (
        torch_input_tensor_nchw.shape[2] + 2 * padding_hw[0] - dilation_hw[0] * (filter_hw[0] - 1) - 1
    ) // stride_hw[0] + 1
    out_w = (
        torch_input_tensor_nchw.shape[3] + 2 * padding_hw[1] - dilation_hw[1] * (filter_hw[1] - 1) - 1
    ) // stride_hw[1] + 1
    out_knitted = torch.zeros((torch_input_tensor_nchw.shape[0], torch_weight_tensor_oihw.shape[0], out_h, out_w))

    for split_h in range(dilation_hw[0]):
        out_knitted[:, :, split_h :: dilation_hw[0], :] = out_splited[
            :, split_h * out_channels : (split_h + 1) * out_channels, :, :
        ]

    return out_knitted
