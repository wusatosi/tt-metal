# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, torch_tensor_map, HS, WS, BS
import ttnn
import torch


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, shard_layout, config",
    (
        (353, 384, 8, 8, WS, None),
        (128, 128, 32, 32, BS, None),
        (16, 16, 256, 256, HS, {"act_block_h": 32}),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "fp32_accum",
    [True, False],
)
@pytest.mark.parametrize(
    "packer_l1_acc",
    [False],
)
@pytest.mark.parametrize(
    "filter, padding",
    [
        [3, (1, 2, 2, 3)],
        [1, 0],
        [5, (2, 4, 3, 5)],
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_conv_features(
    device,
    torch_tensor_map,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    shard_layout,
    config,
    filter,
    stride,
    padding,
    output_layout,
    fp32_accum,
    packer_l1_acc,
):
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and shard_layout == WS:
        pytest.skip("Bug in Width Sharded Row Major Tensor Creation when height%32!=0. #19408")

    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")

    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat16 and packer_l1_acc and fp32_accum:
        pytest.skip("skipping due to pack_untilize_dst issue!")

    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter,
        filter,
        stride,
        stride,
        padding,
        config,
        shard_layout=shard_layout,
        output_layout=output_layout,
        has_bias=True,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        preprocess_weights_on_device=True,
        run_twice=True,
    )


SliceHeight = ttnn.Conv2dSliceHeight
SliceWidth = ttnn.Conv2dSliceWidth


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, slice_type, num_slices, weights_dtype, activations_dtype, kernel, stride, padding, dilation, input_channels_alignment, act_block_h_override,  math_fidelity",
    # fmt: off
    (
        (2, 512,  512,  128,   128,   SliceWidth,    4,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1), 32, 32 * 8,  ttnn.MathFidelity.LoFi  ),
        (2, 64,   64,   384,   64,    SliceHeight,   6,  ttnn.bfloat8_b, ttnn.bfloat16, (4, 4), (2, 2), (1, 1), (1, 1), 32, 0,       ttnn.MathFidelity.LoFi  ),
        (1, 4,    32,   1024,  1024,  SliceWidth,    4,  ttnn.bfloat8_b, ttnn.bfloat16, (5, 5), (1, 1), (0, 0), (1, 1), 16, 32,      ttnn.MathFidelity.LoFi  ),
        (1, 64,   128,  992,   992,   SliceWidth,   64,  ttnn.bfloat8_b, ttnn.bfloat16, (2, 2), (1, 1), (0, 0), (1, 1), 32, 32 * 4,  ttnn.MathFidelity.LoFi  ),
    )
    # fmt: on
)
@pytest.mark.parametrize(
    "has_bias, fp32_accum, packer_l1_acc",
    [[True, False, False]],
)
def test_conv_dram(
    device,
    torch_tensor_map,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    has_bias,
    weights_dtype,
    activations_dtype,
    slice_type,
    num_slices,
    kernel,
    stride,
    padding,
    dilation,
    input_channels_alignment,
    act_block_h_override,
    math_fidelity,
    fp32_accum,
    packer_l1_acc,
):
    if device.core_grid.y == 7:
        pytest.skip("Tests have been configured for N150.")
    config = {
        "act_block_h": act_block_h_override,
    }
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        padding,
        config,
        has_bias=True,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        preprocess_weights_on_device=False,
        transpose_shards=True,
        run_twice=False,
        fast_compare=False,
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=slice_type,
            num_slices=num_slices,
        ),
    )


import torch
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout
from loguru import logger


def get_torch_act_func_from_string(act_string):
    act_func_map = {
        "relu": torch.nn.functional.relu,
        "silu": torch.nn.functional.silu,
        "mish": torch.nn.functional.mish,
        "sigmoid": torch.nn.functional.sigmoid,
        "sigmoid_approx": torch.nn.functional.sigmoid,
        "tanh": torch.nn.functional.tanh,
        "log": torch.log,
        "softplus": torch.nn.functional.softplus,
        "gelu": torch.nn.functional.gelu,
        "sqrt": torch.sqrt,
    }
    if act_string == "":
        return None
    if act_string in act_func_map:
        return act_func_map[act_string]
    raise RuntimeError(f"Activation function {act_string} not supported")


def run_conv_l1(
    device,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    use_1d_systolic_array,
    config_override,
    dilation=1,
    use_shallow_conv_variant=False,
    transpose_mcast=True,
    enable_auto_formatting=False,
    padded_input_channels=None,
    fp32_accum=False,
    packer_l1_acc=False,
    output_layout=ttnn.TILE_LAYOUT,
    deallocate_activation=False,
    debug=False,
    groups=1,
    bias=True,
    has_bias=True,
    shard_layout=None,
    auto_shard=False,
    memory_config=None,
    input_mesh_mapper=None,
    weight_mesh_mapper=None,
    output_mesh_composer=None,
    activation="",
):
    has_bias = bias

    total_batch_size = batch_size
    torch.manual_seed(0)
    conv_input_shape = [total_batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [output_channels, input_channels // groups, filter_height, filter_width]
    conv_bias_shape = [1, 1, 1, output_channels]
    torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
    torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float() if has_bias else None
    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1) if has_bias else None,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation, dilation),
        groups=groups,
    )
    act_func = get_torch_act_func_from_string(activation)
    if act_func:
        torch_out_golden_tensor = act_func(torch_out_golden_tensor)

    output_shape_nhwc = [
        torch_out_golden_tensor.shape[0],
        torch_out_golden_tensor.shape[2],
        torch_out_golden_tensor.shape[3],
        torch_out_golden_tensor.shape[1],
    ]
    reader_patterns_cache = {}
    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor,
        weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32,
        mesh_mapper=weight_mesh_mapper,
    )
    tt_bias_tensor = None
    if has_bias:
        tt_bias_tensor = ttnn.from_torch(
            torch_bias_tensor,
            weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32,
            mesh_mapper=weight_mesh_mapper,
        )
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        ttnn.bfloat16,
        mesh_mapper=input_mesh_mapper,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
    )
    if shard_layout is None and not auto_shard:
        shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if use_1d_systolic_array else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        shard_layout=shard_layout,
        input_channels_alignment=(
            16 if use_shallow_conv_variant or (input_channels == 16 and input_height == 115) else 32
        ),
        deallocate_activation=deallocate_activation,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        output_layout=output_layout,
        activation=activation,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=packer_l1_acc,
    )
    if config_override and "act_block_h" in config_override and not auto_shard:
        conv_config.act_block_h_override = config_override["act_block_h"]
    if config_override and "act_block_w_div" in config_override and not auto_shard:
        conv_config.act_block_w_div = config_override["act_block_w_div"]
    if config_override and "num_cores_nhw" in config_override:
        if config_override["num_cores_nhw"] == 98:
            conv_config.core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (11, 7)), ttnn.CoreRange((0, 8), (1, 8))})
            conv_config.override_sharding_config = True
            print("Setting num_cores_nhw to 98")

    [tt_output_tensor_on_device, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=(filter_height, filter_width),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation, dilation),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=groups,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        return_weights_and_bias=True,
        return_output_dim=True,
    )
    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor, mesh_composer=output_mesh_composer)
    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(
        total_batch_size, out_height, out_width, torch_output_tensor.shape[-1]
    )
    torch_output_tensor = torch_output_tensor[:, :, :, :output_channels]
    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))
    reader_patterns_cache.clear()
    if not fp32_accum:
        pcc = 0.985
    elif math_fidelity == ttnn.MathFidelity.LoFi and activations_dtype == ttnn.bfloat8_b:
        pcc = 0.996
    else:
        pcc = 0.997
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing
    if memory_config:
        output_memory_config = ttnn.get_memory_config(tt_output_tensor_on_device)
        logger.info(f"Output Memory Config : {output_memory_config}")
        assert output_memory_config == memory_config


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, use_1d_systolic_array, config_override, use_shallow_conv_variant, bias, activation",
    (
        (1, 64, 80, 80, 80, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, ""),  # Passed
        (1, 80, 320, 80, 80, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, ""),  # Passed
        (1, 64, 80, 40, 40, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, ""),  # Passed
        (1, 80, 320, 40, 40, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, ""),  # Passed
        (1, 64, 80, 20, 20, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, ""),  # Passed
        (1, 80, 320, 20, 20, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, ""),  # Passed
        (1, 1, 16, 4, 8400, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, ""),  # Passed
        (1, 160, 80, 320, 320, 3, 3, 2, 2, 1, 1, 1, 1, 1, True, {"act_block_h": 32}, False, False, "silu"),  # Passed
        (1, 80, 160, 160, 160, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 80, 80, 160, 160, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 80, 80, 160, 160, 3, 3, 1, 1, 1, 1, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 160, 160, 160, 160, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 320, 160, 160, 160, 3, 3, 2, 2, 1, 1, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 160, 320, 80, 80, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 160, 160, 80, 80, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 160, 160, 80, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 320, 320, 80, 80, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 640, 320, 80, 80, 3, 3, 2, 2, 1, 1, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 320, 640, 40, 40, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 320, 320, 40, 40, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 320, 320, 40, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 640, 640, 40, 40, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 1280, 640, 40, 40, 3, 3, 2, 2, 1, 1, 1, 1, 1, False, {"act_block_h": 32}, False, False, "silu"),  # Passed
        (1, 640, 1280, 20, 20, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 640, 640, 20, 20, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 640, 640, 20, 20, 3, 3, 1, 1, 1, 1, 1, 1, 1, False, {"act_block_h": 32}, False, False, "silu"),  # Passed
        (1, 1280, 1280, 20, 20, 1, 1, 1, 1, 0, 0, 1, 1, 1, False, {"act_block_h": 32}, False, False, "silu"),  # Passed
        (1, 1280, 2560, 20, 20, 1, 1, 1, 1, 0, 0, 1, 1, 1, False, {"act_block_h": 32}, False, False, "silu"),  # Passed
        (1, 320, 1280, 40, 40, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 160, 640, 80, 80, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 320, 320, 80, 80, 3, 3, 2, 2, 1, 1, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 640, 640, 40, 40, 3, 3, 2, 2, 1, 1, 1, 1, 1, False, {"act_block_h": 32}, False, False, "silu"),  # Passed
        (1, 80, 320, 80, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 80, 80, 80, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 320, 320, 80, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 80, 640, 40, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 80, 80, 40, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 320, 640, 40, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 80, 1280, 20, 20, 3, 3, 1, 1, 1, 1, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 80, 80, 20, 20, 3, 3, 1, 1, 1, 1, 1, 1, 1, True, None, False, False, "silu"),  # Passed
        (1, 320, 1280, 20, 20, 3, 3, 1, 1, 1, 1, 1, 1, 1, False, {"act_block_h": 32}, False, False, "silu"),  # Passed
        (1, 320, 320, 20, 20, 3, 3, 1, 1, 1, 1, 1, 1, 1, True, None, False, False, "silu"),  # Passed
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
def test_yolov5xu_640x640(
    device,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    groups,
    use_1d_systolic_array,
    config_override,
    use_shallow_conv_variant,
    bias,
    activation,
):
    run_conv_l1(
        device,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        use_1d_systolic_array,
        config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        dilation=dilation_h,
        groups=groups,
        bias=bias,
        activation=activation,
    )
