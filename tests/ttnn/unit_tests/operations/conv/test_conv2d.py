# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, torch_tensor_map, HS, WS, BS
import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout
from loguru import logger


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, shard_layout, config",
    (
        (256, 256, 8, 8, WS, None),
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


@pytest.fixture(scope="module")
def torch_tensor_map(request):
    torch_tensor_map = {}

    return torch_tensor_map


def randomize_torch_tensor(torch_tensor_map, tensor_shape):
    if tensor_shape in torch_tensor_map.keys():
        torch_tensor = torch_tensor_map[tensor_shape]
    else:
        torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16).float()
        torch_tensor_map[tensor_shape] = torch_tensor

    return torch_tensor


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
    torch_tensor_map,
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
    config_override,
    dilation=1,
    use_shallow_conv_variant=False,
    transpose_shards=True,  # https://github.com/tenstorrent/tt-metal/issues/17897
    fp32_accum=False,
    packer_l1_acc=False,
    output_layout=ttnn.TILE_LAYOUT,
    deallocate_activation=False,
    debug=False,
    groups=1,
    has_bias=True,
    shard_layout=None,
    auto_shard=False,
    memory_config=None,
    input_mesh_mapper=None,
    weight_mesh_mapper=None,
    output_mesh_composer=None,
    enable_split_reader=False,
    activation="",
    preprocess_weights_on_device=True,
):
    if isinstance(device, ttnn.MeshDevice):
        assert input_mesh_mapper is not None, "Expected mesh mapper for input tensor when using device mesh"
        assert weight_mesh_mapper is not None, "Expected mesh mapper for weight tensors when using device mesh"
        assert output_mesh_composer is not None, "Expected mesh composer for output tensor when using device mesh"
        num_devices = len(device.get_device_ids())
        total_batch_size = num_devices * batch_size  # Batch size across all devices
        logger.info(f"Using {num_devices} devices for this test")
    else:
        total_batch_size = batch_size

    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if (
        shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        and output_channels > 256
        and output_layout == ttnn.ROW_MAJOR_LAYOUT
    ):
        pytest.xfail(
            "Untilize_out is not supported when out_c > 256 for Height Sharded. https://github.com/tenstorrent/tt-metal/issues/18633"
        )

    torch.manual_seed(0)
    conv_input_shape = (total_batch_size, input_channels, input_height, input_width)
    conv_weight_shape = (output_channels, input_channels // groups, filter_height, filter_width)
    conv_bias_shape = (1, 1, 1, output_channels)
    torch_input_tensor_nchw = randomize_torch_tensor(torch_tensor_map, conv_input_shape)
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))

    torch_weight_tensor = randomize_torch_tensor(torch_tensor_map, conv_weight_shape)
    torch_bias_tensor = randomize_torch_tensor(torch_tensor_map, conv_bias_shape) * 10 if has_bias else None

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
        activations_dtype if activations_dtype == ttnn.float32 else ttnn.bfloat16,
        mesh_mapper=input_mesh_mapper,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
    )

    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        shard_layout=shard_layout if not auto_shard else None,
        input_channels_alignment=8 if use_shallow_conv_variant and not auto_shard else 32,
        deallocate_activation=deallocate_activation,
        enable_act_double_buffer=False,
        enable_split_reader=enable_split_reader,
        enable_subblock_padding=False,
        output_layout=output_layout,
        activation=activation,
        transpose_shards=transpose_shards,
        preprocess_weights_on_device=preprocess_weights_on_device,
        always_preprocess_weights=True,
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

    [tt_output_tensor_on_device, [out_height, out_width], [d_w, d_b]] = ttnn.conv2d(
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
        conv_op_cache=reader_patterns_cache,
        debug=debug,
        groups=groups,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        return_output_dim=True,
        return_weights_and_bias=True,
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
        if input_channels * filter_height * filter_width > 10000:
            pcc = 0.97
    elif math_fidelity == ttnn.MathFidelity.LoFi and activations_dtype == ttnn.bfloat8_b:
        pcc = 0.996
    else:
        pcc = 0.997

    if activation == "tanh":
        # Scale down PCC for tanh.
        # tanh has a range of -1 to 1. So discrepancies in output values which are close to 0 tend to disproportionately affect the PCC.
        pcc = pcc * 0.99

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing, pcc_msg

    if memory_config:
        output_memory_config = ttnn.get_memory_config(tt_output_tensor_on_device)
        logger.info(f"Output Memory Config : {output_memory_config}")
        assert output_memory_config == memory_config


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels,input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override, use_shallow_conv_variant, auto_shard",
    (
        (1, 64, 3, 640, 480, 3, 3, 2, 2, 1, 1, None, None, False, True),  # PASSED
        (1, 128, 64, 320, 240, 3, 3, 2, 2, 1, 1, None, None, False, True),  # PASSED
        (1, 64, 128, 160, 120, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 64, 64, 160, 120, 3, 3, 1, 1, 1, 1, None, None, False, True),  # PASSED
        (1, 128, 128, 160, 120, 1, 1, 1, 1, 0, 0, None, None, False, True)(  # PASSED
            1, 256, 128, 160, 120, 3, 3, 2, 2, 1, 1, None, None, False, True
        )(  # PASSED
            1, 128, 256, 80, 60, 1, 1, 1, 1, 0, 0, None, None, False, True
        ),  # PASSED
        (1, 128, 128, 80, 60, 3, 3, 1, 1, 1, 1, None, None, False, True),  # PASSED
        (1, 256, 256, 80, 60, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 512, 256, 80, 60, 3, 3, 2, 2, 1, 1, None, None, False, True),  # PASSED
        (1, 256, 512, 40, 30, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 256, 256, 40, 30, 3, 3, 1, 1, 1, 1, None, None, False, True),  # PASSED
        (1, 512, 512, 40, 30, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 1024, 512, 40, 30, 3, 3, 2, 2, 1, 1, None, None, False, True),  # PASSED
        (1, 512, 1024, 20, 15, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 512, 512, 20, 15, 3, 3, 1, 1, 1, 1, None, None, False, True),  # PASSED
        (1, 1024, 1024, 20, 15, 1, 1, 1, 1, 0, 0, None, None, False, True)(  # PASSED
            1, 1024, 2048, 20, 15, 1, 1, 1, 1, 0, 0, None, None, False, True
        )(  # PASSED
            1, 256, 1024, 20, 15, 1, 1, 1, 1, 0, 0, None, None, False, True
        ),  # PASSED
        (1, 256, 256, 80, 60, 3, 3, 2, 2, 1, 1, None, None, False, True),  # PASSED
        (1, 256, 768, 40, 30, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 128, 256, 40, 30, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 128, 128, 40, 30, 3, 3, 1, 1, 1, 1, None, None, False, True),  # PASSED
        (1, 256, 256, 40, 30, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 128, 128, 160, 120, 3, 3, 2, 2, 1, 1, None, None, False, True)(  # PASSED
            1, 128, 384, 80, 60, 1, 1, 1, 1, 0, 0, None, None, False, True
        ),  # PASSED
        (1, 64, 128, 80, 60, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 64, 64, 80, 60, 3, 3, 1, 1, 1, 1, None, None, False, True),  # PASSED
        (1, 128, 128, 80, 60, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 128, 128, 80, 60, 3, 3, 2, 2, 1, 1, None, None, False, True),  # PASSED
        (1, 256, 256, 40, 30, 3, 3, 2, 2, 1, 1, None, None, False, True),  # PASSED
        (1, 256, 512, 20, 15, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 256, 256, 20, 15, 3, 3, 1, 1, 1, 1, None, None, False, True),  # PASSED
        (1, 512, 512, 20, 15, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 80, 128, 80, 60, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 68, 128, 80, 60, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 1, 17, 4, 4800, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 80, 256, 40, 30, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 68, 256, 40, 30, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 1, 17, 4, 1200, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 80, 512, 20, 15, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 68, 512, 20, 15, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
        (1, 1, 17, 4, 300, 1, 1, 1, 1, 0, 0, None, None, False, True),  # PASSED
    ),
)
@pytest.mark.parametrize(
    "activation",
    ["relu", "silu"],
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
def test_conv_yolov6l(
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
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    activation,
    shard_layout,
    config_override,
    use_shallow_conv_variant,
    auto_shard,
):
    if device.core_grid.y == 7:
        pytest.skip("This test is not supported for N300")
    run_conv_l1(
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
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        dilation=1,
        shard_layout=shard_layout,
        use_shallow_conv_variant=use_shallow_conv_variant,
        groups=1,
        activation=activation,
        auto_shard=auto_shard,
    )
