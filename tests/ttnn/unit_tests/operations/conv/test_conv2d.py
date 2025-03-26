# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
from models.utility_functions import (
    is_grayskull,
    is_wormhole_b0,
    is_blackhole,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc, check_with_pcc_without_tensor_printout
import ttnn

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
BS = ttnn.TensorMemoryLayout.BLOCK_SHARDED
WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED


# Cache map used for torch tensor reuse - the tensor will not be generated if a tensor of the same dimensions has already been generated
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


def run_conv(
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
    dilation_hw=(1, 1),
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
    enable_act_double_buffer=False,
    activation="",
    preprocess_weights_on_device=True,
    run_twice=False,
    input_memory_config=None,
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
        dilation=dilation_hw,
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
        memory_config=input_memory_config,
    )

    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        shard_layout=shard_layout if not auto_shard else None,
        input_channels_alignment=8 if use_shallow_conv_variant and not auto_shard else 32,
        deallocate_activation=deallocate_activation,
        enable_act_double_buffer=enable_act_double_buffer,
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
        dilation=dilation_hw,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache=reader_patterns_cache,
        debug=debug,
        groups=groups,
        memory_config=memory_config,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    if run_twice:
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
            dilation=dilation_hw,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=conv_config,
            compute_config=compute_config,
            conv_op_cache=reader_patterns_cache,
            debug=debug,
            groups=groups,
            memory_config=memory_config,
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


def run_conv_with_split(
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
    shard_layout=None,
    split_factor=2,
    fp32_accum=False,
    packer_l1_acc=False,
    auto_shard=False,
):
    torch.manual_seed(0)
    assert input_channels % split_factor == 0
    split_input_channels = input_channels // split_factor
    full_conv_input_shape = (batch_size, input_channels, input_height, input_width)
    full_conv_weight_shape = (output_channels, input_channels, filter_height, filter_width)
    torch_input_tensor_nchw = randomize_torch_tensor(torch_tensor_map, full_conv_input_shape)
    torch_weight_tensor = randomize_torch_tensor(torch_tensor_map, full_conv_weight_shape)
    conv_bias_shape = (1, 1, 1, output_channels)
    torch_bias_tensor = randomize_torch_tensor(torch_tensor_map, conv_bias_shape)
    torch_bias_zeroes_tensor = randomize_torch_tensor(torch_tensor_map, conv_bias_shape)
    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
    )

    split_input_tensors = torch.split(torch_input_tensor_nchw, split_input_channels, 1)
    split_weight_tensors = torch.split(torch_weight_tensor, split_input_channels, 1)

    reader_patterns_cache = {}

    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        shard_layout=shard_layout if not auto_shard else None,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=packer_l1_acc,
    )
    if config_override and "act_block_h" in config_override:
        conv_config.act_block_h_override = config_override["act_block_h"]
        print("Setting Act Block H to ", conv_config.act_block_h_override)
    torch_output_tensor = None
    for i in range(split_factor):
        tt_weight_tensor = ttnn.from_torch(
            split_weight_tensors[i], weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )
        if i == 0:
            tt_bias_tensor = ttnn.from_torch(
                torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
            )
        else:
            tt_bias_tensor = ttnn.from_torch(
                torch_bias_zeroes_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
            )
        torch_input_tensor = torch.permute(split_input_tensors[i], (0, 2, 3, 1))
        tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)
        # tt_input_tensor_on_device = convs[i].copy_input_to_device(tt_input_tensor)
        # tt_output_tensor_on_device = convs[i](tt_input_tensor_on_device)
        [tt_output_tensor_on_device, [out_height, out_width]] = ttnn.conv2d(
            input_tensor=tt_input_tensor,
            weight_tensor=tt_weight_tensor,
            in_channels=split_input_channels,
            out_channels=output_channels,
            device=device,
            bias_tensor=tt_bias_tensor,
            kernel_size=(filter_height, filter_width),
            stride=(stride_h, stride_w),
            padding=(pad_h, pad_w),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=conv_config,
            compute_config=compute_config,
            conv_op_cache=reader_patterns_cache,
            return_output_dim=True,
        )
        tt_conv_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
        torch_conv_output_tensor = ttnn.to_torch(tt_conv_output_tensor)
        print(f"Output shape : {batch_size} {out_height} {out_width} {output_channels}")
        torch_conv_output_tensor = torch_conv_output_tensor.reshape(batch_size, out_height, out_width, output_channels)

        # torch_output_tensor is in row major layout and NHWC shape
        # NHWC to NCHW
        torch_conv_output_tensor = torch.permute(torch_conv_output_tensor, (0, 3, 1, 2))
        if i == 0:
            torch_output_tensor = torch_conv_output_tensor
        else:
            torch_output_tensor = torch.add(torch_output_tensor, torch_conv_output_tensor)
        print("Split output shapes ", torch_output_tensor.shape, torch_conv_output_tensor.shape)

    if math_fidelity == ttnn.MathFidelity.LoFi and activations_dtype == ttnn.bfloat8_b:
        pcc = 0.9969
    else:
        pcc = 0.998
    assert_with_pcc(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)


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
    "filter, pad",
    [
        [3, 1],
        [1, 0],
        [5, 2],
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
    pad,
    output_layout,
    fp32_accum,
    packer_l1_acc,
):
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
        pad,
        pad,
        config,
        shard_layout=shard_layout,
        output_layout=output_layout,
        has_bias=True,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        preprocess_weights_on_device=True,
        run_twice=True,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 2 * 16384}], indirect=True)
@pytest.mark.parametrize("groups", [1, 2])
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
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "filter, pad",
    [
        [3, 1],
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_conv_features_multi_device(
    mesh_device,
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
    pad,
    output_layout,
    groups,
):
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")

    run_conv(
        mesh_device,
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
        pad,
        pad,
        config,
        shard_layout=shard_layout,
        output_layout=output_layout,
        has_bias=True,
        input_mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        weight_mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        output_mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        groups=groups,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, shard_layout, config",
    (
        (256, 256, 8, 8, WS, None),
        (128, 128, 32, 32, BS, None),
        (32, 32, 256, 256, HS, {"act_block_h": 32}),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype, output_layout",
    [
        [ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.bfloat8_b, ttnn.TILE_LAYOUT],
    ],
)
@pytest.mark.parametrize(
    "fp32_accum",
    [True],
)
@pytest.mark.parametrize(
    "has_bias",
    [True],
)
@pytest.mark.parametrize(
    "filter, pad",
    [
        [3, 1],
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4])
@pytest.mark.parametrize("activation", ["", "relu", "silu", "sigmoid", "tanh", "sqrt", "gelu"])
def test_conv_activation(
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
    pad,
    output_layout,
    fp32_accum,
    has_bias,
    activation,
):
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")

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
        pad,
        pad,
        config,
        shard_layout=shard_layout,
        output_layout=output_layout,
        has_bias=has_bias,
        fp32_accum=fp32_accum,
        packer_l1_acc=False,
        activation=activation,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, pad_h, pad_w, act_block_w_div",
    (
        (2, 128, 256, 9, 9, 3, 3, 1, 1, 1),
        (2, 576, 576, 9, 9, 3, 3, 0, 0, 1),
        (2, 960, 960, 5, 5, 3, 3, 0, 0, 1),
        (2, 256, 2048, 9, 9, 3, 3, 1, 1, 1),
        (2, 512, 2048, 17, 17, 3, 3, 1, 1, 1),
        (2, 768, 768, 17, 17, 3, 3, 0, 0, 1),
        (2, 1280, 2560, 15, 15, 3, 3, 1, 1, 1),
        (2, 1280, 1280, 17, 17, 3, 3, 1, 1, 1),
        [1, 3024, 1232, 14, 14, 1, 1, 0, 0, 1],
        (2, 768, 32, 9, 9, 3, 3, 1, 1, 1),
        (2, 64, 128, 9, 9, 3, 3, 1, 1, 1),
        (2, 32, 128, 9, 9, 3, 3, 1, 1, 1),
    ),
)
@pytest.mark.parametrize(
    "has_bias",
    [True],
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
@pytest.mark.parametrize("tilized_input", [True, False], ids=["tilized", "row_major"])
def test_conv_ws(
    device,
    torch_tensor_map,
    use_program_cache,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    pad_h,
    pad_w,
    act_block_w_div,
    stride,
    has_bias,
    weights_dtype,
    activations_dtype,
    auto_shard,
    tilized_input,
):
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 grid for wormhole_b0")

    stride_h = stride
    stride_w = stride
    fp32_accum = True
    packer_l1_acc = True
    deallocate_activation = False
    debug = False
    groups = 1

    conv_input_shape = (batch_size, input_channels, input_height, input_width)
    conv_weight_shape = (output_channels, input_channels // groups, filter_height, filter_width)
    conv_bias_shape = (1, 1, 1, output_channels)

    torch_input_tensor_nchw = randomize_torch_tensor(torch_tensor_map, conv_input_shape)
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))

    torch_weight_tensor = randomize_torch_tensor(torch_tensor_map, conv_weight_shape)

    tt_bias_tensor = None
    torch_bias_tensor = None
    if has_bias:
        torch_bias_tensor = randomize_torch_tensor(torch_tensor_map, conv_bias_shape) * 50
        tt_bias_tensor = ttnn.from_torch(
            torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )
        torch_bias_tensor = torch_bias_tensor.reshape(-1)
    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        groups=groups,
    )

    reader_patterns_cache = {}
    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16)

    tt_input_tensor = ttnn.reshape(tt_input_tensor, [1, 1, input_height * input_width * batch_size, input_channels])
    if tilized_input:
        tt_input_tensor = ttnn.to_layout(tt_input_tensor, ttnn.TILE_LAYOUT)

    if auto_shard and (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        if input_channels == 2048:
            pytest.skip("Test is not supported on n300 (8,7) grid due to #13541")

    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED if not auto_shard else None,
        input_channels_alignment=32,
        deallocate_activation=deallocate_activation,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        reshard_if_not_optimal=True,
        act_block_w_div=act_block_w_div if not auto_shard else 1,
        act_block_h_override=32,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=packer_l1_acc,
    )
    [tt_output_tensor_on_device, [out_height, out_width]] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=(filter_height, filter_width),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache=reader_patterns_cache,
        debug=debug,
        groups=groups,
        return_output_dim=True,
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    # torch_output_tensor = torch_output_tensor[:, :, : batch_size * out_height * out_width, :]
    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_height, out_width, output_channels)
    logger.info(f"Output Shape : {torch_output_tensor.shape}")
    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))
    reader_patterns_cache.clear()

    pcc = 0.99
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)
    logger.info(f"{pcc_msg} Threshold : {pcc}")
    if not passing:
        logger.error("Fails with PCC ", pcc_msg)
    assert passing


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, groups, shard_layout, config_override, use_shallow_conv_variant",
    (
        # mlp sub_module
        (1, 3, 32, 512, 512, 7, 7, 4, 4, 3, 3, 1, HS, {"act_block_h": 64}, False),  # ncrisc build failed
        # efficient selfattention sub_module
        (1, 32, 32, 128, 128, 8, 8, 8, 8, 0, 0, 1, HS, None, False),  # ncrisc build failed, Two times called in model
        (1, 64, 64, 64, 64, 4, 4, 4, 4, 0, 0, 1, HS, None, False),  # ncrisc build failed, Two times called in model
        (1, 160, 160, 32, 32, 2, 2, 2, 2, 0, 0, 1, HS, None, False),  # pass , Two times called in model
        # dwconv sub_module
        (1, 128, 128, 128, 128, 3, 3, 1, 1, 1, 1, 128, HS, {"act_block_h": 64}, False),
        (1, 256, 256, 64, 64, 3, 3, 1, 1, 1, 1, 256, HS, None, False),  # pass , Two times called in model
        (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, 640, ttnn.TensorMemoryLayout.BLOCK_SHARDED, {"act_block_h": 32}, False),
        # (1,1024, 1024, 16, 16, 3, 3, 1, 1, 1, 1, 1024, BS, None, False), #Switch to Width Sharding
        # decode_head sub_module
        # (1,1024, 256, 128, 128, 1, 1, 1, 1, 0, 0, 1, BS, {"act_block_h": 32}, False), #pass for activation_dtype=bf8 but fails for bf16
        (1, 256, 150, 128, 128, 1, 1, 1, 1, 0, 0, 1, HS, None, False),
        (1, 32, 16, 64, 64, 1, 1, 1, 1, 0, 0, 1, HS, None, False),
        (1, 96, 24, 32, 32, 1, 1, 1, 1, 0, 0, 1, HS, None, False),
        (1, 576, 576, 8, 8, 3, 3, 1, 1, 0, 0, 576, WS, None, False),
        (1, 576, 576, 8, 8, 3, 3, 2, 2, 0, 0, 576, WS, None, False),
        (1, 960, 960, 4, 4, 3, 3, 1, 1, 0, 0, 960, WS, None, False),
        (1, 144, 24, 32, 32, 1, 1, 1, 1, 0, 0, 1, HS, None, False),
        (1, 144, 32, 16, 16, 1, 1, 1, 1, 0, 0, 1, HS, None, False),
    ),
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
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_conv_for_segformer_512x512(
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
    shard_layout,
    config_override,
    use_shallow_conv_variant,
    groups,
    output_layout,
    auto_shard,
):
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
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        groups=groups,
        output_layout=output_layout,
        has_bias=False,
        auto_shard=auto_shard,
        shard_layout=shard_layout,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override",
    (
        # unique convs in rn50 (complete list)
        # first conv post folding and input_channels padding to tile width
        # (8, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, True, None), HANGS!!
        (16, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, HS, {"act_block_h": 256}),
        # (20, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, HS, {"act_block_h": 32}),  Out of Memory!!
        # rn50 layer1
        (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
        (16, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
        (20, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
        # rn50 layer2
        (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, HS, None),
        (16, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, HS, None),
        (20, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, HS, {"act_block_h": 32}),
        (8, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, HS, None),
        (16, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, HS, None),
        (20, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, HS, None),
        # rn50 layer3
        (8, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, BS, None),
        (16, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, BS, None),
        (20, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, BS, None),
        (8, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, BS, None),
        (16, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, BS, None),
        (20, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, BS, None),
        # rn50 layer4
        (8, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, BS, None),
        (16, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, BS, None),
        (20, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, BS, None),
        (8, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
        (16, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
        (20, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
        ## small test
        (1, 64, 64, 8, 8, 3, 3, 1, 1, 1, 1, BS, {"num_cores_nhw": 2, "grid_size": (2, 2)}),
        (1, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, BS, {"num_cores_nhw": 4, "grid_size": (2, 4)}),
        # (1, 160, 160, 7, 7, 3, 3, 1, 1, 1, 1, BS, None), sliding_window_op_infra/sliding_window.cpp:341: indices_length_last_core <= indices_length_per_core
        (8, 256, 256, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
        # r50 1x1s2 shapes
        # Fails with packer_l1_acc = True (20, 256, 64, 56, 56, 1, 1, 2, 2, 0, 0, BS, None),  # r50 first bottleneck downsample shape
        (20, 256, 64, 56, 56, 1, 1, 2, 2, 0, 0, HS, None),  # r50 first bottleneck downsample shape
        # Fails with packer_l1_acc = True (20, 512, 256, 56, 56, 1, 1, 2, 2, 0, 0, BS, None),  # r50 second bottleneck downsample shape
        # (20, 512, 256, 56, 56, 1, 1, 2, 2, 0, 0, HS, None), - doesnt fit
        (20, 1024, 512, 28, 28, 1, 1, 2, 2, 0, 0, BS, None),  # r50 third bottleneck downsample shape
        # (20, 1024, 512, 28, 28, 1, 1, 2, 2, 0, 0, HS, None), - doesnt fit
        (20, 2048, 1024, 14, 14, 1, 1, 2, 2, 0, 0, BS, None),  # r50 fourth bottleneck downsample shape
        # (20, 2048, 1024, 14, 14, 1, 1, 2, 2, 0, 0, HS, None), - doesnt fit
        # (20, 128, 256, 56, 56, 1, 1, 2, 2, 0, 0, HS, None),  ## L2M1 DS: doesn't fit
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("packer_l1_acc", [True])
@pytest.mark.parametrize("has_bias", [True])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_resnet50_conv_wh(
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
    shard_layout,
    config_override,
    packer_l1_acc,
    has_bias,
    auto_shard,
):
    if device.core_grid.y == 7:
        pytest.skip("Issue #6992: Statically allocated circular buffers in program clash with L1 buffers on core range")

    use_shallow_conv_variant = (input_channels == 16) and device.arch() == ttnn.device.Arch.GRAYSKULL
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
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override=config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        transpose_shards=True,  ## use RM (transpose_mcast=False) with 2D on WH
        packer_l1_acc=packer_l1_acc,
        fp32_accum=False,
        has_bias=has_bias,
        auto_shard=auto_shard,
        shard_layout=shard_layout,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override",
    (
        (16, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, HS, {"act_block_h": 256}),
        (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
    ),
)
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
def test_conv_mem_config_wh(
    device,
    torch_tensor_map,
    use_program_cache,
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
    shard_layout,
    config_override,
    memory_config,
):
    if device.core_grid.y == 7:
        pytest.skip("Issue #6992: Statically allocated circular buffers in program clash with L1 buffers on core range")

    use_shallow_conv_variant = (input_channels == 16) and device.arch() != ttnn.device.Arch.WORMHOLE_B0
    run_conv(
        device,
        torch_tensor_map,
        ttnn.MathFidelity.LoFi,
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
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
        shard_layout=shard_layout,
        config_override=config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        transpose_shards=True,  ## use RM (transpose_mcast=False) with 2D on WH
        packer_l1_acc=True,
        fp32_accum=False,
        has_bias=True,
        auto_shard=False,
        memory_config=memory_config,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override",
    (
        # unique convs in rn50 (complete list)
        # first conv post folding and input_channels padding to tile width
        # (8, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, HS, None),
        # (16, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, HS, {"act_block_h": 32}),
        # (20, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, HS, {"act_block_h": 32}),
        # rn50 layer1
        (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
        # (16, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
        # (20, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
        # # rn50 layer2
        (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, HS, None),
        # (16, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, HS, None),
        # (20, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, HS, {"act_block_h": 32}),
        (8, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, HS, None),
        # (16, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, HS, None),
        # (20, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, HS, None),
        # # rn50 layer3
        # (8, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, BS, None),
        # (16, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, BS, None),
        # (20, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, BS, None),
        # (8, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, BS, None),
        # (16, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, BS, None),
        # (20, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, BS, None),
        # # rn50 layer4
        # (8, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, BS, None),
        # (16, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, BS, None),
        # (20, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, BS, None),
        # (8, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
        # (16, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
        # (20, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.float32, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.float32, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "fp32_accum",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4])
@pytest.mark.parametrize("packer_l1_acc", [True])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_resnet50_conv_wh_fp32(
    device,
    torch_tensor_map,
    use_program_cache,
    math_fidelity,
    fp32_accum,
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
    shard_layout,
    config_override,
    packer_l1_acc,
    auto_shard,
):
    if batch_size > 8 and (activations_dtype != ttnn.bfloat8_b or weights_dtype != ttnn.bfloat8_b):
        pytest.skip("Batch > 8 must be run fully bfp8")

    if (
        activations_dtype == ttnn.bfloat16
        and batch_size == 20
        and (
            output_channels == 64
            or (
                stride_h == 2
                and (output_channels == 256 or (output_channels == 128 and weights_dtype == ttnn.bfloat16))
            )
        )
    ):
        pytest.skip("Skipping test because it won't fit in L1!")

    use_shallow_conv_variant = (input_channels == 16) and device.arch() != ttnn.device.Arch.WORMHOLE_B0
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
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        shard_layout=shard_layout,
        config_override=config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        auto_shard=auto_shard,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override",
    (
        # sd convs with HxW=32x32
        # (1, 320, 320, 32, 32, 3, 3, 1, 1, 1, 1, BS, None),
        # (1, 320, 320, 32, 32, 3, 3, 2, 2, 1, 1, BS, None),
        # (1, 640, 640, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),
        # (1, 640, 640, 16, 16, 3, 3, 2, 2, 1, 1, BS, None),
        # (1, 640, 640, 16, 16, 3, 3, 2, 2, 1, 1, BS, None), # bfloat16 activations doesnt fit
        # (1, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, BS, None), # slighlty low pcc with 0.99689. bfloat16 weights doesnt fit
        # (1, 1280, 1280, 8, 8, 3, 3, 2, 2, 1, 1, BS, None), #fails to parallelize with sharding
        # (1, 1280, 1280, 4, 4, 3, 3, 1, 1, 1, 1, BS, None), #fails to parallelize with sharding
        # (1, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, BS, None), # slightly low pcc with 0.99698. bfloat16 weights doesnt fit
        # (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, BS, None), # doesnt fit at all.. for all data types
        # sd convs with HxW=64x64 with batch size = 1
        (1, 320, 16, 64, 64, 3, 3, 1, 1, 1, 1, HS, None),
        (1, 320, 320, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),  # bfloat16 doesnt fit
        (1, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, BS, None),
        (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),  #
        (1, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, BS, None),  # bfloat16 doesnt fit
        (1, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),  # bfloat16 weights doesnt fit
        (1, 1280, 1280, 16, 16, 3, 3, 2, 2, 1, 1, BS, None),  # bfloat16 doesnt fit.
        (1, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, BS, None),  # bfloat16 weights doesnt fit
        # (1, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, BS, None), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        (1, 640, 640, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        # (1, 1280, 2560, 8, 8, 3, 3, 1, 1, 1, 1, BS, None), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (1, 1280, 2560, 16, 16, 3, 3, 1, 1, 1, 1, BS, None), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # sd convs with HxW=64x64 with batch size=2
        # (2, 320, 16, 64, 64, 3, 3, 1, 1, 1, 1, HS, None), Hangs on WH
        (2, 320, 320, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 64}),
        (2, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, BS, None),  # fits with bfloat8_b
        (2, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 64}),
        (2, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, BS, None),  # bfloat16 doesnt fit
        (2, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),  # bfloat16 doesnt fit
        (2, 1280, 1280, 16, 16, 3, 3, 2, 2, 1, 1, BS, {"act_block_h": 32}),  # bfloat16 doesnt fit
        (2, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        # (2, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        (2, 640, 640, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 64}),
        # (2, 1280, 2560, 8, 8, 3, 3, 1, 1, 1, 1, BS, None), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 1280, 2560, 16, 16, 3, 3, 1, 1, 1, 1, BS, None), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 1280, 1920, 16, 16, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 640, 1920, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 640, 1280, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 640, 960, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 320, 960, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 320, 640, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # 1x1 conv
        (2, 320, 960, 64, 64, 1, 1, 1, 1, 0, 0, HS, None),
        # Small conv
        # (1, 32, 32, 16, 16, 3, 3, 2, 2, 1, 1, HS, None),  ## batch = 1 is currently not supported
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
@pytest.mark.parametrize("enable_auto_formatting", [True, False])
# Some tests fail with auto_shard on grayskull
@pytest.mark.parametrize("auto_shard", [False], ids=["no_auto_shard"])
def test_sd_conv(
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
    shard_layout,
    config_override,
    enable_auto_formatting,
    auto_shard,
):
    if filter_height > 1 and (input_channels > 1280 or (input_channels > 640 and input_height > 16)):
        if enable_auto_formatting:
            pytest.skip("Not running split SD conv with auto formatting")
        run_conv_with_split(
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
            shard_layout=shard_layout,
            split_factor=3 if input_channels == 1920 else 2,
        )
    else:
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
            filter_height,
            filter_width,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            config_override,
            shard_layout=shard_layout,
            use_shallow_conv_variant=(input_channels == 16),
            auto_shard=auto_shard,
        )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override",
    (
        # sd convs with HxW=32x32
        # (1, 320, 320, 32, 32, 3, 3, 1, 1, 1, 1, BS, None),
        # (1, 320, 320, 32, 32, 3, 3, 2, 2, 1, 1, BS, None),
        # (1, 640, 640, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),
        # (1, 640, 640, 16, 16, 3, 3, 2, 2, 1, 1, BS, None),
        # (1, 640, 640, 16, 16, 3, 3, 2, 2, 1, 1, BS, None), # bfloat16 activations doesnt fit
        # (1, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, BS, None), # slighlty low pcc with 0.99689. bfloat16 weights doesnt fit
        # (1, 1280, 1280, 8, 8, 3, 3, 2, 2, 1, 1, BS, None), #fails to parallelize with sharding
        # (1, 1280, 1280, 4, 4, 3, 3, 1, 1, 1, 1, BS, None), #fails to parallelize with sharding
        # (1, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, BS, None), # slightly low pcc with 0.99698. bfloat16 weights doesnt fit
        # (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, BS, None), # doesnt fit at all.. for all data types
        # sd convs with HxW=64x64 with batch size = 1
        # (1, 320, 16, 64, 64, 3, 3, 1, 1, 1, 1, HS, None),
        # (1, 320, 320, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),  # bfloat16 doesnt fit
        # (1, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, BS, None),
        # (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),  #
        # (1, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, BS, None),  # bfloat16 doesnt fit
        # (1, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),  # bfloat16 weights doesnt fit
        # (1, 1280, 1280, 16, 16, 3, 3, 2, 2, 1, 1, BS, None),  # bfloat16 doesnt fit.
        # (1, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, BS, None),  # bfloat16 weights doesnt fit
        # (1, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, BS, None),
        # (1, 640, 640, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        # (1, 1280, 2560, 8, 8, 3, 3, 1, 1, 1, 1, BS, None),
        # (1, 1280, 2560, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),
        # # sd convs with HxW=64x64 with batch size=2
        (2, 320, 16, 64, 64, 3, 3, 1, 1, 1, 1, HS, None),
        (2, 320, 320, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 64}),
        (2, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, BS, None),  # fits with bfloat8_b
        (2, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, BS, None),  # bfloat16 doesnt fit
        (2, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),  # bfloat16 doesnt fit
        (2, 1280, 1280, 16, 16, 3, 3, 2, 2, 1, 1, BS, {"act_block_h": 32}),  # bfloat16 doesnt fit
        (2, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),  # bfloat16 doesnt fit
        # (2, 640, 640, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}), L1 Allocation Error
        (2, 1280, 2560, 8, 8, 3, 3, 1, 1, 1, 1, BS, None),
        (2, 1280, 2560, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),
        (2, 1280, 1920, 16, 16, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 640, 1920, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 640, 1280, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 640, 960, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 320, 960, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 320, 640, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        # 1x1 conv
        (2, 320, 960, 64, 64, 1, 1, 1, 1, 0, 0, HS, None),
        # Small conv
        # (1, 32, 32, 16, 16, 3, 3, 2, 2, 1, 1, HS, None), fails
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "activations_dtype, output_layout",
    [(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)],
)
@pytest.mark.parametrize(
    "fp32_accum",
    [
        False,
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
def test_sd_conv_wh(
    device,
    torch_tensor_map,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    fp32_accum,
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
    shard_layout,
    config_override,
    output_layout,
):
    if device.core_grid.y == 7:
        pytest.skip("This test is not supported for N300")

    # Skip test cases raising OOM, but do not affect the SD e2e test
    if (
        (input_channels == 320 and config_override == None and activations_dtype == ttnn.bfloat16)
        or (input_channels == 960 and config_override == None and fp32_accum == True)
        or (
            output_channels == 1280
            and input_height == 32
            and activations_dtype == ttnn.bfloat16
            and weights_dtype == ttnn.bfloat16
        )
    ):
        pytest.skip("Skip the test cases raising OOM but not affecting e2e test")

    if filter_height > 1 and (input_channels > 1280 or (input_channels > 640 and input_height > 16)):
        run_conv_with_split(
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
            shard_layout=shard_layout,
            split_factor=3 if input_channels == 1920 else 2,
            fp32_accum=fp32_accum,
            packer_l1_acc=True,
        )
    else:
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
            filter_height,
            filter_width,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            config_override,
            shard_layout=shard_layout,
            use_shallow_conv_variant=(input_channels == 16),
            fp32_accum=fp32_accum,
            packer_l1_acc=True,
            output_layout=output_layout,
        )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override, use_shallow_conv_variant",
    (
        # unet convs with batch size 2
        # unique convs in unet (complete list)
        (2, 16, 4, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        (2, 16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        (2, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        (2, 32, 16, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 64, 32, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 64, 64, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 32, 96, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 32, 64, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 16, 48, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        (2, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        (2, 16, 32, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        (2, 16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        # (2, 1, 16, 1056, 160, 1, 1, 1, 1, 0, 0, HS, {"act_block_h": 5 * 32}, False) # Enable when issue #11490 resolved
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype, output_layout",
    [(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_unet_conv_wh(
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
    shard_layout,
    config_override,
    use_shallow_conv_variant,
    output_layout,
    auto_shard,
):
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        pytest.skip("Test is not supported on n300 (8,7) grid")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and input_height >= 1056:
        pytest.skip("OOM")
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
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        shard_layout=shard_layout,
        use_shallow_conv_variant=use_shallow_conv_variant,
        transpose_shards=True,  ## use RM (transpose_mcast=False) with 2D on WH
        output_layout=output_layout,
        auto_shard=auto_shard,
    )


@pytest.mark.parametrize(
    "batch_size",
    [1],
)
@pytest.mark.parametrize(
    "groups",
    [2],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override, use_shallow_conv_variant",
    (
        (16, 4, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        (16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        (16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 8 * 32}, True),
        (32, 16, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (64, 32, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (64, 64, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 96, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 64, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (16, 48, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 8 * 32}, True),
        (16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 8 * 32}, True),
        (16, 32, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        (1, 16, 1056, 160, 1, 1, 1, 1, 0, 0, HS, {"act_block_h": 5 * 32}, False),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_unet_conv_groups_2_wh(
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
    shard_layout,
    config_override,
    use_shallow_conv_variant,
    output_layout,
    auto_shard,
    groups,
):
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        pytest.skip("Test is not supported on n300 (8,7) grid")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and input_height >= 1056:
        pytest.skip("OOM")
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        groups * output_channels,
        groups * input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        shard_layout=shard_layout,
        use_shallow_conv_variant=use_shallow_conv_variant,
        transpose_shards=True,  ## use RM (transpose_mcast=False) with 2D on WH
        output_layout=output_layout,
        auto_shard=auto_shard,
        groups=groups,
    )


@pytest.mark.parametrize(
    "batch_size",
    [1],
)
@pytest.mark.parametrize(
    "groups",
    [4, 6],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override, use_shallow_conv_variant",
    (
        (16, 4, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True),
        (16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True),
        (16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True),
        (32, 16, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (64, 32, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (64, 64, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 96, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 64, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        # (16, 48, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True), # OOM - need inplace convolution
        (16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True),
        # (16, 32, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True), # OOM - need inplace convolution
        (1, 16, 1056, 160, 1, 1, 1, 1, 0, 0, HS, {"act_block_h": 2 * 32}, False),
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
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_unet_conv_groups_4_6_wh(
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
    shard_layout,
    config_override,
    use_shallow_conv_variant,
    output_layout,
    groups,
):
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        pytest.skip("Test is not supported on n300 (8,7) grid")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and input_height >= 1056:
        pytest.skip("OOM")
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        groups * output_channels,
        groups * input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        shard_layout=shard_layout,
        use_shallow_conv_variant=use_shallow_conv_variant,
        transpose_shards=True,  ## use RM (transpose_mcast=False) with 2D on WH
        output_layout=output_layout,
        groups=groups,
    )


@pytest.mark.parametrize(
    "batch_size",
    [1],
)
@pytest.mark.parametrize(
    "groups",
    [8],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override, use_shallow_conv_variant",
    (
        (16, 4, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True),
        # (16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True), # OOM - need inplace convolution
        (16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True),
        (32, 16, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (64, 32, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (64, 64, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 96, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        # (32, 64, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False), # OOM - need inplace convolution
        (32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        # (16, 48, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True), # OOM - need inplace convolution
        (16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True),
        # (16, 32, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True), # OOM - need inplace convolution
        # (1, 16, 1056, 160, 1, 1, 1, 1, 0, 0, True, {"act_block_h": 2 * 32}, True), # OOM - need inplace convolution
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
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("auto_shard", [False], ids=["no_auto_shard"])
def test_unet_conv_groups_8_wh(
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
    shard_layout,
    config_override,
    use_shallow_conv_variant,
    output_layout,
    auto_shard,
    groups,
):
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        pytest.skip("Test is not supported on n300 (8,7) grid")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and input_height >= 1056:
        pytest.skip("OOM")
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        groups * output_channels,
        groups * input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        shard_layout=shard_layout,
        use_shallow_conv_variant=use_shallow_conv_variant,
        transpose_shards=True,  ## use RM (transpose_mcast=False) with 2D on WH
        output_layout=output_layout,
        auto_shard=auto_shard,
        groups=groups,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, config_override",
    (
        (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, {"act_reshard_num_cores_nhw": 1}),
        (1, 128, 128, 32, 32, 3, 3, 2, 2, 1, 1, {"act_reshard_num_cores_nhw": 1}),
        (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, {"act_reshard_num_cores_nhw": 4}),
        (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, {"act_reshard_num_cores_nhw": 8}),
        (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, {"act_reshard_num_cores_nhw": 8, "num_cores_nhw": 4}),
        (2, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, {"act_reshard_num_cores_nhw": 8, "num_cores_nhw": 4}),
        (2, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, {"act_reshard_num_cores_nhw": 4, "num_cores_nhw": 8}),
    ),
)
@pytest.mark.parametrize("shard_layout", [BS, HS])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_halo_reshard_conv(
    device,
    torch_tensor_map,
    use_program_cache,
    shard_layout,
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
    auto_shard,
):
    math_fidelity = ttnn.MathFidelity.HiFi4
    activations_dtype = ttnn.bfloat16
    weights_dtype = ttnn.bfloat8_b

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
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        shard_layout=shard_layout,
        auto_shard=auto_shard,
    )


@pytest.mark.skip("New API needs to be tested")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, config_override, xfail",
    (
        (1, 128, 128, 17, 17, 3, 3, 1, 1, 1, 1, {"num_cores_nhw": 4}, False),
        (1, 128, 128, 17, 17, 3, 3, 2, 2, 1, 1, {"num_cores_nhw": 2}, False),
        (2, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, {"num_cores_nhw": 3}, False),
        (2, 64, 64, 23, 23, 3, 3, 2, 2, 1, 1, {"num_cores_nhw": 3}, False),
        (1, 64, 64, 23, 23, 3, 3, 1, 1, 1, 1, {"num_cores_nhw": 10}, True),
    ),
)
@pytest.mark.parametrize("shard_layout", [BS, HS])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_conv_core_nondivis(
    device,
    torch_tensor_map,
    use_program_cache,
    shard_layout,
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
    xfail,
    auto_shard,
):
    if xfail:
        pytest.xfail()

    math_fidelity = ttnn.MathFidelity.HiFi4
    activations_dtype = ttnn.bfloat16
    weights_dtype = ttnn.bfloat8_b

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
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        shard_layout=shard_layout,
        auto_shard=auto_shard,
    )


# The following test takes various shape sizes from resnet50, unet and stable diffusion and tests for different number of groups - all the way to num_groups = num_in_channels (depthwise conv)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width,  act_block_w_div, shard_layout",
    (
        (768, 768, 16, 16, 1, WS),
        (1280, 1280, 16, 16, 1, WS),
        (1280, 1280, 8, 8, 1, WS),
        (1280, 2560, 8, 8, 1, WS),
        (128, 128, 8, 8, 1, BS),
        (128, 128, 16, 16, 1, BS),
        (128, 128, 32, 32, 1, BS),
        (32, 32, 64, 64, 1, HS),
        (32, 32, 128, 64, 1, HS),
        (16, 16, 528, 80, 1, HS),
        (32, 16, 264, 40, 1, HS),
    ),
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
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "filter_hw, dilation_hw, pad_hw",
    [
        [(3, 3), (2, 2), (2, 2)],
        [(3, 3), (3, 3), (3, 3)],
        # [(3,3), (1,4), (3,3)],
        # [(3,3), (4,1), (3,3)],
        # [(3,3), (4,4), (3,3)],
    ],
)
def test_conv_dilation(
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
    act_block_w_div,
    shard_layout,
    filter_hw,
    stride,
    pad_hw,
    output_layout,
    dilation_hw,
):
    config_override = {"act_block_w_div": act_block_w_div}
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
        filter_hw[0],
        filter_hw[1],
        stride,
        stride,
        pad_hw[0],
        pad_hw[1],
        config_override,
        shard_layout=shard_layout,
        output_layout=output_layout,
        dilation_hw=dilation_hw,
        has_bias=False,
    )


# The following test takes various shape sizes from resnet50, unet and stable diffusion and tests for different number of groups - all the way to num_groups = num_in_channels (depthwise conv)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, groups, shard_layout, config_override, use_shallow_conv_variant",
    (
        (1, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, 2, HS, None, False),
        (1, 64, 64, 32, 32, 3, 3, 1, 1, 1, 1, 64, HS, None, False),
        (2, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, 1, HS, None, False),
        (2, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, 2, HS, None, False),
        (2, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, 8, HS, None, False),
        (1, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, 1, HS, None, False),
        (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, 64, HS, None, False),
        (4, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, 128, HS, None, False),
        (8, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, 128, HS, None, False),
        # (8, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, 256, BS, None, False), circular buffer error
        # (16, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, 256, BS, None, False), # doesn't fit with bfloat16 weights
        # (32, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, 512, BS, None, False), # doesn't fit with bfloat16 weights
        (32, 160, 160, 7, 7, 3, 3, 1, 1, 1, 1, 40, BS, None, False),
        (32, 160, 160, 7, 7, 3, 3, 1, 1, 1, 1, 10, BS, None, False),
        (1, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, 8, HS, None, False),
        (1, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, 16, HS, None, False),
        (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 32, HS, None, False),
        (8, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, 2, BS, None, False),
        (8, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, 4, BS, None, False),
        (1, 320, 320, 32, 32, 3, 3, 1, 1, 1, 1, 2, BS, None, False),
        (1, 640, 640, 16, 16, 3, 3, 1, 1, 1, 1, 320, BS, None, False),
        # (1, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, 1, BS, None, False), # doesn't fit with bfloat16 weights
        (2, 64, 32, 66, 10, 3, 3, 1, 1, 1, 1, 32, HS, None, False),
        (2, 32, 96, 132, 20, 3, 3, 1, 1, 1, 1, 2, HS, None, False),
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
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
# ToDo: Renable this when auto shard heuristic is imporved, currently we run out of L1 in for some test cases
# @pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_conv_groups(
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
    shard_layout,
    config_override,
    use_shallow_conv_variant,
    groups,
    output_layout,
):
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
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        shard_layout=shard_layout,
        use_shallow_conv_variant=use_shallow_conv_variant,
        groups=groups,
        output_layout=output_layout,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override, use_shallow_conv_variant, groups",
    (
        # yolov4 convs with batch size 1
        # unique convs in yolov4 (complete list) # groups: number
        # (1, 32, 32, 480, 640, 3, 3, 1, 1, 1, 1, HS, None, False, 32),  # groups: 32
        # (1, 32, 32, 480, 640, 3, 3, 1, 1, 1, 1, HS, None, False, 32),  # groups: 32
        # (1, 64, 64, 480, 640, 3, 3, 1, 1, 1, 1, HS, None, False, 64),  # groups: 64
        # (1, 64, 64, 480, 640, 3, 3, 1, 1, 1, 1, HS, None, False, 64),  # groups: 64
        # (1, 64, 64, 480, 640, 3, 3, 1, 1, 1, 1, HS, None, False, 64),  # groups: 64
        # (1, 64, 64, 480, 640, 3, 3, 1, 1, 1, 1, HS, None, False, 64),  # groups: 64
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, False, 128),  # groups: 128
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False, 256),  # groups: 256
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, False, 512),  # groups: 512
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, False, 512),  # groups: 512
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, False, 512),  # groups: 512
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, False, 512),  # groups: 512
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, False, 512),  # groups: 512
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, False, 512),  # groups: 512
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, False, 512),  # groups: 512
        (1, 128, 128, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, False, 2),  # groups: 512
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_yolov4_conv_groups_larger_than_one(
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
    shard_layout,
    config_override,
    use_shallow_conv_variant,
    groups,
    output_layout,
    auto_shard,
):
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and input_height >= 1056:
        pytest.skip("OOM")
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
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        shard_layout=shard_layout,
        use_shallow_conv_variant=use_shallow_conv_variant,
        groups=groups,
        output_layout=output_layout,
        auto_shard=auto_shard,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    " output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override, use_shallow_conv_variant, groups",
    ((96, 3, 512, 512, 4, 4, 4, 4, 0, 0, HS, None, False, 1),),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "batch_size",
    [1, 8],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_swin_s_conv(
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
    shard_layout,
    config_override,
    use_shallow_conv_variant,
    groups,
    output_layout,
    auto_shard,
):
    if device.core_grid.y == 7:
        pytest.skip("This test is not supported for N300")
    if batch_size == 8:
        pytest.skip("OOM issue for batch_size 8")
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
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        shard_layout=shard_layout,
        use_shallow_conv_variant=use_shallow_conv_variant,
        groups=groups,
        output_layout=output_layout,
        auto_shard=auto_shard,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, dilation, shard_layout",
    (
        (1, 48, 32, 252, 252, 3, 3, 1, 1, 0, 0, 2, HS),
        (1, 56, 48, 248, 248, 3, 3, 1, 1, 0, 0, 4, HS),
        (1, 64, 56, 240, 240, 3, 3, 1, 1, 0, 0, 8, HS),
        (1, 48, 32, 124, 124, 3, 3, 1, 1, 0, 0, 2, HS),
        (1, 56, 48, 120, 120, 3, 3, 1, 1, 0, 0, 4, HS),
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
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_model_k_256x256(
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
    dilation,
    shard_layout,
    auto_shard,
):
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
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        None,
        shard_layout=shard_layout,
        dilation_hw=(dilation, dilation),
        auto_shard=auto_shard,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels,input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override, use_shallow_conv_variant",
    (
        (1, 32, 3, 480, 640, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 64}, True),
        (1, 32, 32, 480, 640, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 32}, False),
        (1, 64, 32, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, False),
        (1, 64, 64, 240, 320, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 64}, False),
        (1, 128, 64, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False),
        (1, 128, 128, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False),
        (1, 256, 128, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, False),
        (1, 256, 256, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, False),
        (1, 512, 256, 30, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (1, 512, 512, 30, 40, 3, 3, 1, 1, 1, 1, BS, None, False),
        (1, 256, 512, 60, 80, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}, False),
        (1, 128, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 32}, False),
        (1, 64, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 32}, False),
        (1, 32, 64, 256, 256, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 32}, False),
        (1, 1, 32, 480, 640, 1, 1, 1, 1, 0, 0, HS, None, False),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_conv_for_vanilla_unet(
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
    shard_layout,
    config_override,
    use_shallow_conv_variant,
    output_layout,
):
    if device.core_grid.y == 7:
        pytest.skip("This test is not supported for N300")
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
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        shard_layout=shard_layout,
        use_shallow_conv_variant=use_shallow_conv_variant,
        groups=1,
        output_layout=output_layout,
        has_bias=False,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_shallow_conv_with_tiled_input(device):
    out_channels, in_channels, kernel_h, kernel_w = 7, 3, 3, 3
    kernel_shape = (out_channels, in_channels, kernel_h, kernel_w)
    batch_size = 1
    img_h, img_w = 100, 100
    input_shape = (batch_size, in_channels, img_h, img_w)

    stride = (1, 1)
    dilation = (1, 1)
    pad = (1, 1)

    torch_kernel = torch.randn(kernel_shape, dtype=torch.bfloat16)
    tt_kernel = ttnn.from_torch(torch_kernel)

    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device)
    tt_input = ttnn.permute(tt_input, (0, 2, 3, 1))
    tt_input = ttnn.reshape(tt_input, (1, 1, batch_size * img_h * img_w, in_channels))
    tt_input = ttnn.to_layout(tt_input, ttnn.TILE_LAYOUT)

    [tt_out, [out_height, out_width]] = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_kernel,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        bias_tensor=None,
        kernel_size=(kernel_h, kernel_w),
        stride=stride,
        padding=pad,
        dilation_hw=(dilation, dilation),
        batch_size=batch_size,
        input_height=img_h,
        input_width=img_w,
        groups=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        return_output_dim=True,
    )

    tt_output_tensor = ttnn.from_device(tt_out)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_height, out_width, torch_output_tensor.shape[-1])
    torch_output_tensor = torch_output_tensor[:, :, :, :out_channels]

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))

    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input, torch_kernel, bias=None, stride=stride, padding=pad, dilation=dilation, groups=1
    )

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=0.99)
    logger.info(f"PCC = {pcc_msg}. Threshold = 0.99")
    assert passing


# Tests running conv2d which maps to matmul w/o sharding the input tensor.
# Output tensor is in DRAM.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("tiled_input", [True, False])
@pytest.mark.parametrize("input_on_device", [True, False])
def test_dram_input_mm_conv(device, torch_tensor_map, tiled_input, input_on_device):
    batch_size = 1
    out_channels, in_channels = 256, 1024
    img_h, img_w = 128, 128
    input_shape = (batch_size, in_channels, img_h, img_w)

    # Params which map conv2d to matmul op.
    kernel_h, kernel_w = 1, 1
    stride = (1, 1)
    dilation = (1, 1)
    pad = (0, 0)

    kernel_shape = (out_channels, in_channels, kernel_h, kernel_w)
    torch_kernel = randomize_torch_tensor(torch_tensor_map, kernel_shape)
    tt_kernel = ttnn.from_torch(torch_kernel, dtype=ttnn.bfloat16)

    torch_input = randomize_torch_tensor(torch_tensor_map, input_shape)
    if input_on_device:
        tt_input = ttnn.from_torch(torch_input, device=device)
        tt_input = ttnn.permute(tt_input, (0, 2, 3, 1))
        tt_input = ttnn.reshape(tt_input, (1, 1, batch_size * img_h * img_w, in_channels))
    else:
        torch_input_nhwc = torch.permute(torch_input, (0, 2, 3, 1))
        tt_input = ttnn.from_torch(torch_input_nhwc, dtype=ttnn.bfloat16)

    if tiled_input:
        tt_input = ttnn.to_layout(tt_input, ttnn.TILE_LAYOUT)

    tt_out = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_kernel,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        kernel_size=(kernel_h, kernel_w),
        stride=stride,
        padding=pad,
        dilation=dilation,
        batch_size=batch_size,
        input_height=img_h,
        input_width=img_w,
    )

    assert tt_out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED

    tt_output_tensor = ttnn.from_device(tt_out)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(batch_size, img_h, img_w, torch_output_tensor.shape[-1])
    torch_output_tensor = torch_output_tensor[:, :, :, :out_channels]

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))

    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input, torch_kernel, bias=None, stride=stride, padding=pad, dilation=dilation, groups=1
    )

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=0.99)
    logger.info(f"PCC = {pcc_msg}. Threshold = 0.99")
    assert passing


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override",
    ((16, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, HS, {"act_block_h": 32 * 49}),),
)
def test_split_reader_regression(
    device,
    torch_tensor_map,
    use_program_cache,
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
    shard_layout,
    config_override,
):
    run_conv(
        device,
        torch_tensor_map,
        ttnn.MathFidelity.LoFi,
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
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
        config_override=config_override,
        use_shallow_conv_variant=True,
        has_bias=False,
        shard_layout=shard_layout,
        enable_split_reader=True,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_small_in_large_out_channels_auto_shard(device, torch_tensor_map):
    batch_size = 2
    in_channels = 16
    out_channels = 1536
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = (0, 0)
    height = 128
    width = 128

    run_conv(
        device,
        torch_tensor_map,
        ttnn.MathFidelity.LoFi,
        ttnn.bfloat16,
        ttnn.bfloat16,
        batch_size,
        out_channels,
        in_channels,
        height,
        width,
        kernel_size[0],
        kernel_size[1],
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        None,
        auto_shard=True,
    )


# fmt: off
@pytest.mark.parametrize(
    "batch, input_channels, output_channels, input_height, input_width, kernel, stride, padding",
    (
        (1, 64, 64, 128, 128, (3, 3), (1, 1), (1, 1)),
    ),
)
#fmt: on

@pytest.mark.parametrize("shard_layout", [BS])
@pytest.mark.parametrize("activation", ["relu"])

@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384*2}], indirect=True)
def test_block_sharding_relu_act_block_h(
    device,
    torch_tensor_map,
    batch,
    input_channels,
    output_channels,
    input_height,
    input_width,
    kernel,
    stride,
    padding,
    shard_layout,
    activation,
):
    config_override = {}
    config_override["act_block_h"] = 32
    run_conv(
        device,
        torch_tensor_map,
        ttnn.MathFidelity.LoFi,
        ttnn.bfloat16,
        ttnn.bfloat16,
        batch,
        output_channels,
        input_channels,
        input_height,
        input_width,
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        config_override=config_override,
        shard_layout=shard_layout,
        activation=activation,
    )

# fmt: off
@pytest.mark.parametrize(
    "batch, input_channels, output_channels, input_height, input_width, weights_dtype, activations_dtype, groups, kernel, stride, padding, dilation, auto_shard, use_shallow_conv_variant, act_block_h_override, act_block_w_div, deallocate_activation, math_fidelity, fp32_accum, packer_l1_acc, enable_split_reader, enable_act_double_buffer",
    (
    #     # torch_conv2d:
    #     (1, 32, 48, 320, 320, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (2, 2), (2, 2), True, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    #     # # torch_split_knit_dilation: 4x
    #     (1, 32, 48, 160, 160, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 1), (1, 1), HS, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    #     # # torch_split_knit_grouped_dilation: 1x
    #     (1, 128, 192, 160, 160, ttnn.bfloat8_b, ttnn.bfloat8_b, 4, (3, 3), (1, 1), (1, 1), (1, 1), HS, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    #     # # torch_split_knit_dilation_h: 2x
    #     (1, 32, 48, 160, 320, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 2), (1, 2), HS, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    #     # # torch_split_knit_grouped_dilation: 1x
    #     (1, 64, 96, 160, 320, ttnn.bfloat8_b, ttnn.bfloat8_b, 2, (3, 3), (1, 1), (1, 2), (1, 2), HS, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),

    #     # # # torch_conv2d:
    #     (1, 48, 56, 320, 320, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (4, 4), (4, 4), True, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    #     # # torch_split_knit_dilation: 16x
    #     (1, 48, 56, 80, 80, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 1), (1, 1), HS, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    #     # # torch_split_knit_grouped_dilation: 1x
    #     (1, 768, 896, 80, 80, ttnn.bfloat8_b, ttnn.bfloat8_b, 16, (3, 3), (1, 1), (1, 1), (1, 1), True, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    #     # # torch_split_knit_dilation_h: 4x
    #     (1, 48, 56, 80, 320, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 4), (1, 4), HS, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    #     # # torch_split_knit_grouped_dilation: 1x
    #     (1, 192, 224, 80, 320, ttnn.bfloat8_b, ttnn.bfloat8_b, 4, (3, 3), (1, 1), (1, 4), (1, 4), True, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),

    #     # # torch_conv2d:
    #     (1, 56, 64, 320, 320, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (8, 8), (8, 8), True, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    #     # # torch_split_knit_dilation: 64x
    #     (1, 56, 64, 40, 40, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 1), (1, 1), HS, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    #     # # torch_split_knit_grouped_dilation: 1x
    #     (1, 3584, 4096, 40, 40, ttnn.bfloat8_b, ttnn.bfloat8_b, 64, (3, 3), (1, 1), (1, 1), (1, 1), True, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    #     # # torch_split_knit_dilation_h: 8x
    #     (1, 56, 64, 40, 320, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 8), (1, 8), HS, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    #     # # torch_split_knit_grouped_dilation: 1x
    #     (1, 448, 512, 40, 320, ttnn.bfloat8_b, ttnn.bfloat8_b, 8, (3, 3), (1, 1), (1, 8), (1, 8), True, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),

    # 320x320
    # # torch_conv2d:
    # (1, 32, 48, 320, 320, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (2, 2), (2, 2), HS, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
    # # torch_split_knit_dilation: 4x
    # (1, 32, 48, 160, 160, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 1), (1, 1), HS, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
    # # torch_split_knit_batched_dilation: 1x
    # (4, 32, 48, 160, 160, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 1), (1, 1), HS, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
    # # # torch_split_knit_grouped_dilation: 1x
    # # (1, 128, 192, 160, 160, ttnn.bfloat8_b, ttnn.bfloat8_b, 4, (3, 3), (1, 1), (1, 1), (1, 1), True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    # # # torch_split_knit_dilation_h: 2x
    # # (1, 32, 48, 160, 320, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 2), (1, 2), True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    # # # torch_split_knit_grouped_dilation: 1x
    # # (1, 64, 96, 160, 320, ttnn.bfloat8_b, ttnn.bfloat8_b, 2, (3, 3), (1, 1), (1, 2), (1, 2), True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),

    # # torch_conv2d:
    # (1, 48, 56, 320, 320, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (4, 4), (4, 4), HS, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
    # # torch_split_knit_dilation: 16x
    # (1, 48, 56, 80, 80, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 1), (1, 1), HS, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
    # # torch_split_knit_batched_dilation: 1x
    # (16, 48, 56, 80, 80, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 1), (1, 1), HS, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
    # # # torch_split_knit_grouped_dilation: 1x
    # # (1, 768, 896, 80, 80, ttnn.bfloat8_b, ttnn.bfloat8_b, 16, (3, 3), (1, 1), (1, 1), (1, 1), True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    # # # torch_split_knit_dilation_h: 4x
    # # (1, 48, 56, 80, 320, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 4), (1, 4), True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    # # # torch_split_knit_grouped_dilation: 1x
    # # (1, 192, 224, 80, 320, ttnn.bfloat8_b, ttnn.bfloat8_b, 4, (3, 3), (1, 1), (1, 4), (1, 4), True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),

    # # torch_conv2d:
    # (1, 56, 64, 320, 320, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (8, 8), (8, 8), HS, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
    # # torch_split_knit_dilation: 64x
    # (1, 56, 64, 40, 40, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 1), (1, 1), HS, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
    # # torch_split_knit_batched_dilation: 1x
    # (64, 56, 64, 40, 40, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 1), (1, 1), HS, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
    # # # torch_split_knit_grouped_dilation: 1x
    # # (1, 3584, 4096, 40, 40, ttnn.bfloat8_b, ttnn.bfloat8_b, 64, (3, 3), (1, 1), (1, 1), (1, 1), True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    # # # torch_split_knit_dilation_h: 8x
    # # (1, 56, 64, 40, 320, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 8), (1, 8), True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    # # # torch_split_knit_grouped_dilation: 1x
    # # (1, 448, 512, 40, 320, ttnn.bfloat8_b, ttnn.bfloat8_b, 8, (3, 3), (1, 1), (1, 8), (1, 8), True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),

    # 1024x128, 256, 512
    # torch_conv2d:
    (1, 32, 48, 1024, 128*4, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (2, 2), (2, 2), HS, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
    # torch_split_knit_dilation: 4x
    (1, 32, 48, 512, 64*4, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 1), (1, 1), HS, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
    # torch_split_knit_batched_dilation: 1x
    (4, 32, 48, 512, 64*4, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 1), (1, 1), HS, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
    # # torch_split_knit_grouped_dilation: 1x
    # (1, 128, 192, 512, 64, ttnn.bfloat8_b, ttnn.bfloat8_b, 4, (3, 3), (1, 1), (1, 1), (1, 1), True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    # # torch_split_knit_dilation_h: 2x
    # (1, 32, 48, 512, 128, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 2), (1, 2), True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    # # torch_split_knit_grouped_dilation: 1x
    # (1, 64, 96, 512, 128, ttnn.bfloat8_b, ttnn.bfloat8_b, 2, (3, 3), (1, 1), (1, 2), (1, 2), True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),

    # torch_conv2d:
    (1, 48, 56, 1024, 128*4, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (4, 4), (4, 4), HS, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
    # torch_split_knit_dilation: 16x
    (1, 48, 56, 256, 32*4, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 1), (1, 1), HS, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
    # torch_split_knit_batched_dilation: 1x
    (16, 48, 56, 256, 32*4, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 1), (1, 1), HS, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
    # # torch_split_knit_grouped_dilation: 1x
    # (1, 768, 896, 256, 32, ttnn.bfloat8_b, ttnn.bfloat8_b, 16, (3, 3), (1, 1), (1, 1), (1, 1), True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    # # torch_split_knit_dilation_h: 4x
    # (1, 48, 56, 256, 128, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 4), (1, 4), True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    # # torch_split_knit_grouped_dilation: 1x
    # (1, 192, 224, 256, 128, ttnn.bfloat8_b, ttnn.bfloat8_b, 4, (3, 3), (1, 1), (1, 4), (1, 4), True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),

    # torch_conv2d:
    (1, 56, 64, 1024, 128*4, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (8, 8), (8, 8), HS, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
    # torch_split_knit_dilation: 64x
    (1, 56, 64, 128, 16*4, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 1), (1, 1), HS, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
    # torch_split_knit_batched_dilation: 1x
    (64, 56, 64, 128, 16*4, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 1), (1, 1), HS, False, 32*2, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
    # # torch_split_knit_grouped_dilation: 1x
    # (1, 3584, 4096, 128, 16, ttnn.bfloat8_b, ttnn.bfloat8_b, 64, (3, 3), (1, 1), (1, 1), (1, 1), True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    # # torch_split_knit_dilation_h: 8x
    # (1, 56, 64, 128, 128, ttnn.bfloat8_b, ttnn.bfloat8_b, 1, (3, 3), (1, 1), (1, 8), (1, 8), True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),
    # # torch_split_knit_grouped_dilation: 1x
    # (1, 448, 512, 128, 128, ttnn.bfloat8_b, ttnn.bfloat8_b, 8, (3, 3), (1, 1), (1, 8), (1, 8), True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),


    ),
)
 #fmt: on

@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384*2}], indirect=True)
def test_conv2d_model_fruit(
    device,
    torch_tensor_map,
    batch,
    input_channels,
    output_channels,
    input_height,
    input_width,
    weights_dtype,
    activations_dtype,
    groups,
    kernel,
    stride,
    padding,
    dilation,
    auto_shard,
    use_shallow_conv_variant,
    act_block_h_override,
    act_block_w_div,
    deallocate_activation,
    math_fidelity,
    fp32_accum,
    packer_l1_acc,
    enable_split_reader,
    enable_act_double_buffer
):
    config_override = {}
    config_override["act_block_h"] = act_block_h_override
    config_override["act_block_w_div"] = act_block_w_div

    run_conv(
        device=device,
        torch_tensor_map=torch_tensor_map,
        math_fidelity=math_fidelity,
        activations_dtype=activations_dtype,
        weights_dtype=weights_dtype,
        batch_size=batch,
        output_channels=output_channels,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        filter_height=kernel[0],
        filter_width=kernel[1],
        stride_h=stride[0],
        stride_w=stride[1],
        pad_h=padding[0],
        pad_w=padding[1],
        config_override=config_override,
        dilation_hw=dilation,
        use_shallow_conv_variant=use_shallow_conv_variant,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        output_layout=ttnn.TILE_LAYOUT,
        deallocate_activation=deallocate_activation,
        debug=False,
        groups=groups,
        has_bias=True,
        # shard_layout=HS,
        # auto_shard=False,
        shard_layout=None if auto_shard == True else auto_shard,
        auto_shard=True if auto_shard == True else False,
        memory_config=None,
        input_mesh_mapper=None,
        weight_mesh_mapper=None,
        output_mesh_composer=None,
        enable_split_reader=enable_split_reader,
        input_memory_config=None,
        enable_act_double_buffer=enable_act_double_buffer,
    )



# Split-Knit Conv2d with high dilation
def torch_split_knit_dilation(torch_input_tensor_nchw, torch_weight_tensor_oihw, torch_bias_tensor, filter_hw, stride_hw, padding_hw, dilation_hw, groups):
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
                    split_h::dilation_hw[0],
                    split_w::dilation_hw[1],
                ]
            )

    # conv2d
    sk_padding_hw=(1,1) if padding_hw[0] > 0 else (0,0)
    sk_dilation_hw=(1,1)
    sk_groups=groups
    """batch, input_channels, output_channels, input_height, input_width, weights_dtype, activations_dtype, groups, kernel, stride, padding, dilation")"""
    print(f"# torch_split_knit_dilation: {len(inputs_splited)}x \n({inputs_splited[0].shape[0]}, {torch_weight_tensor_oihw.shape[1]}, {torch_weight_tensor_oihw.shape[0]}, {inputs_splited[0].shape[2]}, {inputs_splited[0].shape[3]}, ttnn.bfloat8_b, ttnn.bfloat8_b, {sk_groups}, ({torch_weight_tensor_oihw.shape[2]}, {torch_weight_tensor_oihw.shape[3]}), (1, 1), {sk_padding_hw}, {sk_dilation_hw}, True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),")

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
    out_h = (torch_input_tensor_nchw.shape[2] + 2 * padding_hw[0] - dilation_hw[0] * (filter_hw[0] - 1) - 1) // stride_hw[0] + 1
    out_w = (torch_input_tensor_nchw.shape[3] + 2 * padding_hw[1] - dilation_hw[1] * (filter_hw[1] - 1) - 1) // stride_hw[1] + 1
    out_knitted = torch.zeros((torch_input_tensor_nchw.shape[0], torch_weight_tensor_oihw.shape[0], out_h, out_w))

    for split_h in range(dilation_hw[0]):
        for split_w in range(dilation_hw[1]):
            out_knitted[:,:,split_h::dilation_hw[0],split_w::dilation_hw[1]] = outs_splited[split_h * dilation_hw[1] + split_w]


    return out_knitted


# Split-Knit Conv2d with high dilation (batched)
def torch_split_knit_batched_dilation(torch_input_tensor_nchw, torch_weight_tensor_oihw, torch_bias_tensor, filter_hw, stride_hw, padding_hw, dilation_hw, groups):
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
    sk_batch=torch_input_tensor_nchw.shape[0] * dilation_hw[0] * dilation_hw[1]
    inputs_grouped_splited = torch.zeros((sk_batch, torch_input_tensor_nchw.shape[1], torch_input_tensor_nchw.shape[2] // dilation_hw[0], torch_input_tensor_nchw.shape[3] // dilation_hw[1]))
    for split_h in range(dilation_hw[0]):
        for split_w in range(dilation_hw[1]):
            batch_idx = split_h * dilation_hw[1] + split_w
            inputs_grouped_splited[batch_idx,:,:,:] =  torch_input_tensor_nchw[
                    :,
                    :,
                    split_h::dilation_hw[0],
                    split_w::dilation_hw[1],
                ]

    sk_padding_hw=(1,1) if padding_hw[0] > 0 else (0,0)
    sk_dilation_hw=(1,1)
    sk_groups=1

    # conv2d
    out_splited = torch.nn.functional.conv2d(
        inputs_grouped_splited,
        torch_weight_tensor_oihw,
        # bias=torch_bias_tensor.reshape(-1).repeat(dilation_hw[0]*dilation_hw[1]) if torch_bias_tensor is not None else None, # TBD if this is OK
        bias=torch_bias_tensor.reshape(-1) if torch_bias_tensor is not None else None, # TBD if this is OK
        stride=stride_hw,
        padding=sk_padding_hw,
        dilation=sk_dilation_hw,
        groups=sk_groups
    )

    """batch, input_channels, output_channels, input_height, input_width, weights_dtype, activations_dtype, groups, kernel, stride, padding, dilation")"""
    print(f"# torch_split_knit_batched_dilation: 1x \n({inputs_grouped_splited.shape[0]}, {inputs_grouped_splited.shape[1]}, {out_splited.shape[1]}, {inputs_grouped_splited.shape[2]}, {inputs_grouped_splited.shape[3]}, ttnn.bfloat8_b, ttnn.bfloat8_b, {sk_groups}, ({torch_weight_tensor_oihw.shape[2]}, {torch_weight_tensor_oihw.shape[3]}), (1, 1), {sk_padding_hw}, {sk_dilation_hw}, True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),")

    # knit
    out_h = (torch_input_tensor_nchw.shape[2] + 2 * padding_hw[0] - dilation_hw[0] * (filter_hw[0] - 1) - 1) // stride_hw[0] + 1
    out_w = (torch_input_tensor_nchw.shape[3] + 2 * padding_hw[1] - dilation_hw[1] * (filter_hw[1] - 1) - 1) // stride_hw[1] + 1
    out_knitted = torch.zeros((torch_input_tensor_nchw.shape[0], torch_weight_tensor_oihw.shape[0], out_h, out_w))

    # for out_channel in range(out_channels):
    for split_h in range(dilation_hw[0]):
        for split_w in range(dilation_hw[1]):
            src_batch_idx = split_h * dilation_hw[1] + split_w
            out_knitted[:,:,split_h::dilation_hw[0],split_w::dilation_hw[1]] = out_splited[src_batch_idx,:,:,:]

    return out_knitted

# Split-Knit Conv2d with high dilation (batched)
def ttnn_split_knit_batched_dilation(device, torch_input_tensor_nchw, torch_weight_tensor_oihw, torch_bias_tensor, filter_hw, stride_hw, padding_hw, dilation_hw, groups):
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
    print(f"sk_batch_size {sk_batch_size}, sk_in_channels {sk_in_channels}, sk_out_channels {sk_out_channels}, sk_input_height {sk_input_height}, sk_input_width {sk_input_width}")
    torch_inputs_grouped_splited = torch.zeros((sk_batch_size, torch_input_tensor_nchw.shape[1], torch_input_tensor_nchw.shape[2] // dilation_hw[0], torch_input_tensor_nchw.shape[3] // dilation_hw[1]))
    for split_h in range(dilation_hw[0]):
        for split_w in range(dilation_hw[1]):
            batch_idx = split_h * dilation_hw[1] + split_w
            torch_inputs_grouped_splited[batch_idx,:,:,:] =  torch_input_tensor_nchw[
                    :,
                    :,
                    split_h::dilation_hw[0],
                    split_w::dilation_hw[1],
                ]



    tt_input_tensor_nchw = ttnn.from_torch(
        torch_inputs_grouped_splited,
        activations_dtype if activations_dtype == ttnn.float32 else ttnn.bfloat16,
        mesh_mapper=None,
    ).to(device)

    tt_input_tensor = ttnn.permute(tt_input_tensor_nchw, (0, 2, 3, 1))

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor_oihw,
        weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32,
        mesh_mapper=None
    )

    tt_bias_tensor = ttnn.from_torch(
        torch_bias_tensor,
        weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32,
        mesh_mapper=None,
    )

    sk_padding_hw=(1,1) if padding_hw[0] > 0 else (0,0)
    sk_dilation_hw=(1,1)
    sk_groups=1

    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        shard_layout=HS,
        input_channels_alignment=32,
        deallocate_activation=True,
        enable_act_double_buffer=True,
        enable_split_reader=True,
        enable_subblock_padding=False,
        # output_layout=ttnn.ROW_MAJOR_LAYOUT,
        output_layout=ttnn.TILE_LAYOUT,
        activation="",
        act_block_h_override = 32*2,
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
    tt_output_splited_tensor_on_device = tt_output_splited_tensor_on_device.reshape(sk_batch_size, out_h, out_w, tt_output_splited_tensor_on_device.shape[-1])
    ttnn.synchronize_device(device)

    # host fallback
    out_splited = ttnn.to_torch(tt_output_splited_tensor_on_device, mesh_composer=None)
    out_splited = torch.permute(out_splited, (0, 3, 1, 2))


    # knit (HOST)
    full_out_h = (torch_input_tensor_nchw.shape[2] + 2 * padding_hw[0] - dilation_hw[0] * (filter_hw[0] - 1) - 1) // stride_hw[0] + 1
    full_out_w = (torch_input_tensor_nchw.shape[3] + 2 * padding_hw[1] - dilation_hw[1] * (filter_hw[1] - 1) - 1) // stride_hw[1] + 1
    out_knitted = torch.zeros((torch_input_tensor_nchw.shape[0], torch_weight_tensor_oihw.shape[0], full_out_h, full_out_w))

    # for out_channel in range(out_channels):
    for split_h in range(dilation_hw[0]):
        for split_w in range(dilation_hw[1]):
            src_batch_idx = split_h * dilation_hw[1] + split_w
            out_knitted[:,:,split_h::dilation_hw[0],split_w::dilation_hw[1]] = out_splited[src_batch_idx,:,:,:]

    return out_knitted

# Split-Knit Conv2d with high dilation, grouped conv2d approach
def torch_split_knit_grouped_dilation(torch_input_tensor_nchw, torch_weight_tensor_oihw, torch_bias_tensor, filter_hw, stride_hw, padding_hw, dilation_hw, groups):
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
    inputs_grouped_splited = torch.zeros((torch_input_tensor_nchw.shape[0], dilation_hw[0] * dilation_hw[1] * torch_input_tensor_nchw.shape[1], torch_input_tensor_nchw.shape[2] // dilation_hw[0], torch_input_tensor_nchw.shape[3] // dilation_hw[1]))
    torch_weight_tensor_oihw_grouped = torch.zeros((dilation_hw[0] * dilation_hw[1] * torch_weight_tensor_oihw.shape[0], torch_weight_tensor_oihw.shape[1], torch_weight_tensor_oihw.shape[2], torch_weight_tensor_oihw.shape[3]))
    for split_h in range(dilation_hw[0]):
        for split_w in range(dilation_hw[1]):
            group_idx = split_h * dilation_hw[1] + split_w
            inputs_grouped_splited[:, group_idx*in_channels:(group_idx+1)*in_channels ,:,:] =  torch_input_tensor_nchw[
                    :,
                    :,
                    split_h::dilation_hw[0],
                    split_w::dilation_hw[1],
                ]
            torch_weight_tensor_oihw_grouped[group_idx*out_channels:(group_idx+1)*out_channels,:,:,:] = torch_weight_tensor_oihw

    sk_padding_hw=(1,1) if padding_hw[0] > 0 else (0,0)
    sk_dilation_hw=(1,1)
    sk_groups=dilation_hw[0] * dilation_hw[1]

    # conv2d
    out_splited = torch.nn.functional.conv2d(
        inputs_grouped_splited,
        torch_weight_tensor_oihw_grouped,
        bias=torch_bias_tensor.reshape(-1).repeat(dilation_hw[0]*dilation_hw[1]) if torch_bias_tensor is not None else None,
        stride=stride_hw,
        padding=sk_padding_hw,
        dilation=sk_dilation_hw,
        groups=sk_groups
    )

    """batch, input_channels, output_channels, input_height, input_width, weights_dtype, activations_dtype, groups, kernel, stride, padding, dilation")"""
    print(f"# torch_split_knit_grouped_dilation: 1x \n({inputs_grouped_splited.shape[0]}, {inputs_grouped_splited.shape[1]}, {out_splited.shape[1]}, {inputs_grouped_splited.shape[2]}, {inputs_grouped_splited.shape[3]}, ttnn.bfloat8_b, ttnn.bfloat8_b, {sk_groups}, ({torch_weight_tensor_oihw_grouped.shape[2]}, {torch_weight_tensor_oihw_grouped.shape[3]}), (1, 1), {sk_padding_hw}, {sk_dilation_hw}, True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),")

    # knit
    out_h = (torch_input_tensor_nchw.shape[2] + 2 * padding_hw[0] - dilation_hw[0] * (filter_hw[0] - 1) - 1) // stride_hw[0] + 1
    out_w = (torch_input_tensor_nchw.shape[3] + 2 * padding_hw[1] - dilation_hw[1] * (filter_hw[1] - 1) - 1) // stride_hw[1] + 1
    out_knitted = torch.zeros((torch_input_tensor_nchw.shape[0], torch_weight_tensor_oihw.shape[0], out_h, out_w))

    # for out_channel in range(out_channels):
    for split_h in range(dilation_hw[0]):
        for split_w in range(dilation_hw[1]):
            src_group_idx = split_h * dilation_hw[1] + split_w
            out_knitted[:,:,split_h::dilation_hw[0],split_w::dilation_hw[1]] = out_splited[:,src_group_idx*out_channels:(src_group_idx+1)*out_channels,:,:]


    return out_knitted

# Split-Knit Conv2d with high dilation, height only
def torch_split_knit_dilation_h(torch_input_tensor_nchw, torch_weight_tensor_oihw, torch_bias_tensor, filter_hw, stride_hw, padding_hw, dilation_hw, groups):
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
                split_h::dilation_hw[0],
                :,
            ]
        )

    # conv2d
    sk_padding_hw=(1,padding_hw[1]) if padding_hw[0] > 0 else (0,padding_hw[1])
    sk_dilation_hw=(1,dilation_hw[1])
    sk_groups=groups
    """batch, input_channels, output_channels, input_height, input_width, weights_dtype, activations_dtype, groups, kernel, stride, padding, dilation")"""
    print(f"# torch_split_knit_dilation_h: {len(inputs_splited)}x \n({inputs_splited[0].shape[0]}, {torch_weight_tensor_oihw.shape[1]}, {torch_weight_tensor_oihw.shape[0]}, {inputs_splited[0].shape[2]}, {inputs_splited[0].shape[3]}, ttnn.bfloat8_b, ttnn.bfloat8_b, {sk_groups}, ({torch_weight_tensor_oihw.shape[2]}, {torch_weight_tensor_oihw.shape[3]}), (1, 1), {sk_padding_hw}, {sk_dilation_hw}, True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),")

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
    out_h = (torch_input_tensor_nchw.shape[2] + 2 * padding_hw[0] - dilation_hw[0] * (filter_hw[0] - 1) - 1) // stride_hw[0] + 1
    out_w = (torch_input_tensor_nchw.shape[3] + 2 * padding_hw[1] - dilation_hw[1] * (filter_hw[1] - 1) - 1) // stride_hw[1] + 1
    out_knitted = torch.zeros((torch_input_tensor_nchw.shape[0], torch_weight_tensor_oihw.shape[0], out_h, out_w))

    for split_h in range(dilation_hw[0]):
        out_knitted[:,:,split_h::dilation_hw[0],:] = outs_splited[split_h]

    return out_knitted

# Split-Knit Conv2d with high dilation, grouped conv2d approach, height-only
def torch_split_knit_grouped_dilation_h(torch_input_tensor_nchw, torch_weight_tensor_oihw, torch_bias_tensor, filter_hw, stride_hw, padding_hw, dilation_hw, groups):
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
    inputs_grouped_splited = torch.zeros((torch_input_tensor_nchw.shape[0], dilation_hw[0] * torch_input_tensor_nchw.shape[1], torch_input_tensor_nchw.shape[2] // dilation_hw[0], torch_input_tensor_nchw.shape[3]))
    torch_weight_tensor_oihw_grouped = torch.zeros((dilation_hw[0] * torch_weight_tensor_oihw.shape[0], torch_weight_tensor_oihw.shape[1], torch_weight_tensor_oihw.shape[2], torch_weight_tensor_oihw.shape[3]))
    for split_h in range(dilation_hw[0]):
        inputs_grouped_splited[:, split_h*in_channels:(split_h+1)*in_channels ,:,:] =  torch_input_tensor_nchw[
                :,
                :,
                split_h::dilation_hw[0],
                :,
            ]
        torch_weight_tensor_oihw_grouped[split_h*out_channels:(split_h+1)*out_channels,:,:,:] = torch_weight_tensor_oihw

    # conv2d
    sk_padding_hw=(1,padding_hw[1]) if padding_hw[0] > 0 else (0,padding_hw[1])
    sk_dilation_hw=(1,dilation_hw[1])
    sk_groups=dilation_hw[0]
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
    print(f"# torch_split_knit_grouped_dilation: 1x \n({inputs_grouped_splited.shape[0]}, {inputs_grouped_splited.shape[1]}, {out_splited.shape[1]}, {inputs_grouped_splited.shape[2]}, {inputs_grouped_splited.shape[3]}, ttnn.bfloat8_b, ttnn.bfloat8_b, {sk_groups}, ({torch_weight_tensor_oihw_grouped.shape[2]}, {torch_weight_tensor_oihw_grouped.shape[3]}), (1, 1), {sk_padding_hw}, {sk_dilation_hw}, True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),")

    # knit
    out_h = (torch_input_tensor_nchw.shape[2] + 2 * padding_hw[0] - dilation_hw[0] * (filter_hw[0] - 1) - 1) // stride_hw[0] + 1
    out_w = (torch_input_tensor_nchw.shape[3] + 2 * padding_hw[1] - dilation_hw[1] * (filter_hw[1] - 1) - 1) // stride_hw[1] + 1
    out_knitted = torch.zeros((torch_input_tensor_nchw.shape[0], torch_weight_tensor_oihw.shape[0], out_h, out_w))

    for split_h in range(dilation_hw[0]):
        out_knitted[:,:,split_h::dilation_hw[0],:] = out_splited[:,split_h*out_channels:(split_h+1)*out_channels,:,:]

    return out_knitted

# fmt: off
@pytest.mark.parametrize(
    "input_shape_nchw, output_channels, filter_hw, stride_hw, padding_hw, dilation_hw",
    (
        # ((1, 32, 320, 320), 48, (3, 3), (1, 1), (2, 2), (2, 2)),
        # ((1, 48, 320, 320), 56, (3, 3), (1, 1), (4, 4), (4, 4)),
        # ((1, 56, 320, 320), 64, (3, 3), (1, 1), (8, 8), (8, 8)),
        # ((1, 32, 320, 320), 48, (3, 3), (1, 1), (0, 0), (2, 2)),
        # ((1, 48, 320, 320), 56, (3, 3), (1, 1), (0, 0), (4, 4)),
        # ((1, 56, 320, 320), 64, (3, 3), (1, 1), (0, 0), (8, 8)),

        ((1, 32, 1024, 128), 48, (3, 3), (1, 1), (2, 2), (2, 2)),
        ((1, 48, 1024, 128), 56, (3, 3), (1, 1), (4, 4), (4, 4)),
        ((1, 56, 1024, 128), 64, (3, 3), (1, 1), (8, 8), (8, 8)),
    ),
)
# fmt: on
def test_conv2d_split_knit_dilation(
    input_shape_nchw, torch_tensor_map, output_channels, filter_hw, stride_hw, padding_hw, dilation_hw
):
    has_bias = True
    groups = 1
    assert all(d % 2 == 0 for d in dilation_hw), "dilation must be even"
    assert all(s == 1 for s in stride_hw), "stride must be 1"
    assert groups == 1, "groups must be 1"

    torch.manual_seed(0)
    batch_size, input_channels, input_height, input_width = input_shape_nchw

    conv_input_shape_nchw = (batch_size, input_channels, input_height, input_width)
    conv_weight_shape_oihw = (output_channels, input_channels // groups, filter_hw[0], filter_hw[1])
    conv_bias_shape = (1, 1, 1, output_channels)
    torch_input_tensor_nchw = randomize_torch_tensor(torch_tensor_map, conv_input_shape_nchw)
    # torch_input_tensor_nhwc = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))

    torch_weight_tensor_oihw = randomize_torch_tensor(torch_tensor_map, conv_weight_shape_oihw)
    torch_bias_tensor = randomize_torch_tensor(torch_tensor_map, conv_bias_shape) if has_bias else None

    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor_nchw,
        torch_weight_tensor_oihw,
        bias=torch_bias_tensor.reshape(-1) if has_bias else None,
        stride=stride_hw,
        padding=padding_hw,
        dilation=dilation_hw,
        groups=groups,
    )
    """batch, input_channels, output_channels, input_height, input_width, weights_dtype, activations_dtype, groups, kernel, stride, padding, dilation")"""
    print(
        f"# torch_conv2d:\n({torch_input_tensor_nchw.shape[0]}, {torch_weight_tensor_oihw.shape[1]}, {torch_weight_tensor_oihw.shape[0]}, {torch_input_tensor_nchw.shape[2]}, {torch_input_tensor_nchw.shape[3]}, ttnn.bfloat8_b, ttnn.bfloat8_b, {1}, ({torch_weight_tensor_oihw.shape[2]}, {torch_weight_tensor_oihw.shape[3]}), (1, 1), {padding_hw}, {dilation_hw}, True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),"
    )

    # =================== Split-Knit Conv2d ===================
    torch_split_knit_out = torch_split_knit_dilation(
        torch_input_tensor_nchw,
        torch_weight_tensor_oihw,
        torch_bias_tensor,
        filter_hw,
        stride_hw,
        padding_hw,
        dilation_hw,
        groups,
    )
    pcc = 0.999
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_out_golden_tensor, torch_split_knit_out, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing, pcc_msg

    # =================== Split-Knit Conv2d, batched conv2d approach ===================
    torch_split_knit_batched_out = torch_split_knit_batched_dilation(
        torch_input_tensor_nchw,
        torch_weight_tensor_oihw,
        torch_bias_tensor,
        filter_hw,
        stride_hw,
        padding_hw,
        dilation_hw,
        groups,
    )
    pcc = 0.999
    passing, pcc_msg = check_with_pcc_without_tensor_printout(
        torch_out_golden_tensor, torch_split_knit_batched_out, pcc=pcc
    )
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing, pcc_msg

    # =================== Split-Knit Conv2d, grouped conv2d approach ===================
    torch_split_knit_grouped_out = torch_split_knit_grouped_dilation(
        torch_input_tensor_nchw,
        torch_weight_tensor_oihw,
        torch_bias_tensor,
        filter_hw,
        stride_hw,
        padding_hw,
        dilation_hw,
        groups,
    )
    pcc = 0.999
    grouped_passing, pcc_msg = check_with_pcc_without_tensor_printout(
        torch_out_golden_tensor, torch_split_knit_grouped_out, pcc=pcc
    )
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert grouped_passing, pcc_msg

    # =================== Split-Knit Conv2d with height only ===================
    torch_split_knit_h_out = torch_split_knit_dilation_h(
        torch_input_tensor_nchw,
        torch_weight_tensor_oihw,
        torch_bias_tensor,
        filter_hw,
        stride_hw,
        padding_hw,
        dilation_hw,
        groups,
    )
    height_passing, pcc_msg = check_with_pcc_without_tensor_printout(
        torch_out_golden_tensor, torch_split_knit_h_out, pcc=pcc
    )
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert height_passing, pcc_msg

    # =================== Split-Knit Conv2d with height only, grouped conv2d approach =================
    torch_split_knit_grouped_h_out = torch_split_knit_grouped_dilation_h(
        torch_input_tensor_nchw,
        torch_weight_tensor_oihw,
        torch_bias_tensor,
        filter_hw,
        stride_hw,
        padding_hw,
        dilation_hw,
        groups,
    )
    grouped_height_passing, pcc_msg = check_with_pcc_without_tensor_printout(
        torch_out_golden_tensor, torch_split_knit_grouped_h_out, pcc=pcc
    )
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert grouped_height_passing, pcc_msg


# fmt: off
@pytest.mark.parametrize(
    "input_shape_nchw, output_channels, filter_hw, stride_hw, padding_hw, dilation_hw",
    (
        # ((1, 32, 320, 320), 48, (3, 3), (1, 1), (2, 2), (2, 2)),
        # ((1, 48, 320, 320), 56, (3, 3), (1, 1), (4, 4), (4, 4)),
        # ((1, 56, 320, 320), 64, (3, 3), (1, 1), (8, 8), (8, 8)),
        # ((1, 32, 320, 320), 48, (3, 3), (1, 1), (0, 0), (2, 2)),
        # ((1, 48, 320, 320), 56, (3, 3), (1, 1), (0, 0), (4, 4)),
        # ((1, 56, 320, 320), 64, (3, 3), (1, 1), (0, 0), (8, 8)),

        # ((1, 32, 1024, 64), 48, (3, 3), (1, 1), (2, 2), (2, 2)),
        # ((1, 48, 1024, 64), 56, (3, 3), (1, 1), (4, 4), (4, 4)),
        # ((1, 56, 1024, 64), 64, (3, 3), (1, 1), (8, 8), (8, 8)),

        ((1, 32, 1024, 256), 48, (3, 3), (1, 1), (2, 2), (2, 2)),
        ((1, 48, 1024, 256), 56, (3, 3), (1, 1), (4, 4), (4, 4)),
        ((1, 56, 1024, 256), 64, (3, 3), (1, 1), (8, 8), (8, 8)),

        # 1024x512 E   Out of Memory: Not enough space to allocate 35651584 B L1 buffer across 64 banks, where each bank needs to store 557056 B
        # 1024x512 E   Out of Memory: Not enough space to allocate 70287360 B L1 buffer across 64 banks, where each bank needs to store 1098240 B
        # 1024x512 E   Out of Memory: Not enough space to allocate 70287360 B L1 buffer across 64 banks, where each bank needs to store 1098240 B

        # ((1, 32, 320, 320), 48, (3, 3), (1, 1), (2, 2), (2, 2)),
        # ((1, 48, 320, 320), 56, (3, 3), (1, 1), (4, 4), (4, 4)),
        # ((1, 56, 320, 320), 64, (3, 3), (1, 1), (8, 8), (8, 8)),
    ),
)
# fmt: on

@pytest.mark.parametrize("device_params", [{"l1_small_size": 16 * 1024}], indirect=True)
def test_conv2d_split_knit_batched_dilation(
    device, input_shape_nchw, torch_tensor_map, output_channels, filter_hw, stride_hw, padding_hw, dilation_hw
):
    has_bias = True
    groups = 1
    assert all(d % 2 == 0 for d in dilation_hw), "dilation must be even"
    assert all(s == 1 for s in stride_hw), "stride must be 1"
    assert groups == 1, "groups must be 1"

    torch.manual_seed(0)
    batch_size, input_channels, input_height, input_width = input_shape_nchw

    conv_input_shape_nchw = (batch_size, input_channels, input_height, input_width)
    conv_weight_shape_oihw = (output_channels, input_channels // groups, filter_hw[0], filter_hw[1])
    conv_bias_shape = (1, 1, 1, output_channels)
    torch_input_tensor_nchw = randomize_torch_tensor(torch_tensor_map, conv_input_shape_nchw)
    # torch_input_tensor_nhwc = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))

    torch_weight_tensor_oihw = randomize_torch_tensor(torch_tensor_map, conv_weight_shape_oihw)
    torch_bias_tensor = randomize_torch_tensor(torch_tensor_map, conv_bias_shape) if has_bias else None

    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor_nchw,
        torch_weight_tensor_oihw,
        bias=torch_bias_tensor.reshape(-1) if has_bias else None,
        stride=stride_hw,
        padding=padding_hw,
        dilation=dilation_hw,
        groups=groups,
    )
    """batch, input_channels, output_channels, input_height, input_width, weights_dtype, activations_dtype, groups, kernel, stride, padding, dilation")"""
    print(
        f"# torch_conv2d:\n({torch_input_tensor_nchw.shape[0]}, {torch_weight_tensor_oihw.shape[1]}, {torch_weight_tensor_oihw.shape[0]}, {torch_input_tensor_nchw.shape[2]}, {torch_input_tensor_nchw.shape[3]}, ttnn.bfloat8_b, ttnn.bfloat8_b, {1}, ({torch_weight_tensor_oihw.shape[2]}, {torch_weight_tensor_oihw.shape[3]}), (1, 1), {padding_hw}, {dilation_hw}, True, False, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, True),"
    )

    # =================== Split-Knit Conv2d, batched conv2d approach ===================
    torch_split_knit_batched_out = torch_split_knit_batched_dilation(
        torch_input_tensor_nchw,
        torch_weight_tensor_oihw,
        torch_bias_tensor,
        filter_hw,
        stride_hw,
        padding_hw,
        dilation_hw,
        groups,
    )
    pcc = 0.999
    passing, pcc_msg = check_with_pcc_without_tensor_printout(
        torch_out_golden_tensor, torch_split_knit_batched_out, pcc=pcc
    )
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing, pcc_msg

    # =================== TTNN Split-Knit Conv2d, batched conv2d approach ===================

    ttnn_split_knit_batched_out = ttnn_split_knit_batched_dilation(
        device,
        torch_input_tensor_nchw,
        torch_weight_tensor_oihw,
        torch_bias_tensor,
        filter_hw,
        stride_hw,
        padding_hw,
        dilation_hw,
        groups,
    )
    pcc = 0.999
    passing, pcc_msg = check_with_pcc_without_tensor_printout(
        torch_out_golden_tensor, ttnn_split_knit_batched_out, pcc=pcc
    )
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing, pcc_msg
