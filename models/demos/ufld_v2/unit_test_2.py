# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
import math
import torchvision.models as models
from models.demos.ufld_v2.reference.ufld_v2_model import TuSimple34
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d
from models.utility_functions import pad_and_fold_conv_filters_for_unity_stride
from models.demos.ttnn_resnet.tt.ttnn_functional_resnet50_model_utils import get_conv_input_memory_config


def _nearest_y(x, y):
    return math.ceil(x / y) * y


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize(
    "model_name",
    [
        "resnet",
        # "ufld"
    ],
)
def test_resenet_fold(device, model_name):
    if model_name == "resnet":
        torch_input_tensor = torch.randn(16, 3, 224, 224)
        torch_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        torch_model.eval()
        stride_h, stride_w = torch_model.conv1.stride
        conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(torch_model.conv1, torch_model.bn1)
    elif model_name == "ufld":
        torch_model = TuSimple34(input_height=320, input_width=800)
        torch_model.eval()
        torch_input_tensor = torch.randn((1, 3, 320, 800))  # nchw
        stride_h, stride_w = torch_model.res_model.conv1.stride
        conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(torch_model.res_model.conv1, torch_model.res_model.bn1)
    batch_size = torch_input_tensor.shape[0]
    print("after folding", conv_weight.shape, conv_bias.shape)

    # pad & fold conv for unity stride
    conv_weight = pad_and_fold_conv_filters_for_unity_stride(conv_weight, stride_h, stride_w)
    print("after folding to unit stride", conv_weight.shape, conv_bias.shape)
    conv_in_channels = conv_weight.shape[1]
    conv_out_channels = conv_weight.shape[0]
    # tt
    tt_conv_weight = ttnn.from_torch(conv_weight)
    tt_conv_bias = ttnn.from_torch(conv_bias.reshape(1, 1, 1, -1))
    print("tt wieghts", tt_conv_weight.shape, tt_conv_bias.shape)
    p(tt_conv_weight, "tt_conv_weight")
    p(tt_conv_bias, "tt_conv_bias")

    conv_kernel_size = (4, 4)
    conv_stride = (1, 1)
    conv_padding = (0, 0)
    conv_input_height = (torch_input_tensor.shape[-2] // 2) + 3
    conv_input_width = (torch_input_tensor.shape[-1] // 2) + 3
    conv_output_height = ((conv_input_height - conv_kernel_size[0] + 2 * conv_padding[0]) // conv_stride[0]) + 1
    conv_output_width = ((conv_input_width - conv_kernel_size[1] + 2 * conv_padding[1]) // conv_stride[1]) + 1
    print(
        "conv_paramsare k,s,p,h,w,oh,ow",
        conv_kernel_size,
        conv_stride,
        conv_padding,
        conv_input_height,
        conv_input_width,
        conv_output_height,
        conv_output_width,
    )

    # fold params
    padding = 3
    fold_stride_h = stride_h
    fold_stride_w = stride_w
    _, c, h, w = torch_input_tensor.shape
    n = torch_input_tensor.shape[0]
    h += padding * 2
    w += padding * 2
    C = _nearest_y(c, 4)
    print("c and C", c, C)
    fold_pad_c = C - c
    fold_pad_h = padding
    fold_pad_w = padding
    fold_output_shape = (
        n,
        h // fold_stride_h,
        w // fold_stride_w,
        C * (fold_stride_h * fold_stride_w),
    )
    print("folding params", fold_pad_c, fold_pad_h, fold_pad_w, fold_output_shape)
    num_cores_x = 8
    num_cores_y = 8
    chhannells_alignment = 16
    # configs
    fold_compute_grid_size = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}
    )
    override_fold_mem_config = get_conv_input_memory_config(
        n,
        conv_in_channels,
        conv_input_height,
        conv_input_width,
        conv_out_channels,
        conv_output_height,
        conv_output_width,
        device.compute_with_storage_grid_size(),
        chhannells_alignment,
        False,
    )

    # input tensor
    n, c, h, w = torch_input_tensor.shape
    core_grid = ttnn.CoreGrid(y=8, x=8)
    num_cores = core_grid.x * core_grid.y
    shard_h = (n * w * h + num_cores - 1) // num_cores
    grid_size = core_grid
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(
        shard_grid, (torch_input_tensor.shape[-2], torch_input_tensor.shape[-1]), ttnn.ShardOrientation.ROW_MAJOR
    )
    input_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    print("mem config is", input_mem_config)
    # torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
    # torch_input_tensor = torch_input_tensor.reshape(1, 1, h * w * n, c)
    tt_inputs_host = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    p(tt_inputs_host, "input_tt before sharding")
    tt_inputs_host = tt_inputs_host.to(device, input_mem_config)
    p(tt_inputs_host, "input_tt after sharding")
    print("over ride config is", override_fold_mem_config)
    print(
        "input params to fold",
        tt_inputs_host.shape,
        fold_stride_h,
        fold_stride_w,
        fold_pad_c,
        fold_pad_h,
        fold_pad_w,
        fold_compute_grid_size,
    )
    fold_output_tensor = ttnn.fold(
        tt_inputs_host,
        fold_stride_h,
        fold_stride_w,
        use_transpose_as_fold=True,
        pad_c=fold_pad_c,
        pad_h=fold_pad_h,
        pad_w=fold_pad_w,
        grid_size=fold_compute_grid_size,
        override_memory_config=override_fold_mem_config,
    )  # shard shapes mis-matched
    p(fold_output_tensor, "output of fold")

    # n, c, h, w = fold_output_tensor.shape
    # fold_output_tensor = ttnn.reshape(fold_output_tensor, (1, 1, n * c * h, w))
    # p(fold_output_tensor, "fold_output_tensor after reshape")
    # conv_config = ttnn.Conv2dConfig(
    #     dtype=ttnn.bfloat16,
    #     weights_dtype=ttnn.bfloat8_b,
    #     activation="relu",
    #     deallocate_activation=True,
    #     input_channels_alignment=16,
    #     act_block_h_override=1568,
    #     transpose_shards=True,
    #     enable_act_double_buffer=True,
    #     enable_split_reader=True,
    #     enable_subblock_padding=False,
    #     shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    #     reshard_if_not_optimal=False,
    # )
    # conv1_compute_config = ttnn.init_device_compute_kernel_config(
    #     device.arch(),
    #     math_fidelity=ttnn.MathFidelity.LoFi,
    #     packer_l1_acc=True,
    # )
    # conv_kwargs = {
    #     "in_channels": conv_in_channels,
    #     "out_channels": conv_out_channels,
    #     "batch_size": batch_size,
    #     "input_height": conv_input_height,
    #     "input_width": conv_input_width,
    #     "kernel_size": conv_kernel_size,
    #     "stride": conv_stride,
    #     "padding": conv_padding,
    #     "dilation": (1, 1),
    #     "groups": 1,
    #     "device": device,
    #     "conv_config": conv_config,
    # }

    # if not ttnn.is_tensor_storage_on_device(tt_conv_weight):
    #     tt_conv_weight = ttnn.prepare_conv_weights(
    #         weight_tensor=tt_conv_weight,
    #         weights_format="OIHW",
    #         input_memory_config=fold_output_tensor.memory_config(),
    #         input_layout=fold_output_tensor.get_layout(),
    #         has_bias=True,
    #         **conv_kwargs,
    #     )

    #     tt_conv_bias = ttnn.prepare_conv_bias(
    #         bias_tensor=tt_conv_bias,
    #         input_memory_config=fold_output_tensor.memory_config(),
    #         input_layout=fold_output_tensor.get_layout(),
    #         **conv_kwargs,
    #     )
    #     tt_conv_weight = ttnn.to_device(tt_conv_weight, device)
    #     tt_conv_bias = ttnn.to_device(tt_conv_bias, device)
    #     p(tt_conv_weight, "weights")
    #     p(tt_conv_bias, "bias")
    #     x, [x_height, x_width] = ttnn.conv2d(
    #         input_tensor=fold_output_tensor,
    #         weight_tensor=tt_conv_weight,
    #         bias_tensor=tt_conv_bias,
    #         **conv_kwargs,
    #         compute_config=conv1_compute_config,
    #         return_output_dim=True,
    #         return_weights_and_bias=False,
    #     )
    #     p(x, "1st conv out")
    #     print("h,w after 1st conv", [x_height, x_width])
