# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
import math
from models.demos.ufld_v2.reference.ufld_v2_model import TuSimple34
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d
from models.utility_functions import pad_and_fold_conv_filters_for_unity_stride


def _nearest_y(x, y):
    return math.ceil(x / y) * y


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_folded_conv(device):
    # Model
    torch_model = TuSimple34(input_height=320, input_width=800)
    torch_model.eval()
    stride_h, stride_w = torch_model.res_model.conv1.stride
    kernel_h, kernel_w = torch_model.res_model.conv1.kernel_size
    print("before folding", torch_model.res_model.conv1, torch_model.res_model.bn1)

    # folding conv+bn
    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(torch_model.res_model.conv1, torch_model.res_model.bn1)
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

    # new attrib
    conv_kernel_size = (4, 4)
    conv_stride = (1, 1)
    conv_padding = (0, 0)
    conv_input_height = 163
    conv_input_width = 403
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
    _, c, h, w = (1, 3, 320, 800)
    n = 1
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
    # override_fold_mem_config = get_conv_input_memory_config(
    #     n,
    #     conv_in_channels,
    #     conv_input_height,
    #     conv_input_width,
    #     conv_out_channels,
    #     conv_output_height,
    #     conv_output_width,
    #     device.compute_with_storage_grid_size(),
    #     chhannells_alignment,
    #     False,
    # )

    # input tensor
    torch_input_tensor = torch.randn((1, 3, 320, 800))
    n, c, h, w = torch_input_tensor.shape
    core_grid = ttnn.CoreGrid(y=8, x=8)
    num_cores = core_grid.x * core_grid.y
    shard_h = (n * w * h + num_cores - 1) // num_cores
    grid_size = core_grid
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, 16), ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
    torch_input_tensor = torch_input_tensor.reshape(1, 1, h * w * n, c)
    tt_inputs_host = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    tt_inputs_host = ttnn.pad(
        tt_inputs_host, ((0, 0), (0, 0), (0, 0), (0, 13)), 0
    )  # ttnn.pad(tt_inputs_host, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)
    p(tt_inputs_host, "input_tt before sharding")
    tt_inputs_host = tt_inputs_host.to(device, ttnn.DRAM_MEMORY_CONFIG)
    p(tt_inputs_host, "input_tt after sharding")
    # print("over ride config is", override_fold_mem_config)
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
        override_memory_config=ttnn.DRAM_MEMORY_CONFIG,  # override_fold_mem_config,
    )  # shard shapes mis-matched
