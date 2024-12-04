# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import math
import ttnn


def run_interleaved_to_sharded(
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    dtype,
):
    in_n, in_c, in_h, in_w = act_shape
    kernel_h, kernel_w = kernel_size
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32)

    # construct the tensor in NCHW shape
    act = torch.empty(act_shape, dtype=torch.bfloat16)
    for n in range(act_shape[0]):
        for c in range(act_shape[1]):
            for h in range(act_shape[2]):
                for w in range(act_shape[3]):
                    act[n, c, h, w] = w + h * in_w
    act_shape = (1, 1, in_n * in_h * in_w, in_c)
    act_permuted = torch.permute(act, (0, 2, 3, 1))
    act_reshaped = act_permuted.reshape(act_shape)
    tt_act = ttnn.from_torch(act_reshaped, dtype)
    tt_act_device = ttnn.to_device(tt_act, device)

    # shard the tensor
    out_h = math.floor((in_h + 2 * pad_h - (dilation_h * kernel_h - 1) - 1) / stride_h) + 1
    out_w = math.floor((in_w + 2 * pad_w - (dilation_w * kernel_w - 1) - 1) / stride_w) + 1
    parallel_config = ttnn._ttnn.operations.conv2d.determine_parallel_config(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        batch_size=in_n,
        input_channels=in_c,
        output_height=out_h,
        output_width=out_w,
        output_channels=in_c,
        device=device,
        block_shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        is_out_tiled=False,
    )
    sharded_memory_config = ttnn._ttnn.operations.conv2d.create_sharded_memory_config_from_parallel_config(
        tensor_shape=tt_act_device.shape,
        parallel_config=parallel_config,
        tile_size=32 if dtype == ttnn.bfloat8_b else 1,
    )
    tt_sharded_device = ttnn.to_memory_config(tt_act_device, sharded_memory_config)
    tt_sharded_host = tt_sharded_device.cpu()
    sharded_host = torch.Tensor(ttnn.to_torch(tt_sharded_host))

    print(parallel_config)
    print(sharded_memory_config)
    print(sharded_host)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "act_shape",  ## NCHW
    (
        (
            [1, 64, 14, 14],
            [1, 40, 14, 14],
        )
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    ((2, 2),),
)
@pytest.mark.parametrize(
    "padding",
    ((0, 0),),
)
@pytest.mark.parametrize(
    "stride",
    ((2, 2),),
)
@pytest.mark.parametrize("dilation", ((1, 1),))
@pytest.mark.parametrize("dtype", (ttnn.bfloat16,))
def test_run_interleaved_to_sharded(
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    dtype,
):
    run_interleaved_to_sharded(
        act_shape,
        kernel_size,
        padding,
        stride,
        dilation,
        device,
        dtype,
    )
