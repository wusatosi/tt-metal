# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
from models.utility_functions import (
    is_wormhole_b0,
    skip_for_grayskull,
)
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
import ttnn

torch.set_printoptions(linewidth=400, profile="full", sci_mode=False)


@pytest.mark.parametrize(
    "input_shape_nhwc",
    (((1, 9, 65, 4)),),
)
@pytest.mark.parametrize("num_output_channels_after_knit", [1])
def test_conv_split_knit(device, input_shape_nhwc, num_output_channels_after_knit):
    torch.manual_seed(0)

    B = input_shape_nhwc[0]
    H = input_shape_nhwc[1]
    W = input_shape_nhwc[2]
    C = input_shape_nhwc[3]

    torch_input_tensor = torch.randn([1, 1, B * H * W, C], dtype=torch.bfloat16).float()

    tensor_permuted = torch_input_tensor.reshape([B, H, W, C])
    tensor_permuted = tensor_permuted.permute(0, 3, 1, 2)

    ref_knit_tensor_out = torch.empty(
        tensor_permuted.shape[0],
        tensor_permuted.shape[1] // C,
        tensor_permuted.shape[2] * (C // 2),
        tensor_permuted.shape[3] * (C // 2),
        dtype=torch.bfloat16,
    )

    ref_knit_tensor_out[:, :, 0::2, 0::2] = tensor_permuted[:, 0, :]
    ref_knit_tensor_out[:, :, 0::2, 1::2] = tensor_permuted[:, 1, :]
    ref_knit_tensor_out[:, :, 1::2, 0::2] = tensor_permuted[:, 2, :]
    ref_knit_tensor_out[:, :, 1::2, 1::2] = tensor_permuted[:, 3, :]

    # pad torch_input_tensor channels to 32

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    tt_input_tensor = ttnn.pad(
        tt_input_tensor,
        [tt_input_tensor.shape[0], tt_input_tensor.shape[1], tt_input_tensor.shape[2], 32],
        [0, 0, 0, 0],
        0,
    )

    core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_shape = (
        tt_input_tensor.shape[2] // core_range_set.num_cores(),
        tt_input_tensor.shape[3],
    )
    tt_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_range_set,
            shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, tt_mem_config)
    tt_knited_tensor = ttnn.conv_knit(tt_input_tensor, 2, 1, W, C)
    ttnn.synchronize_device(device)

    tt_knited_tensor_out = ttnn.to_torch(tt_knited_tensor, mesh_composer=None)
    tt_knited_tensor_out = tt_knited_tensor_out.reshape(ref_knit_tensor_out.shape)
    print("Out shape is: ", tt_knited_tensor_out.shape)

    row_id = 4
    print("TT output  is:", tt_knited_tensor_out[:, :, row_id, :])
    print("Ref is:", ref_knit_tensor_out[:, :, row_id, :])

    pcc = 0.999
    passing, pcc_msg = check_with_pcc_without_tensor_printout(tt_knited_tensor_out, ref_knit_tensor_out, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    # assert passing
