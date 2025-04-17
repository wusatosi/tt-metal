# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

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

try:
    from tracy import signpost
except ModuleNotFoundError:
    # Dummy function to avoid import error if tracy is not available
    def signpost(*args, **kwargs):
        pass


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


# fmt: off
@pytest.mark.parametrize(
    "input_shape_nchw, output_channels, filter_hw, stride_hw, padding_hw, dilation_hw, pcc",
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

        ((1, 32, 1024, 256), 48, (3, 3), (1, 1), (2, 2), (2, 2), 0.999),
        ((1, 48, 1024, 256), 56, (3, 3), (1, 1), (4, 4), (4, 4), 0.999),
        ((1, 56, 1024, 256), 64, (3, 3), (1, 1), (8, 8), (8, 8), 0.998), # flaky

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
def test_split_knit_batched_dilation_conv2d(
    device, input_shape_nchw, torch_tensor_map, output_channels, filter_hw, stride_hw, padding_hw, dilation_hw, pcc
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

    # =================== TTNN conv2d, default approach ===================

    # =================== Split-Knit Conv2d, batched conv2d approach ===================
    # torch_split_knit_batched_out = ttnn.get_golden_function(ttnn.experimental.split_knit_batch_dilation_conv2d)(
    #     torch_input_tensor_nchw,
    #     torch_weight_tensor_oihw,
    #     torch_bias_tensor,
    #     filter_hw,
    #     stride_hw,
    #     padding_hw,
    #     dilation_hw,
    #     groups,
    # )
    # pcc = 0.999
    # passing, pcc_msg = check_with_pcc_without_tensor_printout(
    #     torch_out_golden_tensor, torch_split_knit_batched_out, pcc=pcc
    # )
    # logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    # assert passing, pcc_msg

    # =================== TTNN Split-Knit Conv2d, batched conv2d approach ===================

    ttnn_split_knit_batched_out = ttnn.experimental.split_knit_batch_dilation_conv2d(
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
    passing, pcc_msg = check_with_pcc_without_tensor_printout(
        torch_out_golden_tensor, ttnn_split_knit_batched_out, pcc=pcc
    )
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing, pcc_msg
