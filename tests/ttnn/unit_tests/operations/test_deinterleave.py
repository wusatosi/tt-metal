# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn._ttnn
from models.utility_functions import comp_allclose_and_pcc, comp_pcc
from loguru import logger

from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    get_lib_dtype,
)


def torch_deinterleave_to_batch(torch_input_nhwc, stride_hw):
    torch_deinterleaved_to_batch = torch.zeros(
        torch_input_nhwc.shape[0] * stride_hw[0] * stride_hw[1],
        torch_input_nhwc.shape[1] // stride_hw[0],
        torch_input_nhwc.shape[2] // stride_hw[1],
        torch_input_nhwc.shape[3],
    )

    print(f"torch_deinterleaved_to_batch shape: {torch_deinterleaved_to_batch.shape}")
    print(f"torch_input_nhwc shape: {torch_input_nhwc.shape}")
    for src_batch in range(torch_input_nhwc.shape[0]):
        for split_h in range(stride_hw[0]):
            for split_w in range(stride_hw[1]):
                batch_idx = src_batch * stride_hw[0] * stride_hw[1] + split_h * stride_hw[1] + split_w
                torch_deinterleaved_to_batch[batch_idx, :, :, :] = torch_input_nhwc[
                    src_batch,
                    split_h :: stride_hw[0],
                    split_w :: stride_hw[1],
                    :,
                ]
    return torch_deinterleaved_to_batch


def run_deinterleave(
    shape_nhwc,
    input_memory_config,
    stride_hw,
    device,
):
    input_dtype = "bfloat16"
    # torch_input = 2 * torch.rand(size=shape_nhwc, dtype=get_lib_dtype(torch, input_dtype)) - 1
    torch_input = torch.ones(size=shape_nhwc, dtype=get_lib_dtype(torch, input_dtype))
    torch_input[:, ::2, ::2, :] = 10
    torch_input[:, ::2, 1::2, :] = 20
    torch_input[:, 1::2, ::2, :] = 30
    torch_input[:, 1::2, 1::2, :] = 40

    print(f"input={torch_input}")

    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=get_lib_dtype(ttnn, input_dtype),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=input_memory_config,
    ).to(device)

    # print(f"Input tensor mem: {ttnn_input.memory_config()}")
    compute_kernel_options = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    print(f"shard_shape {input_memory_config.shard_spec.shape}")
    print(f"shard_spec mode {input_memory_config.shard_spec.mode}")

    ttnn_output = ttnn.experimental.deinterleave(
        ttnn_input,
        compute_kernel_config=compute_kernel_options,
        stride_hw=stride_hw,
    )

    torch_output = ttnn.to_torch(ttnn_output)  # .reshape(shape)

    print(f"ttnn_output shape={ttnn_output.shape}")
    print(f"torch_output {torch_output[:,:,:,:]}")

    # passing, out = comp_allclose_and_pcc(torch.ops.aten.clone(torch_input), torch_output, rtol=0.01, atol=0.01)
    # passing, out = comp_allclose_and_pcc(torch_deinterleave_to_batch(torch_input), torch_output, pcc=0.999)
    # logger.info(out)

    torch_output = torch_output.view(  # TBD where to do this
        torch_input.shape[0] * stride_hw[0] * stride_hw[1],
        torch_input.shape[1] // stride_hw[0],
        torch_input.shape[2] // stride_hw[1],
        torch_input.shape[3],
    )

    print(f"torch_output 0 {torch_output[0,:,:,:]}")
    print(f"torch_output 1 {torch_output[1,:,:,:]}")
    print(f"torch_output 2 {torch_output[2,:,:,:]}")
    print(f"torch_output 3 {torch_output[3,:,:,:]}")
    # assert passing, out

    golden_output = torch_deinterleave_to_batch(torch_input, stride_hw)
    # print(f"golden={golden_output}")
    # print("============")
    print(f"golden_shape={golden_output.shape}")
    print(f"torch_shape={torch_output.shape}")
    passing, out = comp_allclose_and_pcc(golden_output, torch_output, rtol=0.01, atol=0.01, pcc=0.999)
    logger.info(out)
    assert passing


@pytest.mark.parametrize(
    "shape",
    [
        # [1, 256, 1024, 32],
        # [1, 256//64, 1024, 32],
        # [1, 4, 32, 32],
        # [1, 1024//64, 256, 64],  # 61 us
        # [1, 1024//64, 256, 32],  # 61 us
        # [1, 1024//64, 128, 64],  # 31 us
        # [1, 1024//64, 128, 32],  # 31 us
        [1, 16, 128, 32],  # _ us
        [1, 16, 128, 64],  # _ us
        [1, 16, 128, 128],  # _ us
        # [1, 2, 64, 32],   # _ us
        # [1, 2, 64, 64],   # _ us
        # [1, 2, 64, 128],  # _ us
        # [1, 2, 64, 256],  # _ us
        # [1, 2, 64, 512],  # _ us
        # [1, 2, 64, 1024],  # _ us
    ],
)
def test_deinterleave_shape(
    shape,
    device,
):
    torch.manual_seed(2025)

    memory_config = ttnn.create_sharded_memory_config_(
        shape=shape,
        core_grid=ttnn.CoreGrid(x=1, y=1),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        strategy=ttnn.ShardStrategy.HEIGHT,
    )

    print(f"Memory config: {memory_config}")

    run_deinterleave(
        shape,
        memory_config,
        [2, 2],
        device,
    )
