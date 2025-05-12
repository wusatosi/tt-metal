# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
import math
from models.demos.ttnn_resnet.tt.ttnn_functional_resnet50_model_utils import get_conv_input_memory_config
from tests.ttnn.utils_for_testing import assert_with_pcc


def _nearest_y(x, y):
    return math.ceil(x / y) * y


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_work_around_fold(device):
    input_shape = (16, 3, 224, 224)
    torch_input_tensor = torch.randn(input_shape)
    tt_inputs_host = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    # ttnn fold

    # sharding
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
    tt_inputs_device = tt_inputs_host.to(device, input_mem_config)
    p(tt_inputs_device, "input_tt after sharding")

    override_fold_mem_config = get_conv_input_memory_config(
        n,
        16,  # conv_in_channels,
        115,  # conv_input_height,
        115,  # conv_input_width,
        64,  # conv_out_channels,
        112,  # conv_output_height,
        112,  # conv_output_width,
        device.compute_with_storage_grid_size(),
        16,  # chhannells_alignment,
        False,
    )
    num_cores_x, num_cores_y = 8, 8
    fold_compute_grid_size = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}
    )
    fold_output_tensor = ttnn.fold(
        tt_inputs_device,
        2,  # fold_stride_h,
        2,  # fold_stride_w,
        use_transpose_as_fold=True,
        pad_c=1,  # fold_pad_c,
        pad_h=3,  # fold_pad_h,
        pad_w=3,  # fold_pad_w,
        grid_size=fold_compute_grid_size,
        override_memory_config=override_fold_mem_config,
    )
    p(fold_output_tensor, "fold_output_tensor")

    # workaround
    input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
    tt_input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    p(tt_input_tensor, "input_tt before sharding")
    tt_input_tensor = tt_input_tensor.to(device, ttnn.L1_MEMORY_CONFIG)
    p(tt_inputs_device, "input_tt before padding")
    # new
    tt_input_tensor = ttnn.pad(tt_input_tensor, ((0, 0), (0, 6), (0, 6), (0, 1)), 0)
    tt_input_tensor = ttnn.reshape(tt_input_tensor, (16, 115, 2, 115, 2, 4))
    tt_input_tensor = ttnn.permute(tt_input_tensor, (0, 1, 3, 2, 4, 5))
    p(tt_input_tensor, "input_tt before permute")
    tt_input_tensor = ttnn.reshape(tt_input_tensor, (16, 115, 115, 16))
    p(tt_input_tensor, "input_tt after padding")
    # tt_input_tensor = ttnn.pad(
    #     tt_input_tensor, ((0, 0), (0,3), (0,3), (0, 0)), 0
    # )
    p(tt_input_tensor, "final shape")

    assert_with_pcc(ttnn.to_torch(tt_input_tensor), ttnn.to_torch(fold_output_tensor), 1.0)
