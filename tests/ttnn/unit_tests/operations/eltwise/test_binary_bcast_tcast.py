# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from functools import partial
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import compare_pcc
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.utility_functions import torch_random


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1])),
        (torch.Size([5, 1, 1]), torch.Size([1, 32, 128])),
    ),
)
@pytest.mark.parametrize(
    "dtype",
    ([ttnn.int32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32, ttnn.bfloat4_b]),
)
def test_binary_scalar_ops(input_shapes, dtype, device):
    torch.manual_seed(0)
    a_shape, b_shape = input_shapes

    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), dtype)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), dtype)(b_shape)

    a_tt = ttnn.from_torch(
        a_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    b_tt = ttnn.from_torch(
        b_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    cq_id = 0
    out_tt = ttnn.add(a_tt, b_tt, queue_id=cq_id, use_legacy=False)
    out_pt = a_pt + b_pt

    comp_pass = compare_pcc([out_tt], [out_pt])
    assert comp_pass


def filled_tensor(shape, base):
    base = base.flatten()
    total = torch.tensor(shape, dtype=torch.int32).prod().item()
    repeated = base.repeat((total // len(base)) + 1)[:total]
    return repeated.reshape(shape)


@pytest.mark.parametrize(
    "shape, sub_core_grid",
    [
        # (
        #     (torch.Size([1, 2, 32, 960])),  # 60 tiles - 2 tiles / 30 cores
        #     ttnn.CoreRangeSet(
        #         [
        #             ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 7)),  # 8 * 3 cores
        #             ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 7)),  # 8 * 2 cores
        #         ]
        #     ),
        # ),
        # (
        #     (torch.Size([1, 7, 32, 96])),  # 21 tiles - 3 tiles/7 cores
        #     ttnn.CoreRangeSet(
        #         [
        #             ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 7)),  # 8 cores
        #         ]
        #     ),
        # ),
        # (
        #     (torch.Size([1, 3, 32, 96])),  # 9 tiles - 3 tiles/core
        #     ttnn.CoreRangeSet(
        #         [
        #             ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 2)),  # 3 cores
        #         ]
        #     ),
        # ),
        # (
        #     (torch.Size([1, 8, 32, 128])),  # 32 tiles - 4 tiles/core
        #     ttnn.CoreRangeSet(
        #         [
        #             ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 7)),  # 8 cores
        #         ]
        #     ),
        # ),
        (
            (torch.Size([1, 101, 32, 32])),  # 17 tiles  # prime num - single core case
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 7)),  # 8 cores
                ]
            ),
        ),
    ],
)
def test_typecast_subcore_grid(device, shape, sub_core_grid):
    torch.manual_seed(0)

    base1 = torch.tensor([[[[700, 100, 65000, 9500]]]], dtype=torch.int32)
    in_data1 = filled_tensor(shape, base1)
    base2 = torch.tensor([[[[70000, 1000, 65000, 95000]]]], dtype=torch.int32)
    in_data2 = filled_tensor(shape, base2)
    # in_data2 = torch.rand((input_shapes), dtype=torch.float32) * (high - low + 10) + low

    input_mem_config = ttnn.DRAM_MEMORY_CONFIG

    input_tensor1 = ttnn.from_torch(
        in_data1,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )
    input_tensor2 = ttnn.from_torch(
        in_data2,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    input_tensor3 = ttnn.typecast(
        input_tensor1,
        ttnn.uint32,
        memory_config=input_mem_config,
        # sub_core_grids=sub_core_grid,
    )
    # print("output_tensor0", input_tensor3)
    input_tensor3 = ttnn.typecast(
        input_tensor3,
        ttnn.int32,
        memory_config=input_mem_config,
        # sub_core_grids=sub_core_grid,
    )
    # # print("output_tensor0", input_tensor3)
    # output_tensor = ttnn.add(input_tensor3, input_tensor2)
    # # print("output_tensor", output_tensor)
    # output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)
    # golden_function = ttnn.get_golden_function(ttnn.add)
    # golden_tensor = golden_function(in_data1, in_data2)

    # # print("golden_tensor", golden_tensor)
    # # print("output_tensor", output_tensor)

    # assert torch.equal(golden_tensor, output_tensor)


@pytest.mark.parametrize(
    "shape, sub_core_grid",
    [
        (
            (torch.Size([1, 101, 4, 4])),  # 101 tiles  # prime num - single core case
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 7)),  # 8 cores
                ]
            ),
        ),
    ],
)
def test_untilize_subcore_grid(device, shape, sub_core_grid):
    torch.manual_seed(0)

    base1 = torch.tensor([[[[700, 100, 65000, 9500]]]], dtype=torch.int32)
    in_data1 = filled_tensor(shape, base1)
    base2 = torch.tensor([[[[70000, 1000, 65000, 95000]]]], dtype=torch.int32)
    in_data2 = filled_tensor(shape, base2)
    # in_data2 = torch.rand((input_shapes), dtype=torch.float32) * (high - low + 10) + low

    input_mem_config = ttnn.DRAM_MEMORY_CONFIG

    input_tensor1 = ttnn.from_torch(
        in_data1,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    input_tensor3 = ttnn.untilize(
        input_tensor1,
        memory_config=input_mem_config,
        sub_core_grids=sub_core_grid,
    )
    print("output_tensor0", input_tensor3)
