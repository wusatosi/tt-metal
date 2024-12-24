# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial
import random

import torch
import ttnn

from tests.sweep_framework.sweep_utils.utils import gen_shapes, gen_rand_exclude_range, sanitize_shape_rm
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_shape": gen_shapes([1, 1, 1, 1], [6, 12, 256, 256], [1, 1, 1, 1], 8)
        + gen_shapes([1, 1, 1], [12, 256, 256], [1, 1, 1], 8)
        + gen_shapes([1, 1], [256, 256], [1, 1], 8),
        "exclude_range": [[-1, 1]],
        "grad_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_b_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "grad_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Unary operation requires tensor to be in Tile layout when working with non-sharded input tensor"
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT and (
        test_vector["input_dtype"] == ttnn.bfloat8_b or test_vector["input_dtype"] == ttnn.bfloat8_b
    ):
        return True, "bfloat8_b is only supported on tiled layout"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    exclude_range,
    grad_dtype,
    input_a_dtype,
    input_b_dtype,
    input_layout,
    grad_memory_config,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    if input_layout == ttnn.ROW_MAJOR_LAYOUT:
        input_shape = sanitize_shape_rm(input_shape)

    torch_grad_tensor = gen_func_with_cast_tt(
        partial(gen_rand_exclude_range, excluderange=exclude_range, low=-100, high=100), grad_dtype
    )(input_shape)
    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(gen_rand_exclude_range, excluderange=exclude_range, low=-100, high=100), input_a_dtype
    )(input_shape)
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(gen_rand_exclude_range, excluderange=exclude_range, low=-100, high=100), input_b_dtype
    )(input_shape)
    torch_input_tensor_a.requires_grad = True
    torch_input_tensor_b.requires_grad = True

    # while torch.any(torch_grad_tensor == 0.0):
    #   torch_grad_tensor = torch.where(torch_grad_tensor == 0.0,
    #                                  torch_grad_tensor.flatten()[random.choice(torch.where(torch_grad_tensor.flatten() != 0.0)[0])].item(),
    #                                  torch_grad_tensor)
    # while torch.any(torch_input_tensor_a == 0.0):
    #   torch_input_tensor_a = torch.where(torch_input_tensor_a == 0.0,
    #                                 torch_input_tensor_a.flatten()[random.choice(torch.where(torch_input_tensor_a.flatten() != 0.0)[0])].item(),
    #                                 torch_input_tensor_a)
    # while torch.any(torch_input_tensor_b == 0.0):
    #   torch_input_tensor_b = torch.where(torch_input_tensor_b == 0.0,
    #                                 torch_input_tensor_b.flatten()[random.choice(torch.where(torch_input_tensor_b.flatten() != 0.0)[0])].item(),
    #                                 torch_input_tensor_b)

    assert not torch.any(torch_grad_tensor == 0.0)
    assert not torch.any(torch_input_tensor_a == 0.0)
    assert not torch.any(torch_input_tensor_b == 0.0)

    golden_function = ttnn.get_golden_function(ttnn.hypot_bw)
    torch_output_tensors = golden_function(torch_grad_tensor, torch_input_tensor_a, torch_input_tensor_b)

    grad_tensor = ttnn.from_torch(
        torch_grad_tensor,
        dtype=grad_dtype,
        layout=input_layout,
        device=device,
        memory_config=grad_memory_config,
    )
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_a_memory_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_b_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_b_memory_config,
    )

    start_time = start_measuring_time()
    output_tensors = ttnn.hypot_bw(grad_tensor, input_tensor_a, input_tensor_b, memory_config=output_memory_config)
    e2e_perf = stop_measuring_time(start_time)

    passed = []
    output_string = ""
    for i in range(len(torch_output_tensors)):
        output_tensor = ttnn.to_torch(output_tensors[i])
        passed_, output_string_ = check_with_pcc(torch_output_tensors[i], output_tensor, 0.999)
        passed.append(passed_)
        output_string += output_string_ + ", "

    if all(passed):
        passed = True
    else:
        passed = False

    output_string = output_string[:-2]

    return [(passed, output_string), e2e_perf]
