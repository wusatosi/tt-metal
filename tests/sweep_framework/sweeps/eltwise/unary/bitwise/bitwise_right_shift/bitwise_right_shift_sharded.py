# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial
import itertools
import torch
import random
import ttnn
import math

from tests.sweep_framework.sweep_utils.utils import (
    gen_shapes,
    sanitize_shape_rm,
    get_device_grid_size,
)
from tests.sweep_framework.sweep_utils.sharding_utils import (
    gen_unary_sharded_spec,
    parse_sharding_spec,
    invalidate_vector_sharding,
)
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


TIMEOUT = 120

Y, X = get_device_grid_size()

random.seed(0)


# expand parse_sharding_spec function for this op
def gen_sharded_spec(
    num_shapes, num_core_samples, shard_orientation, sharding_strategy, max_tensor_size_per_core=256 * 256
):
    for input_spec in gen_unary_sharded_spec(
        num_shapes, num_core_samples, shard_orientation, sharding_strategy, max_tensor_size_per_core
    ):
        input_spec["shift_bits"] = random.randint(1, 30)
        yield input_spec


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "xfail": {
        "input_spec": list(gen_sharded_spec(16, 4, "ROW_MAJOR", "TENSOR_HW"))
        + list(gen_sharded_spec(16, 4, "COL_MAJOR", "TENSOR_HW"))
        + list(gen_sharded_spec(16, 4, "ROW_MAJOR", "BLOCK"))
        + list(gen_sharded_spec(16, 4, "COL_MAJOR", "BLOCK"))
        + list(gen_sharded_spec(16, 4, "ROW_MAJOR", "HEIGHT"))
        + list(gen_sharded_spec(16, 4, "COL_MAJOR", "HEIGHT"))
        + list(gen_sharded_spec(16, 4, "ROW_MAJOR", "WIDTH"))
        + list(gen_sharded_spec(16, 4, "COL_MAJOR", "WIDTH")),
        "input_a_dtype": [ttnn.int32],
        "input_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    sharding_spec = dict(filter(lambda item: item[0] not in ["shift_bits"], test_vector["input_spec"].items()))
    input_shape, core_grid_size, shard_orientation, sharding_strategy, tensor_hw_as_shard_shape = parse_sharding_spec(
        sharding_spec
    )
    input_layout, dtype = test_vector["input_layout"], test_vector["input_a_dtype"]
    invalidated, output_str = invalidate_vector_sharding(
        input_shape, input_layout, core_grid_size, sharding_strategy, shard_orientation, tensor_hw_as_shard_shape
    )
    if invalidated:
        return True, output_str

    return False, None


def mesh_device_fixture():
    device = ttnn.open_device(device_id=0)
    assert ttnn.device.is_wormhole_b0(device), "This op is available for Wormhole_B0 only"
    yield (device, "Wormhole_B0")
    ttnn.close_device(device)
    del device


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a device_mesh_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_spec,
    input_a_dtype,
    input_layout,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    shift_bits = input_spec["shift_bits"]
    sharding_spec = dict(filter(lambda item: item[0] not in ["shift_bits"], input_spec.items()))
    input_shape, core_grid_size, shard_orientation, sharding_strategy, tensor_hw_as_shard_shape = parse_sharding_spec(
        sharding_spec
    )
    y, x = core_grid_size
    device_grid_size = ttnn.CoreGrid(y=y, x=x)

    sharded_config = ttnn.create_sharded_memory_config(
        shape=input_shape,
        core_grid=device_grid_size,
        strategy=sharding_strategy,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=tensor_hw_as_shard_shape,
    )

    if input_layout == ttnn.ROW_MAJOR_LAYOUT:
        input_shape = sanitize_shape_rm(input_shape)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-2147483647, high=2147483648, dtype=torch.int64), input_a_dtype
    )(input_shape)

    torch_output_tensor = torch.bitwise_right_shift(torch_input_tensor_a, shift_bits).to(torch.int32)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_layout,
        device=device,
        memory_config=sharded_config,
    )

    start_time = start_measuring_time()
    result = ttnn.bitwise_right_shift(input_tensor_a, shift_bits=shift_bits, memory_config=sharded_config)
    output_tensor = ttnn.to_torch(result).to(torch.int32)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]


from tests.sweep_framework.framework.permutations import *

for suite in parameters.keys():
    device_id = 0
    device = ttnn.open_device(device_id=device_id)
    suite_vectors = list(permutations(parameters[suite]))
    print(len(suite_vectors))
    passes = 0
    lowpcc = 0
    for vector in suite_vectors:
        if invalidate_vector(vector)[0]:
            continue
        passed, _ = run(**vector, device=device)
        if passed[0] != True:
            lowpcc += 1
            print(passed)
            print(vector)
        else:
            passes += 1
    print(lowpcc)
    print(passes)
    ttnn.close_device(device)
