# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

import torch

from loguru import logger

from models.demos.ttnn_resnet.tests.resnet50_test_infra import create_test_infra
from models.utility_functions import (
    run_for_blackhole,
    is_blackhole,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    (
        (16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
        (16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),
    ),
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [True, False],
    ids=[
        "pretrained_weight_true",
        "pretrained_weight_false",
    ],
)
def test_resnet_50(
    device,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    use_pretrained_weight,
    model_location_generator,
):
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        pytest.skip("Test is not supported on n300 (8,7) grid")

    if is_blackhole() and use_pretrained_weight:
        pytest.skip(
            "Skipping pretrained weight test on blackhole due to PCC error: https://github.com/tenstorrent/tt-metal/issues/17558"
        )

    test_infra = create_test_infra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        use_pretrained_weight,
        model_location_generator=model_location_generator,
    )
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    # First run configures convs JIT
    test_infra.run()
    # Optimized run
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    # More optimized run with caching
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    passed, message = test_infra.validate()
    assert passed, message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [False],
    ids=[
        "pretrained_weight_false",
    ],
)
def test_l2m2_dummy(
    device,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    use_pretrained_weight,
    model_location_generator,
):
    test_infra = create_test_infra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        use_pretrained_weight,
        model_location_generator=model_location_generator,
    )
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)

    input_shape = (1, 1, batch_size * 28 * 28, 512)
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=act_dtype, layout=ttnn.TILE_LAYOUT)

    layer2_module2_input_shape = ttnn.Shape(input_shape)
    ## 98
    core_range_set = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(12, 6),
            ),
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 7),
                ttnn.CoreCoord(6, 7),
            ),
        }
    )
    mem_config = ttnn.create_sharded_memory_config_(
        layer2_module2_input_shape,
        core_range_set,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
        tile_layout=True,
    )

    test_infra.input_tensor = tt_inputs_host.to(device, mem_config)
    # First run configures convs JIT
    test_infra.run()
    # passed, message = test_infra.validate()
    # assert passed, message
