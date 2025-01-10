# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.utility_functions import run_for_wormhole_b0
from models.demos.wormhole.stable_diffusion_3_5.tests.perf_e2e_stable_diffusion3_5 import run_perf_sd35


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 1500000}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, enable_async_mode, expected_inference_time, expected_compile_time",
    (
        (1, True, 0.005, 30),
        # (1, False, 0.0046, 30),
    ),
    indirect=["enable_async_mode"],
)
def test_perf_trace(
    device,
    use_program_cache,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    enable_async_mode,
):
    mode = "async" if enable_async_mode else "sync"
    run_perf_sd35(
        batch_size,
        expected_inference_time,
        expected_compile_time,
        device,
        mode,
    )
