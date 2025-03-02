# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.operations.prefetcher_common import run_prefetcher_mm
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.unit_tests.llama_tg.prefetcher_utils import LlamaPrefetcher


@pytest.mark.parametrize(
    "num_reader_cores, num_tensors, num_layers",
    [
        (12, 1, 1000),
    ],
)
@pytest.mark.parametrize(
    "input_shapes, dtypes, perf_target_us",
    [
        (
            [(2048, 1280)],
            [ttnn.bfloat8_b],
            10,
        ),
        (
            [(1280, 2048)],
            [ttnn.bfloat8_b],
            10,
        ),
        (
            [(2048, 3584)],
            [ttnn.bfloat4_b],
            10,
        ),
        (
            [(3584, 2048)],
            [ttnn.bfloat8_b],
            15,
        ),
    ],
    ids=[
        "qkv",
        "do",
        "ff13",
        "ff2",
    ],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((2, 2), id="2x2_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
def test_run_prefetcher(
    mesh_device,
    num_tensors,
    input_shapes,
    num_layers,
    num_reader_cores,
    dtypes,
    perf_target_us,
    use_program_cache,
    function_level_defaults,
):
    device = mesh_device.get_device(mesh_device.get_device_ids()[0])
    # Only run these tests on unharvested TG
    device_grid = (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y)
    if device_grid != (7, 10):
        pytest.skip("Skipping test_run_prefetcher because it only works with a 7x10 grid")

    assert num_tensors == 1, "This should only be run with 1 tensor at a time, because it is meant to profile the op"

    llama_prefetcher = LlamaPrefetcher(mesh_device, num_tensors, num_layers)

    ttnn.dram_prefetcher(
        llama_prefetcher.input_tensors,
        num_layers,
        llama_prefetcher.global_circular_buffer,
        non_blocking=True,
    )
    logger.info("Prefetcher setup complete")

    ttnn.synchronize_devices(mesh_device)
    mesh_device.clear_loaded_sub_device_manager()
    mesh_device.remove_sub_device_manager(llama_prefetcher.sub_device_manager)

    # profiler = BenchmarkProfiler()
    # run_prefetcher_mm(
    #     mesh_device,
    #     num_tensors,
    #     input_shapes,
    #     num_layers,
    #     num_reader_cores,
    #     dtypes,
    #     profiler=profiler,
    #     measure_perf=True
    # )

    # time_taken = profiler.get_duration("op-duration")
    # iters = num_layers
    # latency_us = time_taken / iters * 1e6
    # logger.info(f"Time taken: {time_taken} s")
    # logger.info(f"Time per iter: {latency_us} us")

    # assert latency_us < perf_target_us, f"Latency {latency_us} us is greater than target {perf_target_us} us"
