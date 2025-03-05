# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.operations.prefetcher_common import run_prefetcher_mm
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tests.ttnn.unit_tests.llama_tg.prefetcher_utils import (
    LlamaPrefetcher,
    MatmulOp,
)
from models.perf.device_perf_utils import run_device_perf_detailed
from tracy import signpost

NUM_ITERATIONS = 10


def run_pf_mm_impl(
    mesh_device,
    num_op_iters,
    num_layers,
    in0_dtype,
    in1_dtype,
    fidelity,
    fp32_acc_mode,
    packer_l1_acc,
    B,
    M,
    K,
    N,
):
    num_tensors = 1
    llama_prefetcher = LlamaPrefetcher(mesh_device, num_tensors, num_layers)
    matmul_op = MatmulOp(
        mesh_device,
        in0_dtype,
        in1_dtype,
        fidelity,
        fp32_acc_mode,
        packer_l1_acc,
        B,
        M,
        K,
        N,
    )

    ##### Run the Op with prefetcher interference #####
    def run_op():
        llama_prefetcher.run_op()

        for _ in range(num_op_iters):
            matmul_op.run_op()

        mesh_device.reset_sub_device_stall_group()

    #### Compile Model #####
    logger.info("Compiling model")
    run_op()

    ##### Capture Trace #####
    logger.info("Capturing trace")

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    run_op()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    ##### Run Trace #####
    logger.info("Running trace")
    ttnn.synchronize_devices(mesh_device)
    signpost("start")
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_devices(mesh_device)
    signpost("stop")

    mesh_device.clear_loaded_sub_device_manager()
    mesh_device.remove_sub_device_manager(llama_prefetcher.sub_device_manager)


@pytest.mark.parametrize(
    "num_op_iters, num_layers",
    [
        (NUM_ITERATIONS, 1000),
    ],
)
@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, fidelity, fp32_acc_mode, packer_l1_acc, B, M, K, N",
    [
        (
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
            True,
            True,
            1,
            32,
            2048,
            1280,
        ),
        (
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
            True,
            True,
            1,
            32,
            1280,
            2048,
        ),
        (
            ttnn.bfloat8_b,
            ttnn.bfloat4_b,
            ttnn.MathFidelity.LoFi,
            True,
            True,
            1,
            32,
            2048,
            3584,
        ),
        (
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
            False,
            True,
            1,
            32,
            3584,
            2048,
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
def test_run_pf_mm(
    mesh_device,
    num_op_iters,
    num_layers,
    in0_dtype,
    in1_dtype,
    fidelity,
    fp32_acc_mode,
    packer_l1_acc,
    B,
    M,
    K,
    N,
    use_program_cache,
    function_level_defaults,
):
    device = mesh_device.get_device(mesh_device.get_device_ids()[0])
    # Only run these tests on unharvested TG
    device_grid = (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y)
    if device_grid != (7, 10):
        pytest.skip("Skipping test_run_prefetcher because it only works with a 7x10 grid")

    mesh_device.enable_async(True)

    run_pf_mm_impl(
        mesh_device,
        num_op_iters,
        num_layers,
        in0_dtype,
        in1_dtype,
        fidelity,
        fp32_acc_mode,
        packer_l1_acc,
        B,
        M,
        K,
        N,
    )


@pytest.mark.parametrize(
    "mm_type, perf_target_us",
    [
        ("qkv", 8),
        ("do", 7.5),
        ("ff13", 9),
        ("ff2", 13.5),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_pf_mm_perf(
    mm_type,
    perf_target_us,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"matmul_{mm_type}"

    subdir = "llama_pf_mm_perf"
    command = f"pytest tests/ttnn/unit_tests/llama_tg/test_prefetched_matmul_perf.py::test_run_pf_mm -k {mm_type}"
    cols = ["DEVICE KERNEL"]
    op_name = "Matmul"

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(command, subdir, cols, op_name, has_signposts=True)
    profiler.end(step_name)
    profiler.end("run")

    # Get the measured performance
    measured_min_us = results[cols[0]]["MIN"] / 1000
    measured_max_us = results[cols[0]]["MAX"] / 1000
    measured_avg_us = results[cols[0]]["AVG"] / 1000
    measured_std_us = results[cols[0]]["STD"] / 1000

    logger.info(f"Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"matmul-{mm_type}-min-us", measured_min_us)
    benchmark_data.add_measurement(profiler, 0, step_name, f"matmul-{mm_type}-max-us", measured_max_us)
    benchmark_data.add_measurement(profiler, 0, step_name, f"matmul-{mm_type}-avg-us", measured_avg_us)
    benchmark_data.add_measurement(profiler, 0, step_name, f"matmul-{mm_type}-std-us", measured_std_us)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"matmul",
        ml_model_name="llama70b-tg-mm",
    )

    assert measured_avg_us < perf_target_us, f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"
