# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.operations.prefetcher_common import run_prefetcher_mm
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.unit_tests.llama_tg.prefetcher_utils import (
    LlamaPrefetcher,
    MatmulOp,
)


@pytest.mark.parametrize(
    "num_op_iters, num_layers",
    [
        (10, 1000),
    ],
)
@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, fidelity, fp32_acc_mode, packer_l1_acc, B, M, K, N, perf_target_us",
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
            1024,
            10,
        ),
        (
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
            True,
            True,
            1,
            32,
            1024,
            2048,
            10,
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
            10,
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
            10,
        ),
    ],
    ids=[
        "qkv",
        "do",
        "ff13",
        "ff2",
    ],
)
# @pytest.mark.parametrize("mesh_device", [pytest.param((2, 2), id="2x2_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
def test_run_prefetcher(
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
    perf_target_us,
    use_program_cache,
    function_level_defaults,
):
    device = mesh_device.get_device(mesh_device.get_device_ids()[0])
    # Only run these tests on unharvested TG
    device_grid = (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y)
    if device_grid != (7, 10):
        pytest.skip("Skipping test_run_prefetcher because it only works with a 7x10 grid")

    mesh_device.enable_async(True)

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
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_devices(mesh_device)

    mesh_device.clear_loaded_sub_device_manager()
    mesh_device.remove_sub_device_manager(llama_prefetcher.sub_device_manager)
