# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import typing
import pytest
import ttnn
import tempfile
from loguru import logger
import os
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor

NUM_TRACE_LOOPS = int(os.getenv("NUM_TRACE_LOOPS", 15))


@pytest.mark.parametrize(
    "shape", [(1, 1, 512, 512), (1, 1, 32, 32), (1, 3, 32, 32), (1, 1, 256, 256), (1, 3, 512, 512), (1, 3, 128, 128)]
)
@pytest.mark.parametrize("use_all_gather", [True, False])
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("enable_multi_cq", [True, False])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 60000, "num_command_queues": 2}], indirect=True)
def test_multi_device_single_trace(t3k_mesh_device, shape, use_all_gather, enable_async, enable_multi_cq):
    if t3k_mesh_device.get_num_devices() <= 1:
        pytest.skip("This test requires multiple devices")

    # Trace requires program cache to be enabled
    t3k_mesh_device.enable_async(enable_async)
    t3k_mesh_device.enable_program_cache()

    # Preallocate activation tensors. These will be used when capturing and executing the trace
    input_0_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, t3k_mesh_device)
    input_1_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, t3k_mesh_device)

    # Op chains to be traced
    def run_op_chain(input_0, input_1):
        single_dev_output = ttnn.neg(input_0)
        # single_dev_output = ttnn.neg(ttnn.add(ttnn.mul(input_1, ttnn.neg(ttnn.gelu(input_0))), ttnn.relu(input_1)))
        if use_all_gather:
            return ttnn.all_gather(single_dev_output, dim=0, num_links=1)
        return single_dev_output

    if enable_multi_cq:
        trace_cq = 0
        data_movement_cq = 1

        def event_sync(device, record_cq, wait_cq):
            event = ttnn.record_event(device, record_cq)
            ttnn.wait_for_event(wait_cq, event)

    else:
        trace_cq = 0
        data_movement_cq = 0

        def event_sync(device, record_cq, wait_cq):
            pass

    # Compile program binaries
    run_op_chain(input_0_dev, input_1_dev)

    # Capture Trace
    logger.info("Capture Trace")

    tid = ttnn.begin_trace_capture(t3k_mesh_device, cq_id=trace_cq)
    for i in range(10):
        output_tensor = run_op_chain(input_0_dev, input_1_dev)
    ttnn.end_trace_capture(t3k_mesh_device, tid, cq_id=trace_cq)
    logger.info("Done Trace Capture")

    for i in range(15):
        # Create torch inputs
        torch_input_tensor_0 = torch.rand(
            (t3k_mesh_device.get_num_devices(), shape[1], shape[2], shape[3]), dtype=torch.bfloat16
        )
        torch_input_tensor_1 = torch.rand(
            (t3k_mesh_device.get_num_devices(), shape[1], shape[2], shape[3]), dtype=torch.bfloat16
        )
        # Compute PT Golden
        torch_output_golden = torch.neg(torch_input_tensor_0)
        # torch_output_golden = torch.neg(
        # torch.add(
        # torch.mul(torch_input_tensor_1, torch.neg(torch.nn.functional.gelu(torch_input_tensor_0))),
        # torch.relu(torch_input_tensor_1),
        # )
        # )
        # Convert torch tensors to TTNN Multi-Device Host Tensors
        ttnn_input_tensor_0 = ttnn.from_torch(
            torch_input_tensor_0, layout=ttnn.TILE_LAYOUT, mesh_mapper=ShardTensorToMesh(t3k_mesh_device, dim=0)
        )
        ttnn_input_tensor_1 = ttnn.from_torch(
            torch_input_tensor_1, layout=ttnn.TILE_LAYOUT, mesh_mapper=ShardTensorToMesh(t3k_mesh_device, dim=0)
        )

        # Copy TTNN host tensors into preallocated Mult-Device tensors
        logger.info("Send Inputs to Device")
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_0, input_0_dev, cq_id=data_movement_cq)
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_1, input_1_dev, cq_id=data_movement_cq)
        event_sync(t3k_mesh_device, data_movement_cq, trace_cq)
        logger.info("Execute Trace")
        # Execute trace
        ttnn.execute_trace(t3k_mesh_device, tid, cq_id=trace_cq, blocking=False)
        event_sync(t3k_mesh_device, trace_cq, data_movement_cq)
        if use_all_gather:
            # Device All-Gather: Iterate through tensors on all devices. Ensure they match the full tensor
            logger.info("Read Back Trace Outputs with All Gather")
            device_tensors: typing.List[ttnn.Tensor] = ttnn.get_device_tensors(output_tensor)
            for device_tensor in device_tensors:
                device_tensor_torch = ttnn.to_torch(device_tensor, cq_id=data_movement_cq)
                assert_with_pcc(device_tensor_torch, torch_output_golden, pcc=0.99)

        else:
            # Perform host All-Gather
            logger.info("Read Back Trace Outputs")
            ttnn_torch_output_tensor = ttnn.to_torch(
                output_tensor,
                mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=0),
                device=t3k_mesh_device,
                cq_id=data_movement_cq,
            )
            assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.96)

    # Release trace buffer once workload is complete
    ttnn.release_trace(t3k_mesh_device, tid)

    t3k_mesh_device.enable_async(False)
