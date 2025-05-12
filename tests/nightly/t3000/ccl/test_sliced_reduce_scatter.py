# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger

from tests.ttnn.unit_tests.operations.ccl.test_new_all_reduce import check_mesh_tensor_alloc
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc


def run_with_trace(
    mesh_device,
    topology,
    input_tensor,
    persistent_intermediate_buffer,
    persistent_output_buffer,
    scatter_dim,
    num_links,
    output_mem_config,
    multi_device_global_semaphore,
    subdevice_id=None,
):
    # Compile Run
    logger.info("Compiling model")
    for i in range(len(input_tensor)):
        tt_out_tensor = ttnn.experimental.sliced_reduce_scatter_async(
            input_tensor[i],
            scatter_dim=scatter_dim,
            persistent_intermediate_buffer=persistent_intermediate_buffer[i],
            persistent_output_buffer=persistent_output_buffer[i],
            multi_device_global_semaphore=multi_device_global_semaphore[i],
            num_links=num_links,
            memory_config=output_mem_config,
            topology=topology,
            subdevice_id=subdevice_id,
        )
    ttnn.synchronize_device(mesh_device)

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    tt_out_tensor_list = []
    for i in range(len(input_tensor)):
        tt_out_tensor = ttnn.experimental.sliced_reduce_scatter_async(
            input_tensor[i],
            scatter_dim=scatter_dim,
            persistent_intermediate_buffer=persistent_intermediate_buffer[i],
            persistent_output_buffer=persistent_output_buffer[i],
            multi_device_global_semaphore=multi_device_global_semaphore[i],
            num_links=num_links,
            memory_config=output_mem_config,
            topology=topology,
            subdevice_id=subdevice_id,
        )
        tt_out_tensor_list.append(tt_out_tensor)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Run the op
    logger.info("Starting Trace perf test...")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)

    return tt_out_tensor_list


def run_sliced_reduce_scatter_impl(
    mesh_device,
    num_devices,
    logical_shape,
    scatter_dim,
    num_links,
    input_dtype,
    layout,
    topology,
    num_iters=1,
    trace_mode=False,
    mem_config=None,
    do_check=True,
    reuse_inputs=False,
):
    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)
    # create global semaphore handles
    ccl_semaphore_handles = [
        [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_devices)]
        for _ in range(num_iters)
    ]

    logger.info(f"Logical shape: {logical_shape}")
    logger.info(f"scatter_dim: {scatter_dim}")

    input_mem_config = mem_config
    output_mem_config = mem_config
    ###

    ### Create persistent output buffers
    logger.info("Creating persistent buffers")
    output_shape = list(logical_shape)
    output_shape[scatter_dim] //= num_devices
    persistent_intermediate_buffers = [
        ttnn.from_torch(
            torch.zeros(logical_shape),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=input_dtype,
            memory_config=output_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for _ in range(num_iters)
    ]
    persistent_output_buffers = [
        ttnn.from_torch(
            torch.zeros(output_shape),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=input_dtype,
            memory_config=output_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for _ in range(num_iters)
    ]

    for im_buf, out_buf in zip(persistent_intermediate_buffers, persistent_output_buffers):
        check_mesh_tensor_alloc(im_buf)
        check_mesh_tensor_alloc(out_buf)

    logger.info("Done creating persistent buffers")

    input_tensor_mesh_list = []
    output_tensor_goldens_list = []

    for i in range(num_iters if not reuse_inputs else 1):
        input_tensors = [torch.rand(logical_shape).bfloat16() for _ in range(num_devices)]

        output_tensor_goldens_list.append(torch.stack(input_tensors, dim=0).sum(dim=0).chunk(num_devices, scatter_dim))

        tt_input_tensors = []
        for i, t in enumerate(input_tensors):
            tt_input_tensors.append(ttnn.Tensor(t, input_dtype).to(layout))
            logger.info(f"using device {mesh_device.get_device_ids()[i]}")

        input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors).to(mesh_device, input_mem_config)

        input_tensor_mesh_list.append(input_tensor_mesh)

    tt_out_tensor_list = []
    if trace_mode:
        tt_out_tensor_list = run_with_trace(
            mesh_device,
            topology,
            input_tensor_mesh_list,
            persistent_intermediate_buffers,
            persistent_output_buffers,
            scatter_dim,
            num_links,
            output_mem_config,
            multi_device_global_semaphore=ccl_semaphore_handles,
            subdevice_id=worker_sub_device_id,
        )
    else:
        for i in range(num_iters):
            tt_out_tensor = ttnn.experimental.sliced_reduce_scatter_async(
                input_tensor_mesh_list[i if not reuse_inputs else 0],
                persistent_intermediate_buffer=persistent_intermediate_buffers[i],
                persistent_output_buffer=persistent_output_buffers[i],
                scatter_dim=scatter_dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                num_links=num_links,
                memory_config=output_mem_config,
                topology=topology,
                subdevice_id=worker_sub_device_id,
            )
            tt_out_tensor_list.append(persistent_output_buffers[i])

        logger.info(f"Waiting for op")
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done op")

    passed = True
    if do_check:
        for tensor_index in range(len(tt_out_tensor_list)):
            tt_out_tensor = tt_out_tensor_list[tensor_index]
            output_tensors = output_tensor_goldens_list[tensor_index if not reuse_inputs else 0]
            for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
                tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
                output_tensor = output_tensors[i]
                intermediate_tensor = (
                    ttnn.get_device_tensors(persistent_intermediate_buffers[tensor_index])[i]
                    .cpu()
                    .to(ttnn.ROW_MAJOR_LAYOUT)
                    .to_torch()
                )
                input_tensor = (
                    ttnn.get_device_tensors(input_tensor_mesh_list[tensor_index])[i]
                    .cpu()
                    .to(ttnn.ROW_MAJOR_LAYOUT)
                    .to_torch()
                )
                logger.info(f"Checking for device {t.device().id()}")
                eq, output = comp_pcc(tt_output_tensor, output_tensor)
                if not eq:
                    logger.error(f"output mismatch for tensor {i}")
                    passed = False

    assert (
        mesh_device.num_program_cache_entries() == num_iters
    ), f"Device has {mesh_device.num_program_cache_entries()} program cache entries"

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    if do_check and not passed:
        assert eq, f"{i} FAILED: {output}"


@pytest.mark.parametrize(
    "num_devices, num_links, logical_shape, scatter_dim, layout",
    [
        (8, 1, [1, 1, 4096, 2560], 3, ttnn.TILE_LAYOUT),  # SD3.5 vision FF2 output
        (8, 1, [1, 1, 320, 2560], 3, ttnn.TILE_LAYOUT),  # SD3.5 prompt FF2 output
        (8, 1, [1, 1, 32, 2560], 3, ttnn.TILE_LAYOUT),  # SD3.5 prompt FF2 output
    ],
    ids=["sd-vision", "sd-prompt", "small"],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
    ],
)
@pytest.mark.parametrize(
    "num_iters, do_check, reuse_inputs",
    [(1, True, False), (6, False, False), (20, False, True)],
    ids=["check", "perf", "stress"],
)
@pytest.mark.parametrize(
    "enable_trace",
    [True, False],
    ids=["use_trace", "no_trace"],
)
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 100000, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True
)
def test_sliced_reduce_scatter(
    t3k_mesh_device,
    num_devices,
    logical_shape,
    scatter_dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    use_program_cache,
    function_level_defaults,
    do_check,
    reuse_inputs,
    enable_trace,
    is_ci_env,
):
    run_sliced_reduce_scatter_impl(
        t3k_mesh_device,
        num_devices,
        logical_shape,
        scatter_dim,
        num_links,
        input_dtype,
        layout,
        topology=ttnn.Topology.Ring,
        num_iters=num_iters,
        mem_config=mem_config,
        do_check=do_check,
        trace_mode=enable_trace,
        reuse_inputs=reuse_inputs,
    )
