# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.utility_functions import skip_for_grayskull
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from ttnn import ConcatMesh2dToTensor, ShardTensor2dMesh

NUM_BUFFERS = 8


def report_mismatches(golden, actual, max_printable=None):
    printed = 0
    for w in range(golden.shape[0]):
        for z in range(golden.shape[1]):
            for y in range(0, golden.shape[2], 32):
                for x in range(0, golden.shape[3], 32):
                    print_it = (max_printable is None or printed < max_printable) and golden[w, z, y, x] != actual[
                        w, z, y, x
                    ]
                    if print_it:
                        printed += 1
                        print(
                            f"output mismatch for tensor at [{w}, {z}, {y}, {x}]: expected {golden[w, z, y, x]} != actual {actual[w, z, y, x]}"
                        )


def print_tile_corners_of_tensor(t):
    for w in range(t.shape[0]):
        for z in range(t.shape[1]):
            str = ""
            for x in range(0, t.shape[3], 32):
                str += f"{x:<5} "[:5]
            print(f"     {str}")
            for y in range(0, t.shape[2], 32):
                str_vals = f"y={y:<3} "[:5]
                for x in range(0, t.shape[3], 32):
                    yy = 0
                    xx = 0
                    val = int(t[w, z, y + yy, x + xx].item())
                    str_vals += f"{val:<5} "[:5]
                print(f"{str_vals}")


def run_with_trace(
    mesh_device,
    all_gather_topology,
    input_tensor,
    dim,
    persistent_output_tensor,
    num_links,
    cluster_axis,
    output_mem_config,
    ccl_semaphore_handles,
    worker_sub_device_id,
    n_worker=None,
    n_buffer=None,
    num_iter=1,
    warmup_iters=0,
    use_all_gather_async=False,
    profiler=BenchmarkProfiler(),
):
    # Compile Run
    logger.info("Compiling model")
    if use_all_gather_async:
        tt_out_tensor = ttnn.experimental.all_gather_async(
            input_tensor,
            dim,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            topology=all_gather_topology,
            multi_device_global_semaphore=ccl_semaphore_handles[0]
            if type(ccl_semaphore_handles) == list
            else ccl_semaphore_handles,
            persistent_output_tensor=persistent_output_tensor,
            num_links=num_links,
            memory_config=output_mem_config,
            subdevice_id=worker_sub_device_id,
        )
    else:
        tt_out_tensor = ttnn.all_gather(
            input_tensor,
            dim=dim,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            num_links=num_links,
            memory_config=output_mem_config,
            topology=all_gather_topology,
        )
    ttnn.synchronize_device(mesh_device)

    # Capture trace
    logger.info("Capturing trace")

    def capture_trace(n_iters):
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(n_iters):
            if use_all_gather_async:
                tt_out_tensor = ttnn.experimental.all_gather_async(
                    input_tensor,
                    dim,
                    cluster_axis=cluster_axis,
                    mesh_device=mesh_device,
                    topology=all_gather_topology,
                    multi_device_global_semaphore=ccl_semaphore_handles[i % NUM_BUFFERS]
                    if type(ccl_semaphore_handles) == list
                    else ccl_semaphore_handles,
                    persistent_output_tensor=persistent_output_tensor,
                    num_links=num_links,
                    memory_config=output_mem_config,
                    subdevice_id=worker_sub_device_id,
                )
            else:
                tt_out_tensor = ttnn.all_gather(
                    input_tensor,
                    dim=dim,
                    cluster_axis=cluster_axis,
                    mesh_device=mesh_device,
                    num_links=num_links,
                    memory_config=output_mem_config,
                    topology=all_gather_topology,
                )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        return trace_id

    if warmup_iters > 0:
        trace_id_warmup = capture_trace(warmup_iters)
    trace_id = capture_trace(num_iter)

    # Run the op
    logger.info("Starting Trace perf test...")
    profiler.start("all-gather-async-trace-warmup")
    if warmup_iters > 0:
        ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
        ttnn.release_trace(mesh_device, trace_id_warmup)
        ttnn.synchronize_device(mesh_device)
    profiler.end("all-gather-async-trace-warmup")

    profiler.start("all-gather-async-trace")
    signpost("start")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
    profiler.end("all-gather-async-trace")
    time_taken = profiler.get_duration("all-gather-async-trace") - profiler.get_duration(
        "all-gather-async-trace-warmup"
    )
    effective_iter = num_iter - warmup_iters
    logger.info(f"Time taken e2e: {time_taken} s")
    logger.info(f"Time per iter e2e: {time_taken / effective_iter} s")
    logger.info(f"Time per iter e2e: {time_taken / effective_iter * 1e6} us")

    return tt_out_tensor


def run_falcon_attention_ccl(
    mesh_device,
    num_devices_per_line,
    per_chip_output_shape,
    tensor_memory_layout,
    dim,
    num_links,
    input_dtype,
    layout,
    buffer_type: ttnn.BufferType,
    use_program_cache,
    function_level_defaults,
    input_shard_spec: ttnn.ShardSpec = None,
    output_shard_spec: ttnn.ShardSpec = None,
    num_all_gather_instances: int = 1,
    num_iters: int = 1,
    warmup_iters: int = 0,
    cluster_axis: int = 0,
    tile=(32, 32),
    trace_mode=False,
    debug=False,
    profiler=BenchmarkProfiler(),
    # New all-gather-async and persistent fabric params
    use_all_gather_async=False,
    use_persistent_output=False,
):
    if use_persistent_output and not use_all_gather_async:
        pytest.skip("Persistent output tensor requires all-gather-async")

    input_shape_per_chip = list(per_chip_output_shape)
    input_shape_per_chip[dim] //= num_devices_per_line
    tensor_height_per_all_gather = per_chip_output_shape[-2]

    full_mesh_input_shape = list(per_chip_output_shape)
    ## The `all_gather_instances_concat_dim` is the dimension we will split the cluster spanning tensor along in order to split it
    ## off into per-all-gather tensors
    all_gather_instances_concat_dim = 1 if dim == 0 else 0
    full_mesh_input_shape[all_gather_instances_concat_dim] *= num_all_gather_instances
    logger.info(
        f"per_chip_output_shape: {full_mesh_input_shape}, dim: {dim}, all_gather_instances_concat_dim: {all_gather_instances_concat_dim}, num_devices_per_line: {num_devices_per_line}"
    )

    all_gather_instances_goldens = []
    full_input_tensor_unfractured = torch.rand(full_mesh_input_shape, dtype=torch.bfloat16)

    input_mem_config = ttnn.MemoryConfig(tensor_memory_layout, buffer_type=buffer_type, shard_spec=input_shard_spec)
    shard_dims = (dim, all_gather_instances_concat_dim) if cluster_axis == 0 else (all_gather_instances_concat_dim, dim)
    concat_dims = shard_dims

    mesh_shape = (
        (num_devices_per_line, num_all_gather_instances)
        if cluster_axis == 0
        else (num_all_gather_instances, num_devices_per_line)
    )

    if input_shard_spec is not None and output_shard_spec is None:
        output_shard_shape = list(input_shard_spec.shape)
        if dim == len(per_chip_output_shape) - 1:
            output_shard_shape[1] *= num_devices_per_line
        else:
            output_shard_shape[0] *= num_devices_per_line
        output_shard_spec = ttnn.ShardSpec(
            input_shard_spec.grid,
            output_shard_shape,
            input_shard_spec.orientation,
        )
    output_mem_config = ttnn.MemoryConfig(tensor_memory_layout, buffer_type=buffer_type, shard_spec=output_shard_spec)
    ttnn_tensor = ttnn.from_torch(
        full_input_tensor_unfractured,
        tile=ttnn.Tile(tile),
        dtype=input_dtype,
        device=mesh_device,
        layout=layout,
        memory_config=input_mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=shard_dims),
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)
    ttnn_tensor = ttnn.to_memory_config(ttnn_tensor, input_mem_config)
    # TODO: Take as an arg
    linear = True
    if linear:
        all_gather_topology = ttnn.Topology.Linear
        wrap_mesh = False
    else:
        all_gather_topology = ttnn.Topology.Ring
        wrap_mesh = False

    ttnn_persistent_output_tensor = None
    if use_persistent_output:
        ttnn_persistent_output_tensor = ttnn.from_torch(
            torch.zeros(per_chip_output_shape),
            tile=ttnn.Tile(tile),
            dtype=input_dtype,
            device=mesh_device,
            layout=layout,
            memory_config=output_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    sub_device_stall_group = []
    if use_all_gather_async:
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
            ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(NUM_BUFFERS)
        ]
    try:
        # ttnn.visualize_mesh_device(mesh_device, tensor=ttnn_tensor)
        if trace_mode:
            ttnn_tensor_out = run_with_trace(
                input_tensor=ttnn_tensor,
                dim=dim,
                cluster_axis=cluster_axis,
                mesh_device=mesh_device,
                persistent_output_tensor=ttnn_persistent_output_tensor,
                num_links=num_links,
                output_mem_config=output_mem_config,
                ccl_semaphore_handles=ccl_semaphore_handles,
                worker_sub_device_id=worker_sub_device_id,
                all_gather_topology=all_gather_topology,
                num_iter=num_iters,
                warmup_iters=warmup_iters,
                use_all_gather_async=use_all_gather_async,
                profiler=profiler,
            )

        else:
            signpost("start")
            for i in range(num_iters):
                if use_all_gather_async:
                    logger.info("Running all-gather async")
                    ttnn_tensor_out = ttnn.experimental.all_gather_async(
                        ttnn_tensor,
                        dim,
                        cluster_axis=cluster_axis,
                        mesh_device=mesh_device,
                        topology=all_gather_topology,
                        multi_device_global_semaphore=ccl_semaphore_handles[i % NUM_BUFFERS],
                        persistent_output_tensor=ttnn_persistent_output_tensor,
                        num_links=num_links,
                        memory_config=output_mem_config,
                        subdevice_id=worker_sub_device_id,
                    )
                else:
                    ttnn_tensor_out = ttnn.all_gather(
                        ttnn_tensor,
                        dim=dim,
                        cluster_axis=cluster_axis,
                        mesh_device=mesh_device,
                        num_links=num_links,
                        memory_config=output_mem_config,
                        topology=all_gather_topology,
                    )
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            signpost("stop")
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise e
    finally:
        mesh_device.reset_sub_device_stall_group()

    # ttnn.visualize_mesh_device(mesh_device, tensor=ttnn_tensor_out)
    tt_output_tensor = ttnn.to_torch(
        ttnn_tensor_out, mesh_composer=ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=concat_dims)
    )
    output_tensors_list = torch.chunk(tt_output_tensor, num_all_gather_instances, dim=all_gather_instances_concat_dim)
    output_golden = torch.zeros(tt_output_tensor.shape)

    # Check the tensor addresses
    if use_persistent_output:
        persistent_output_tensors = ttnn.get_device_tensors(ttnn_persistent_output_tensor)
        output_tensors = ttnn.get_device_tensors(ttnn_tensor_out)

        for persistent_tensor, output_tensor in zip(persistent_output_tensors, output_tensors):
            assert (
                persistent_tensor.buffer_address() == output_tensor.buffer_address()
            ), "Persistent tensor address mismatch"

    # Repeat the input tensor to represent the fact that the full concatenated input tensor lives across every
    # device in the line
    repeat_factor = [1] * len(output_golden.shape)
    repeat_factor[dim] = num_devices_per_line
    output_golden[:, :, :, :] = full_input_tensor_unfractured.repeat(repeat_factor)

    eq = True
    if input_dtype == ttnn.bfloat16:
        eq, output = comp_equal(tt_output_tensor, output_golden)
        if not eq and debug is True:
            logger.error(f"found mismatches")
            report_mismatches(tt_output_tensor, output_golden, 100)
            print_tile_corners_of_tensor(tt_output_tensor)
    else:
        eq, output = comp_pcc(tt_output_tensor, output_golden)
    if not eq:
        logger.error(f"output mismatch for tensor: {output}")

    assert eq, f"FAILED: {output}"


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, dim, layout",
    [
        (8, 1, [1, 1, 64, 8192], 3, ttnn.TILE_LAYOUT),
        # (8, 1, [1, 1, 128, 8192], 3, ttnn.TILE_LAYOUT),
        # (8, 1, [1, 1, 2048, 8192], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        # ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.DRAM,
        # ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize("replication_factor", [1])
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 8), id="4x8_grid")], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_falcon_attention_ccl(
    num_devices,
    per_chip_output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    buffer_type,
    use_program_cache,
    function_level_defaults,
    mesh_device,
    replication_factor,
    num_iters=1,
):
    if mesh_device.get_num_devices() < 8:
        pytest.skip("Not T3K!")
    run_falcon_attention_ccl(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        ttnn.TensorMemoryLayout.INTERLEAVED,
        dim,
        num_links,
        input_dtype,
        layout,
        buffer_type,
        use_program_cache,
        function_level_defaults,
        num_iters=num_iters,
        num_all_gather_instances=replication_factor,
        cluster_axis=1,
        use_all_gather_async=True,
    )


# test all gather t3k


def run_with_trace_t3k(
    mesh_device,
    all_gather_topology,
    input_tensor_mesh,
    dim,
    num_links,
    output_mem_config,
    multi_device_global_semaphore,
    num_iter=20,
    subdevice_id=None,
):
    # Compile Run
    logger.info("Compiling model")
    tt_out_tensor = ttnn.experimental.all_gather_async(
        input_tensor_mesh,
        dim,
        multi_device_global_semaphore=multi_device_global_semaphore,
        num_links=num_links,
        memory_config=output_mem_config,
        topology=all_gather_topology,
        subdevice_id=subdevice_id,
    )
    ttnn.synchronize_device(mesh_device)

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iter):
        tt_out_tensor = ttnn.experimental.all_gather_async(
            input_tensor_mesh,
            dim,
            multi_device_global_semaphore=multi_device_global_semaphore,
            num_links=num_links,
            memory_config=output_mem_config,
            topology=all_gather_topology,
            subdevice_id=subdevice_id,
        )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Run the op
    logger.info("Starting Trace perf test...")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)

    return tt_out_tensor


def run_all_gather_impl(
    mesh_device,
    num_devices,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    use_program_cache,
    function_level_defaults,
    all_gather_topology,
    num_iters=10,
    trace_mode=False,
    rand_tensor=True,
    mem_config=None,
    input_shard_shape=None,
    input_shard_grid=None,
    output_shard_shape=None,
    output_shard_grid=None,
    tensor_mem_layout=None,
    use_cluster_axis_api=False,
    cluster_axis=None,
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
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)]

    logger.info(f"Output shape: {output_shape}")
    logger.info(f"dim: {dim}")
    logger.info(f"input_shard_shape: {input_shard_shape}")
    logger.info(f"input_shard_grid: {input_shard_grid}")

    ### For sharded all gather only
    if bool(input_shard_shape) != bool(input_shard_grid) and bool(tensor_mem_layout) != bool(input_shard_grid):
        pytest.fail(
            "Both input_shard_shape, shard_grid, and tensor_mem_layout must be provided together or all must be None"
        )
    if input_shard_shape and input_shard_grid:
        input_shard_spec = ttnn.ShardSpec(
            input_shard_grid,
            input_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        input_mem_config = ttnn.MemoryConfig(
            tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec
        )
        if output_shard_shape is None:
            assert (
                output_shard_grid is None
            ), "output_shard_grid must not be provided if output_shard_shape is not provided"
            output_shard_shape = list(input_shard_shape)
            if dim == len(output_shape) - 1:
                output_shard_shape[1] *= num_devices
            else:
                output_shard_shape[0] *= num_devices
            output_shard_spec = ttnn.ShardSpec(
                input_shard_grid,
                output_shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            output_mem_config = ttnn.MemoryConfig(
                tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
            )
        else:
            assert output_shard_grid is not None, "output_shard_grid must be provided if output_shard_shape is provided"
            output_shard_spec = ttnn.ShardSpec(
                output_shard_grid,
                output_shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            output_mem_config = ttnn.MemoryConfig(
                tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
            )
    else:
        assert mem_config is not None
        input_mem_config = mem_config
        output_mem_config = mem_config
    ###

    input_tensor_mesh_list = []
    output_tensor_goldens_list = []

    for i in range(num_iters):
        if rand_tensor:
            output_tensor = torch.rand(output_shape).bfloat16()
        else:
            output_tensor = torch.zeros(output_shape)
            tile_id = 1
            for w in range(output_shape[0]):
                for z in range(output_shape[1]):
                    for y in range(0, output_shape[2], 32):
                        for x in range(0, output_shape[3], 32):
                            output_tensor[w, z, y : y + 32, x : x + 32] = tile_id
                            tile_id += 1

        output_tensor_goldens_list.append(output_tensor)
        input_tensors = torch.chunk(output_tensor, num_devices, dim)
        tt_input_tensors = []
        for i, t in enumerate(input_tensors):
            tt_input_tensors.append(ttnn.Tensor(t, input_dtype).to(layout))
            logger.info(f"using device {mesh_device.get_device_ids()[i]}")

        input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors).to(mesh_device, input_mem_config)

        input_tensor_mesh_list.append(input_tensor_mesh)

    tt_out_tensor_list = []
    if trace_mode:
        tt_out_tensor = run_with_trace_t3k(
            mesh_device,
            all_gather_topology,
            input_tensor_mesh_list[0],
            dim,
            num_links,
            output_mem_config,
            multi_device_global_semaphore=ccl_semaphore_handles[0],
            num_iter=num_iters,
            subdevice_id=worker_sub_device_id,
        )
        tt_out_tensor_list.append(tt_out_tensor)
    else:
        for i in range(num_iters):
            if use_cluster_axis_api:
                tt_out_tensor = ttnn.experimental.all_gather_async(
                    input_tensor_mesh_list[i],
                    dim,
                    cluster_axis=cluster_axis,
                    mesh_device=mesh_device,
                    memory_config=output_mem_config,
                    topology=all_gather_topology,
                    multi_device_global_semaphore=ccl_semaphore_handles[i],
                    subdevice_id=worker_sub_device_id,
                    num_preferred_links=num_links,
                )

            else:
                tt_out_tensor = ttnn.experimental.all_gather_async(
                    input_tensor_mesh_list[i],
                    dim,
                    multi_device_global_semaphore=ccl_semaphore_handles[i],
                    num_links=num_links,
                    memory_config=output_mem_config,
                    topology=all_gather_topology,
                    subdevice_id=worker_sub_device_id,
                )
            tt_out_tensor_list.append(tt_out_tensor)

        logger.info(f"Waiting for op")
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done op")

    passed = True
    for tensor_index in range(len(tt_out_tensor_list)):
        tt_out_tensor = tt_out_tensor_list[tensor_index]
        output_tensor = output_tensor_goldens_list[tensor_index]
        for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
            tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
            logger.info(f"Checking for device {t.device().id()}")

            if input_dtype == ttnn.bfloat16:
                eq, output = comp_equal(tt_output_tensor, output_tensor)
            else:
                eq, output = comp_pcc(tt_output_tensor, output_tensor)
            if not eq:
                logger.error(f"output mismatch for tensor {i}")
                passed = False

    assert (
        mesh_device.num_program_cache_entries() == 1 or mesh_device.num_program_cache_entries() == num_iters
    ), f"Device has {mesh_device.num_program_cache_entries()} program cache entries"
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    if not passed:
        assert eq, f"{i} FAILED: {output}"


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, output_shape, dim, layout",
    [
        # (8, 1, [1, 1, 32, 8192], 3, ttnn.TILE_LAYOUT),
        # (8, 1, [1, 1, 1024, 4096], 2, ttnn.TILE_LAYOUT),
        # (8, 1, [1, 8, 1024, 4096], 1, ttnn.TILE_LAYOUT),
        # (8, 1, [1, 1, 256, 4096], 2, ttnn.TILE_LAYOUT),
        (8, 1, [1, 1, 32768, 4096], 2, ttnn.TILE_LAYOUT),
        (8, 1, [1, 8, 32768, 4096], 1, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        # ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        # ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)
    ],
)
@pytest.mark.parametrize("num_iters", [1])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 23887872, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_all_gather_t3k(
    t3k_mesh_device,
    # pcie_mesh_device,
    num_devices,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    use_program_cache,
    function_level_defaults,
):
    run_all_gather_impl(
        t3k_mesh_device,
        num_devices,
        output_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Linear,
        num_iters=num_iters,
        rand_tensor=False,
        mem_config=mem_config,
    )
