# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_subdevices.tt.model_config import (
    PREFETCHER_NOC1_GRID,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull

from tests.ttnn.unit_tests.operations.ccl.fusion_subtests.rms_test import (
    run_rms_trace,
    run_rms_fuse_impl,
)

from tests.ttnn.unit_tests.operations.ccl.fusion_subtests.concat_fuse_test import (
    run_concat_fuse_impl,
)
from tests.ttnn.unit_tests.operations.ccl.fusion_subtests.all_gather_silu_test import run_all_gather_silu_impl

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

from tests.ttnn.unit_tests.operations.ccl.test_all_gather_TG_post_commit import (
    run_line_all_gather_on_TG_with_mesh_tensor_along_rows,
)


NUM_ITERATIONS = 75


def get_core_range_set(output_core_grid):
    if isinstance(output_core_grid, ttnn.CoreGrid):
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(output_core_grid.x - 1, output_core_grid.y - 1)),
            ]
        )
    else:
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in output_core_grid
            ]
        )
    return output_core_range_set


CORE_RANGE_SET_1x1 = ttnn.CoreRangeSet(
    {
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
    }
)


def run_allgather_only_with_trace(
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
    input_shard_shape,
    input_shard_grid,
    all_gather_topology,
    num_iters=1,
    trace_mode=False,
    output_shard_shape=None,
    output_shard_grid=None,
    tensor_mem_layout=None,
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
    ###

    input_tensor_mesh_list = []
    output_tensor_goldens_list = []

    for i in range(num_iters):
        output_tensor = torch.rand(output_shape).bfloat16()
        output_tensor_goldens_list.append(output_tensor)
        input_tensors = torch.chunk(output_tensor, num_devices, dim)
        tt_input_tensors = []
        for i, t in enumerate(input_tensors):
            tt_input_tensors.append(ttnn.Tensor(t, input_dtype).to(layout))
            logger.info(f"using device {mesh_device.get_devices()[i].id()}")

        input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors).to(mesh_device, input_mem_config)

        input_tensor_mesh_list.append(input_tensor_mesh)

    tt_out_tensor_list = []
    if trace_mode:
        tt_out_tensor = run_allgather_only_with_trace(
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

    if not passed:
        assert eq, f"{i} FAILED: {output}"


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, output_shape, dim, layout, input_shard_shape, input_shard_grid, output_shard_shape, output_shard_grid, tensor_mem_layout",
    [
        # All Reduce test
        (
            8,
            [1, 1, 32, 10240],
            3,
            ttnn.TILE_LAYOUT,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 3))}),
            None,
            None,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        # Before Concat Heads
        (
            4,
            [1, 32, 32, 128],
            1,
            ttnn.TILE_LAYOUT,
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 7))}),
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
        # Reduce Scatter
        (
            8,
            [1, 1, 32, 30720],
            3,
            ttnn.TILE_LAYOUT,
            (32, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 7))}),
            None,
            None,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        # RMS NORM ALL GATHER FUSION
        (
            4,
            [1, 1, 32, 128],
            3,
            ttnn.TILE_LAYOUT,
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            None,
            None,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("num_iters", [8])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_all_gather_only(
    t3k_mesh_device,
    num_devices,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    num_iters,
    use_program_cache,
    function_level_defaults,
    input_shard_shape,
    input_shard_grid,
    output_shard_shape,
    output_shard_grid,
    tensor_mem_layout,
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
        input_shard_shape,
        input_shard_grid,
        all_gather_topology=ttnn.Topology.Linear,
        num_iters=num_iters,
        output_shard_shape=output_shard_shape,
        output_shard_grid=output_shard_grid,
        tensor_mem_layout=tensor_mem_layout,
    )


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, elements_per_batch, input_shard_grid, output_shard_grid",
    [
        # RMS NORM ALL GATHER FUSION
        (
            4,
            8192,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(x, y),
                        ttnn.CoreCoord(x, y),
                    )
                    for x, y in PREFETCHER_NOC1_GRID
                ]
            ),
        ),
    ],
)
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("use_new_version", [True])
@pytest.mark.parametrize("num_iters, warmup_iters", [[100, 10]])
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 23887872,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_tg_trace_rms_fuse(
    mesh_device,
    num_devices,
    elements_per_batch,
    num_links,
    num_iters,
    warmup_iters,
    use_program_cache,
    function_level_defaults,
    input_shard_grid,
    output_shard_grid,
    trace_mode,
    use_new_version,
):
    profiler = BenchmarkProfiler()
    run_rms_trace(
        mesh_device,
        num_devices,
        elements_per_batch,
        num_links,
        use_program_cache,
        function_level_defaults,
        input_shard_grid,
        output_shard_grid,
        ttnn.Topology.Linear,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        profiler=profiler,
        use_new_version=use_new_version,
    )


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, elements_per_batch, input_shard_grid, output_shard_grid",
    [
        # RMS NORM ALL GATHER FUSION No Reshard
        (
            4,
            8192,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
            None,
        ),
        (
            4,
            8192,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(x, y),
                        ttnn.CoreCoord(x, y),
                    )
                    for x, y in PREFETCHER_NOC1_GRID
                ]
            ),
        ),
    ],
)
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("num_iters", [20])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_rms_fuse(
    mesh_device,
    num_devices,
    elements_per_batch,
    num_links,
    num_iters,
    use_program_cache,
    function_level_defaults,
    input_shard_grid,
    output_shard_grid,
):
    run_rms_fuse_impl(
        mesh_device,
        num_devices,
        elements_per_batch,
        num_links,
        use_program_cache,
        function_level_defaults,
        input_shard_grid,
        output_shard_grid,
        ttnn.Topology.Linear,
        num_iters=num_iters,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, output_shape, dim, layout, input_shard_shape, input_shard_grid, output_shard_shape, output_shard_grid, tensor_mem_layout",
    [
        # Before Concat Heads
        (
            4,
            [1, 32, 32, 128],
            1,
            ttnn.ROW_MAJOR_LAYOUT,
            (32, 128),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 2)),
                }
            ),
            (32, 64),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(6, 6), ttnn.CoreCoord(6, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 7), ttnn.CoreCoord(6, 7)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 9), ttnn.CoreCoord(6, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(6, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 1), ttnn.CoreCoord(6, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 2), ttnn.CoreCoord(6, 2)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 4), ttnn.CoreCoord(6, 4)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 5), ttnn.CoreCoord(6, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 5), ttnn.CoreCoord(5, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 6), ttnn.CoreCoord(5, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 7), ttnn.CoreCoord(5, 7)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 9), ttnn.CoreCoord(5, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(5, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 1), ttnn.CoreCoord(5, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 2), ttnn.CoreCoord(5, 2)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 4), ttnn.CoreCoord(5, 4)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 4), ttnn.CoreCoord(1, 4)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 5), ttnn.CoreCoord(1, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 9), ttnn.CoreCoord(1, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(2, 4), ttnn.CoreCoord(2, 4)),
                    ttnn.CoreRange(ttnn.CoreCoord(2, 5), ttnn.CoreCoord(2, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(2, 9), ttnn.CoreCoord(2, 9)),
                ]
            ),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize("num_links", [3])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("num_iters, warmup_iters", [[75, 5]])
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 23887872,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_concat_fuse(
    mesh_device,
    num_devices,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    num_iters,
    warmup_iters,
    use_program_cache,
    function_level_defaults,
    input_shard_shape,
    input_shard_grid,
    output_shard_shape,
    output_shard_grid,
    tensor_mem_layout,
    trace_mode,
):
    profiler = BenchmarkProfiler()
    run_concat_fuse_impl(
        mesh_device,
        num_devices,
        output_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        use_program_cache,
        function_level_defaults,
        input_shard_shape,
        input_shard_grid,
        all_gather_topology=ttnn.Topology.Linear,
        warmup_iters=warmup_iters,
        num_iters=num_iters,
        output_shard_shape=output_shard_shape,
        output_shard_grid=output_shard_grid,
        tensor_mem_layout=tensor_mem_layout,
        trace_mode=trace_mode,
        profiler=profiler,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, output_shape, dim, layout, input_shard_shape, input_shard_grid, output_shard_shape, output_shard_grid, tensor_mem_layout",
    [
        # Before Concat Heads
        (
            4,
            (1, 1, 32, 3840),
            3,
            ttnn.TILE_LAYOUT,
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9))}),
            (32, 160),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(6, 6), ttnn.CoreCoord(6, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 7), ttnn.CoreCoord(6, 7)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 9), ttnn.CoreCoord(6, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(6, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 1), ttnn.CoreCoord(6, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 2), ttnn.CoreCoord(6, 2)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 4), ttnn.CoreCoord(6, 4)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 5), ttnn.CoreCoord(6, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 5), ttnn.CoreCoord(5, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 6), ttnn.CoreCoord(5, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 7), ttnn.CoreCoord(5, 7)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 9), ttnn.CoreCoord(5, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(5, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 1), ttnn.CoreCoord(5, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 2), ttnn.CoreCoord(5, 2)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 4), ttnn.CoreCoord(5, 4)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 4), ttnn.CoreCoord(1, 4)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 5), ttnn.CoreCoord(1, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 9), ttnn.CoreCoord(1, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(2, 4), ttnn.CoreCoord(2, 4)),
                    ttnn.CoreRange(ttnn.CoreCoord(2, 5), ttnn.CoreCoord(2, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(2, 9), ttnn.CoreCoord(2, 9)),
                    # ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(4, 2)),
                ]
            ),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize("num_links", [4])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("num_iters, warmup_iters", [[75, 5]])
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 23887872,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_all_gather_silu(
    mesh_device,
    num_devices,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    num_iters,
    use_program_cache,
    function_level_defaults,
    input_shard_shape,
    input_shard_grid,
    output_shard_shape,
    output_shard_grid,
    tensor_mem_layout,
    trace_mode,
    warmup_iters,
):
    profiler = BenchmarkProfiler()
    run_all_gather_silu_impl(
        mesh_device,
        num_devices,
        output_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        use_program_cache,
        function_level_defaults,
        input_shard_shape,
        input_shard_grid,
        all_gather_topology=ttnn.Topology.Ring,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        output_shard_shape=output_shard_shape,
        output_shard_grid=output_shard_grid,
        tensor_mem_layout=tensor_mem_layout,
        trace_mode=trace_mode,
        profiler=profiler,
    )


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices",
    [
        4,
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "num_iters, warmup_iters",
    [
        (NUM_ITERATIONS, 10),
    ],
)
@pytest.mark.parametrize("shard_grid_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize(
    "tensor_mem_layout, output_shape, num_links, dim, input_shard_shape,input_shard_grid,output_shard_shape, output_shard_grid, layout",
    (
        (  # AllGather after Binary Mult+Silu
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (1, 1, 32, 3840),
            3,
            3,
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 4))}),
            (32, 160),
            get_core_range_set(PREFETCHER_NOC1_GRID),
            ttnn.TILE_LAYOUT,
        ),
        (  # AllGather after Binary Mult+Silu
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (1, 1, 32, 3840),
            3,
            3,
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 4))}),
            (32, 160),
            get_core_range_set(PREFETCHER_NOC1_GRID),
            ttnn.TILE_LAYOUT,
        ),
    ),
    ids=[
        "sdpa",
        "binary_mult",
    ],
)
@pytest.mark.parametrize("replication_factor", [8])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 17068032, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_all_gather_tg_llama(
    mesh_device,
    num_devices,
    output_shape,
    input_shard_shape,
    input_shard_grid,
    output_shard_shape,
    output_shard_grid,
    shard_grid_orientation,
    tensor_mem_layout,
    dim,
    num_links,
    input_dtype,
    layout,
    use_program_cache,
    function_level_defaults,
    replication_factor,
    num_iters,
    warmup_iters,
):
    if len(mesh_device.get_devices()) != 32:
        pytest.skip("Not TG!")
    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        shard_grid_orientation,
    )

    if output_shard_grid is not None and output_shard_shape is not None:
        output_shard_spec = ttnn.ShardSpec(
            output_shard_grid,
            output_shard_shape,
            shard_grid_orientation,
        )
    else:
        output_shard_spec = None

    profiler = BenchmarkProfiler()

    run_line_all_gather_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        output_shape,
        tensor_mem_layout,
        dim,
        num_links,
        input_dtype,
        layout,
        ttnn.BufferType.L1,
        use_program_cache,
        function_level_defaults,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        input_shard_spec=input_shard_spec,
        output_shard_spec=output_shard_spec,
        num_all_gather_instances=replication_factor,
        cluster_axis=1,
        profiler=profiler,
        trace_mode=True,
        use_all_gather_async=True,
        use_persistent_output=True,
    )
