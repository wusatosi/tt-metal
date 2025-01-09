#!/bin/bash

pushd tt_metal
mkdir -p api/tt-metalium
mkdir -p api/grayskull/tt-metalium
mkdir -p api/wormhole/tt-metalium
mkdir -p api/blackhole/tt-metalium

git mv host_api.hpp api/tt-metalium/ 2>/dev/null
git mv impl/kernels/runtime_args_data.hpp api/tt-metalium/ 2>/dev/null
git mv impl/program/program.hpp api/tt-metalium/program_impl.hpp 2>/dev/null
git mv impl/kernels/kernel_types.hpp api/tt-metalium/ 2>/dev/null
git mv impl/buffers/circular_buffer_types.hpp api/tt-metalium/ 2>/dev/null
git mv impl/buffers/semaphore.hpp api/tt-metalium/ 2>/dev/null
git mv impl/dispatch/program_command_sequence.hpp api/tt-metalium/ 2>/dev/null
git mv impl/program/program_device_map.hpp api/tt-metalium/ 2>/dev/null
git mv common/base_types.hpp api/tt-metalium/ 2>/dev/null
git mv common/constants.hpp api/tt-metalium/ 2>/dev/null
git mv impl/kernels/data_types.hpp api/tt-metalium/ 2>/dev/null
git mv llrt/tt_cluster.hpp api/tt-metalium/ 2>/dev/null
git mv common/base.hpp api/tt-metalium/ 2>/dev/null
git mv common/metal_soc_descriptor.h api/tt-metalium/ 2>/dev/null
git mv common/test_common.hpp api/tt-metalium/ 2>/dev/null
git mv common/tt_backend_api_types.hpp api/tt-metalium/ 2>/dev/null
git mv common/core_coord.hpp api/tt-metalium/ 2>/dev/null
git mv tt_stl/reflection.hpp api/tt-metalium/ 2>/dev/null
git mv tt_stl/concepts.hpp api/tt-metalium/ 2>/dev/null
git mv tt_stl/type_name.hpp api/tt-metalium/ 2>/dev/null
git mv common/logger.hpp api/tt-metalium/ 2>/dev/null
git mv tt_stl/span.hpp api/tt-metalium/ 2>/dev/null
git mv common/base.hpp api/tt-metalium/ 2>/dev/null
git mv hw/inc/dev_msgs.h api/tt-metalium/ 2>/dev/null
# FIXME: hostdevcommon
git mv llrt/hal.hpp api/tt-metalium/ 2>/dev/null
git mv common/assert.hpp api/tt-metalium/ 2>/dev/null
git mv common/utils.hpp api/tt-metalium/ 2>/dev/null
git mv detail/util.hpp api/tt-metalium/ 2>/dev/null
git mv impl/device/device.hpp api/tt-metalium/device_impl.hpp 2>/dev/null
git mv impl/dispatch/work_executor.hpp api/tt-metalium/ 2>/dev/null
git mv common/env_lib.hpp api/tt-metalium/ 2>/dev/null
git mv impl/dispatch/lock_free_queue.hpp api/tt-metalium/ 2>/dev/null
git mv impl/allocator/basic_allocator.hpp api/tt-metalium/ 2>/dev/null
git mv impl/allocator/allocator.hpp api/tt-metalium/ 2>/dev/null
git mv impl/allocator/allocator_types.hpp api/tt-metalium/ 2>/dev/null
git mv impl/allocator/algorithms/allocator_algorithm.hpp api/tt-metalium/ 2>/dev/null
git mv impl/allocator/l1_banking_allocator.hpp api/tt-metalium/ 2>/dev/null
git mv jit_build/build.hpp api/tt-metalium/ 2>/dev/null
git mv common/executor.hpp api/tt-metalium/ 2>/dev/null
# FIXME: taskflow
git mv jit_build/data_format.hpp api/tt-metalium/ 2>/dev/null
git mv hw/inc/circular_buffer_constants.h api/tt-metalium/ 2>/dev/null
git mv jit_build/settings.hpp api/tt-metalium/ 2>/dev/null
git mv jit_build/hlk_desc.hpp api/tt-metalium/ 2>/dev/null
# FIXME: Tracy
git mv tt_stl/aligned_allocator.hpp api/tt-metalium/ 2>/dev/null
git mv llrt/rtoptions.hpp api/tt-metalium/ 2>/dev/null
git mv impl/dispatch/dispatch_core_manager.hpp api/tt-metalium/ 2>/dev/null
git mv common/core_descriptor.hpp api/tt-metalium/ 2>/dev/null
git mv impl/dispatch/dispatch_core_common.hpp api/tt-metalium/ 2>/dev/null
git mv llrt/get_platform_architecture.hpp api/tt-metalium/ 2>/dev/null
git mv impl/dispatch/command_queue_interface.hpp api/tt-metalium/ 2>/dev/null
git mv common/math.hpp api/tt-metalium/ 2>/dev/null
git mv impl/dispatch/cq_commands.hpp api/tt-metalium/ 2>/dev/null
git mv impl/dispatch/memcpy.hpp api/tt-metalium/ 2>/dev/null
git mv llrt/llrt.hpp api/tt-metalium/ 2>/dev/null
git mv llrt/tt_memory.h api/tt-metalium/ 2>/dev/null
git mv impl/sub_device/sub_device_manager.hpp api/tt-metalium/ 2>/dev/null
git mv impl/sub_device/sub_device.hpp api/tt-metalium/ 2>/dev/null
git mv impl/sub_device/sub_device_types.hpp api/tt-metalium/ 2>/dev/null
git mv impl/device/program_cache.hpp api/tt-metalium/ 2>/dev/null
git mv tt_stl/unique_any.hpp api/tt-metalium/ 2>/dev/null
# END OF host_api.hpp
# The following files are extra fetches
git mv common/bfloat16.hpp api/tt-metalium/ 2>/dev/null
git mv impl/buffers/global_circular_buffer.hpp api/tt-metalium/global_circular_buffer_impl.hpp 2>/dev/null
git mv include/tt_metal/global_circular_buffer.hpp api/tt-metalium/ 2>/dev/null
git mv impl/buffers/buffer_constants.hpp api/tt-metalium/ 2>/dev/null
git mv common/bfloat4.hpp api/tt-metalium/ 2>/dev/null
git mv common/blockfloat_common.hpp api/tt-metalium/ 2>/dev/null
git mv graph/graph_tracking.hpp api/tt-metalium/ 2>/dev/null
git mv impl/tile/tile.hpp api/tt-metalium/ 2>/dev/null
git mv impl/buffers/buffer.hpp api/tt-metalium/ 2>/dev/null
git mv common/bfloat8.hpp api/tt-metalium/ 2>/dev/null
git mv detail/tt_metal.hpp api/tt-metalium/ 2>/dev/null
git mv common/test_tiles.hpp api/tt-metalium/ 2>/dev/null
git mv impl/dispatch/device_command.hpp api/tt-metalium/ 2>/dev/null
git mv include/tt_metal/types.hpp api/tt-metalium/ 2>/dev/null
git mv impl/device/device_handle.hpp api/tt-metalium/ 2>/dev/null
git mv tt_stl/slotmap.hpp api/tt-metalium/ 2>/dev/null
git mv impl/buffers/global_semaphore.hpp api/tt-metalium/ 2>/dev/null
git mv distributed/mesh_device.hpp api/tt-metalium/ 2>/dev/null
git mv distributed/mesh_device_view.hpp api/tt-metalium/ 2>/dev/null
git mv impl/dispatch/command_queue.hpp api/tt-metalium/ 2>/dev/null
git mv impl/dispatch/worker_config_buffer.hpp api/tt-metalium/ 2>/dev/null
git mv impl/trace/trace_buffer.hpp api/tt-metalium/ 2>/dev/null
git mv impl/trace/trace.hpp api/tt-metalium/ 2>/dev/null
git mv impl/device/device_pool.hpp api/tt-metalium/ 2>/dev/null
git mv tools/profiler/op_profiler.hpp api/tt-metalium/ 2>/dev/null
git mv tools/profiler/profiler.hpp api/tt-metalium/ 2>/dev/null
git mv tools/profiler/profiler_state.hpp api/tt-metalium/ 2>/dev/null
git mv tools/profiler/common.hpp api/tt-metalium/ 2>/dev/null
git mv impl/kernels/kernel.hpp api/tt-metalium/ 2>/dev/null
git mv impl/buffers/circular_buffer.hpp api/tt-metalium/ 2>/dev/null
git mv common/work_split.hpp api/tt-metalium/ 2>/dev/null
git mv detail/reports/compilation_reporter.hpp api/tt-metalium/ 2>/dev/null
git mv common/tilize_untilize.hpp api/tt-metalium/ 2>/dev/null
git mv hw/inc/dataflow_api.h api/tt-metalium/ 2>/dev/null
git mv hw/inc/tt_log.h api/tt-metalium/ 2>/dev/null

# Here be copies.  BAD BAD BAD.  Because reasons.  And ARCH_NAME.  And stuff.
cp hw/inc/grayskull/noc/noc_parameters.h api/grayskull/tt-metalium/ 2>/dev/null
cp hw/inc/wormhole/noc/noc_parameters.h api/wormhole/tt-metalium/ 2>/dev/null
cp hw/inc/blackhole/noc/noc_parameters.h api/blackhole/tt-metalium/ 2>/dev/null

git mv hw/inc/grayskull/eth_l1_address_map.h api/grayskull/tt-metalium/ 2>/dev/null
git mv hw/inc/wormhole/eth_l1_address_map.h api/wormhole/tt-metalium/ 2>/dev/null
git mv hw/inc/blackhole/eth_l1_address_map.h api/blackhole/tt-metalium/ 2>/dev/null
popd

pushd tt_metal
pushd programming_examples
find . \( -name '*.hpp' -o -name '*.h' -o -name '*.cpp' \) -print | xargs sed -Ef ../../reorg-api.consumer.sed -i
popd
find . -path ./third_party -prune -o \( -name '*.hpp' -o -name '*.h' -o -name '*.cpp' \) -print | xargs sed -Ef ../reorg-api.tt-metal.sed -i
pushd api
find . \( -name '*.hpp' -o -name '*.h' -o -name '*.cpp' \) -print | xargs sed -Ef ../../reorg-api.tt-metal-api.sed -i
popd
pushd hostdevcommon
find . \( -name '*.hpp' -o -name '*.h' -o -name '*.cpp' \) -print | xargs sed -Ef ../../reorg-api.tt-metal-hostdevcommon.sed -i
popd
pushd programming_examples
find . \( -name '*.hpp' -o -name '*.h' -o -name '*.cpp' \) -print | xargs sed -Ef ../../reorg-api.tt-metal-programming-examples.sed -i
popd
popd

pushd ttnn
find . \( -name '*.hpp' -o -name '*.h' -o -name '*.cpp' \) -print | xargs sed -Ef ../reorg-api.consumer.sed -i
popd

pushd tt-train
find . \( -name '*.hpp' -o -name '*.h' -o -name '*.cpp' \) -print | xargs sed -Ef ../reorg-api.consumer.sed -i
popd

pushd tests
pushd tt_metal
find . \( -name '*.hpp' -o -name '*.h' -o -name '*.cpp' \) -print | xargs sed -Ef ../../reorg-api.tt-metal-tests.sed -i
popd
pushd ttnn
find . \( -name '*.hpp' -o -name '*.h' -o -name '*.cpp' \) -print | xargs sed -Ef ../../reorg-api.consumer.sed -i
popd
pushd tt_eager
find . \( -name '*.hpp' -o -name '*.h' -o -name '*.cpp' \) -print | xargs sed -Ef ../../reorg-api.consumer.sed -i
popd
popd
