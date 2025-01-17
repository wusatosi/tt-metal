add_library(tt_metal)
add_library(Metalium::Metal ALIAS tt_metal)

set_target_properties(
    tt_metal
    PROPERTIES
        EXPORT_NAME
            Metalium
)

# FIXME: TT-Metalium should support pre-C++20, but we have headers (eg: <reflect>) in our public API that is pushing C++20 on consumers
target_compile_features(tt_metal PUBLIC cxx_std_20)

target_sources(
    tt_metal
    PRIVATE
        tt_metal.cpp
        graph/graph_tracking.cpp
        experimental/hal.cpp
)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
    target_sources(
        tt_metal
        PUBLIC
            FILE_SET api
            TYPE HEADERS
            BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/api
            FILES
                api/tt-metalium/aligned_allocator.hpp
                api/tt-metalium/allocator_algorithm.hpp
                api/tt-metalium/allocator_types.hpp
                api/tt-metalium/allocator.hpp
                api/tt-metalium/assert.hpp
                api/tt-metalium/base_types.hpp
                api/tt-metalium/basic_allocator.hpp
                api/tt-metalium/bfloat16.hpp
                api/tt-metalium/bfloat4.hpp
                api/tt-metalium/bfloat8.hpp
                api/tt-metalium/blockfloat_common.hpp
                api/tt-metalium/buffer.hpp
                api/tt-metalium/buffer_constants.hpp
                api/tt-metalium/build.hpp
                api/tt-metalium/circular_buffer.hpp
                api/tt-metalium/circular_buffer_constants.h
                api/tt-metalium/circular_buffer_types.hpp
                api/tt-metalium/command_queue.hpp
                api/tt-metalium/command_queue_interface.hpp
                api/tt-metalium/common.hpp
                api/tt-metalium/compilation_reporter.hpp
                api/tt-metalium/concepts.hpp
                api/tt-metalium/constants.hpp
                api/tt-metalium/core_coord.hpp
                api/tt-metalium/core_descriptor.hpp
                api/tt-metalium/cq_commands.hpp
                api/tt-metalium/data_format.hpp
                api/tt-metalium/data_types.hpp
                api/tt-metalium/dev_msgs.h
                api/tt-metalium/device.hpp
                api/tt-metalium/device_command.hpp
                api/tt-metalium/device_impl.hpp
                api/tt-metalium/device_pool.hpp
                api/tt-metalium/dispatch_constants.hpp
                api/tt-metalium/dispatch_core_common.hpp
                api/tt-metalium/dispatch_core_manager.hpp
                api/tt-metalium/dispatch_settings.hpp
                api/tt-metalium/dprint_server.hpp
                api/tt-metalium/env_lib.hpp
                api/tt-metalium/event.hpp
                api/tt-metalium/get_platform_architecture.hpp
                api/tt-metalium/global_circular_buffer.hpp
                api/tt-metalium/global_circular_buffer_impl.hpp
                api/tt-metalium/global_semaphore.hpp
                api/tt-metalium/graph_tracking.hpp
                api/tt-metalium/hal.hpp
                api/tt-metalium/hal_exp.hpp
                api/tt-metalium/helpers.hpp
                api/tt-metalium/hlk_desc.hpp
                api/tt-metalium/host_api.hpp
                api/tt-metalium/kernel.hpp
                api/tt-metalium/kernel_types.hpp
                api/tt-metalium/l1_banking_allocator.hpp
                api/tt-metalium/launch_message_ring_buffer_state.hpp
                api/tt-metalium/lock_free_queue.hpp
                api/tt-metalium/logger.hpp
                api/tt-metalium/math.hpp
                api/tt-metalium/memcpy.hpp
                api/tt-metalium/memory_reporter.hpp
                api/tt-metalium/mesh_config.hpp
                api/tt-metalium/mesh_device.hpp
                api/tt-metalium/mesh_device_view.hpp
                api/tt-metalium/metal_soc_descriptor.h
                api/tt-metalium/overloaded.hpp
                api/tt-metalium/persistent_kernel_cache.hpp
                api/tt-metalium/profiler.hpp
                api/tt-metalium/profiler_state.hpp
                api/tt-metalium/program_cache.hpp
                api/tt-metalium/program_command_sequence.hpp
                api/tt-metalium/program_device_map.hpp
                api/tt-metalium/program_impl.hpp
                api/tt-metalium/reflection.hpp
                api/tt-metalium/rtoptions.hpp
                api/tt-metalium/runtime_args_data.hpp
                api/tt-metalium/semaphore.hpp
                api/tt-metalium/settings.hpp
                api/tt-metalium/slotmap.hpp
                api/tt-metalium/span.hpp
                api/tt-metalium/sub_device.hpp
                api/tt-metalium/sub_device_manager.hpp
                api/tt-metalium/sub_device_manager_tracker.hpp
                api/tt-metalium/sub_device_types.hpp
                api/tt-metalium/system_mesh.hpp
                api/tt-metalium/test_common.hpp
                api/tt-metalium/test_tiles.hpp
                api/tt-metalium/tile.hpp
                api/tt-metalium/tilize_untilize.hpp
                api/tt-metalium/trace.hpp
                api/tt-metalium/trace_buffer.hpp
                api/tt-metalium/tt_align.hpp
                api/tt-metalium/tt_backend_api_types.hpp
                api/tt-metalium/tt_cluster.hpp
                api/tt-metalium/tt_log.h
                api/tt-metalium/tt_memory.h
                api/tt-metalium/tt_metal.hpp
                api/tt-metalium/type_name.hpp
                api/tt-metalium/types.hpp
                api/tt-metalium/unique_any.hpp
                api/tt-metalium/util.hpp
                api/tt-metalium/utils.hpp
                api/tt-metalium/work_executor.hpp
                api/tt-metalium/work_split.hpp
                api/tt-metalium/worker_config_buffer.hpp
    )
endif()

target_link_libraries(
    tt_metal
    PUBLIC
        umd::device
        magic_enum
        fmt::fmt-header-only
        span
        TracyClient
        nlohmann_json::nlohmann_json
        TT::Metalium::HostDevCommon
        Reflect::Reflect
    PRIVATE
        Metalium::Metal::Impl
        Metalium::Metal::STL
        metal_common_libs
        profiler
        common
        jit_build
        llrt
        detail
        distributed
        HAL::grayskull
        HAL::wormhole
        HAL::blackhole
)

target_precompile_headers(
    tt_metal
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tracy/public/tracy/Tracy.hpp
        <functional>
        <map>
        <memory>
        <unordered_map>
        <variant>
        <vector>
)

target_include_directories(
    tt_metal
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>"
    PRIVATE
        api/tt-metalium # FIXME: Re-home the tests and remove this
)
target_compile_options(tt_metal PUBLIC -Wno-int-to-pointer-cast)
add_dependencies(tt_metal hw_toolchain)

set_target_properties(
    tt_metal
    PROPERTIES
        INSTALL_RPATH
            "${PROJECT_BINARY_DIR}/lib"
        ADDITIONAL_CLEAN_FILES
            "${PROJECT_BINARY_DIR}/lib;${PROJECT_BINARY_DIR}/obj"
)

add_subdirectory(examples)
if(BUILD_PROGRAMMING_EXAMPLES)
    add_subdirectory(programming_examples)
endif()

# Allow internal files to access the public API "by default" and without the
# scoping that external consumers must use.  Scaoping may still be used if desired.
include_directories(
    api
    api/tt-metalium
)

add_subdirectory(hw)
add_subdirectory(hostdevcommon)
add_subdirectory(common)
add_subdirectory(jit_build)
add_subdirectory(llrt)
add_subdirectory(tools)
add_subdirectory(impl)
add_subdirectory(detail)
add_subdirectory(distributed)
add_subdirectory(tt_stl)

install(
    TARGETS
        tt_metal
        ttmetalium_hostdevcommon
    EXPORT Metalium
    LIBRARY
        COMPONENT metalium
    FILE_SET
    api
        COMPONENT metalium-dev
)
install(EXPORT Metalium NAMESPACE TT:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tt-metalium COMPONENT metalium-dev)

# 3rd party dependencies we export in our public API.
# We must ship not only these libraries, but their public headers as well
# ... or refactor our public API.
install(
    TARGETS
        reflect
        # magic_enum
        TracyClient
        span
        boost_core
        boost_assert
        boost_config
        boost_static_assert
        boost_throw_exception
    # fmt-header-only
    # nlohmann_json
    EXPORT Metalium
    FILE_SET
    api
        COMPONENT metalium-dev
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/metalium-thirdparty
)
install(
    TARGETS
        TracyClient
        DESTINATION
        ${CMAKE_INSTALL_INCLUDEDIR}/metalium-thirdparty/tracy
)
# install(TARGETS fmt
# PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/metalium-thirdparty/fmt
# COMPONENT metalium-dev
# )
