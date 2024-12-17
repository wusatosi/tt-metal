add_library(impl OBJECT ${IMPL_SRC})
add_library(Metalium::Metal::Impl ALIAS impl)

target_sources(
    impl
    PRIVATE
        sub_device/sub_device.cpp
        sub_device/sub_device_manager.cpp
        device/device.cpp
        device/device_handle.cpp
        device/device_pool.cpp
        buffers/buffer.cpp
        buffers/circular_buffer.cpp
        buffers/circular_buffer_types.cpp
        buffers/global_circular_buffer.cpp
        buffers/global_semaphore.cpp
        buffers/semaphore.cpp
        kernels/kernel.cpp
        allocator/algorithms/free_list.cpp
        allocator/allocator.cpp
        allocator/basic_allocator.cpp
        allocator/l1_banking_allocator.cpp
        program/program.cpp
        dispatch/debug_tools.cpp
        dispatch/command_queue.cpp
        dispatch/worker_config_buffer.cpp
        dispatch/data_collection.cpp
        debug/dprint_server.cpp
        debug/noc_logging.cpp
        debug/watcher_server.cpp
        debug/watcher_device_reader.cpp
        trace/trace.cpp
        trace/trace_buffer.cpp
        event/event.cpp
)

target_include_directories(
    impl
    PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tt_metal>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tt_metal/include>"
)

target_link_libraries(
    impl
    PUBLIC
        Metalium::Metal::LLRT
    PRIVATE
        common
        Boost::smart_ptr
        range-v3::range-v3
)

target_compile_options(impl PUBLIC -Wno-int-to-pointer-cast)

# install(TARGETS impl EXPORT Metal COMPONENT metalium-dev)
