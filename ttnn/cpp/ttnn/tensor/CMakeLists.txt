add_library(ttnn_tensor ${LIB_TYPE})
add_library(TT::NN::Tensor ALIAS ttnn_tensor)

TT_ENABLE_UNITY_BUILD(ttnn_tensor)

target_sources(
    ttnn_tensor
    PRIVATE
        tensor_attributes.cpp
        tensor_impl.cpp
        tensor.cpp
        tensor_ops.cpp
        types.cpp
        storage.cpp
        tensor_spec.cpp
        tensor_utils.cpp
        host_buffer/functions.cpp
        host_buffer/host_buffer.cpp
        host_buffer/memory_pin.cpp
        serialization.cpp
        layout/alignment.cpp
        layout/page_config.cpp
        layout/tensor_layout.cpp
        xtensor/partition.cpp
        flatbuffer/tensor_types_to_flatbuffer.cpp
        flatbuffer/tensor_types_from_flatbuffer.cpp
)

target_include_directories(ttnn_tensor PUBLIC api)

install(TARGETS ttnn_tensor LIBRARY)

#[[

target_link_libraries(${SUBLIBRARY_NAME} PUBLIC ${TTNN_PUBLIC_LINK_LIBRARIES})
target_link_directories(${SUBLIBRARY_NAME} PUBLIC ${TTNN_PUBLIC_LINK_DIRS})
use_precompiled_header(${SUBLIBRARY_NAME})
set(PRECOMPILED_HEADER_TARGET "${PRECOMPILED_HEADER_TARGET}" PARENT_SCOPE)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_definitions(${SUBLIBRARY_NAME} PRIVATE DISABLE_NAMESPACE_STATIC_ASSERT)
endif()
#]]
