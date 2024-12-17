add_library(tt_metal)
add_library(Metalium::Metal ALIAS tt_metal)

target_sources(
    tt_metal
    PRIVATE
        tt_metal.cpp
        graph/graph_tracking.cpp
)

target_link_libraries(
    tt_metal
    PUBLIC
        Metalium::Metal::Impl
    PRIVATE
        Metalium::Metal::STL
        umd::device
        metal_common_libs
        magic_enum
        fmt::fmt-header-only
        span
        $<$<BOOL:${ENABLE_TRACY}>:TracyClient>
        profiler
        common
        jit_build
        llrt
        detail
        distributed
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
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}> # FIXME: Should never reach out to a larger scope
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:$<TARGET_PROPERTY:Metalium::Metal::Hardware,INTERFACE_INCLUDE_DIRECTORIES>>
    PRIVATE
        ${UMD_HOME}
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

install(TARGETS tt_metal EXPORT Metal LIBRARY COMPONENT metal FILE_SET api COMPONENT metalium-dev)
# TODO: 3rd party libs
# install(
#     TARGETS
#         reflect
#         magic_enum
#         firmware
#         span
#         fmt-header-only
#         device
#     EXPORT Metal
#     COMPONENT
#     metal-dev
# )
install(EXPORT Metal NAMESPACE Metalium:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metalium COMPONENT metalium-dev)
# install(
#     TARGETS
#         fmt-header-only
#     EXPORT ThirdPartyFmt
#     COMPONENT
#     metal-dev
# )
# install(EXPORT ThirdPartyFmt NAMESPACE fmt:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metalium COMPONENT metal-dev)
# install(
#     TARGETS
#         device
#     EXPORT umd
#     COMPONENT
#     metal-dev
# )
# install(EXPORT umd NAMESPACE umd:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metalium COMPONENT metal-dev)

add_subdirectory(hw)
add_subdirectory(common)
add_subdirectory(jit_build)
add_subdirectory(llrt)
add_subdirectory(tools)
add_subdirectory(impl)
add_subdirectory(detail)
add_subdirectory(distributed)
add_subdirectory(tt_stl)
add_subdirectory(examples)

if(BUILD_PROGRAMMING_EXAMPLES)
    add_subdirectory(programming_examples)
endif()
