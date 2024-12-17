add_library(common STATIC ${COMMON_SRCS}) # FIXME(14541): Should be OBJECT, but can't handle circular deps between Object libs
add_library(Metalium::Metal::Common ALIAS common)

target_sources(
    common
    PRIVATE
        core_assignment.cpp
        core_coord.cpp
        core_descriptor.cpp
        metal_soc_descriptor.cpp
        tt_backend_api_types.cpp
        utils.cpp
        work_split.cpp
)

target_include_directories(
    common
    PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tt_metal>"
)

target_link_libraries(
    common
    PUBLIC
        fmt::fmt-header-only
        nlohmann_json::nlohmann_json
        umd::device
        umd::Firmware
    PRIVATE
        magic_enum
        span
        Metalium::Metal::STL
        Metalium::Metal::LLRT
        yaml-cpp::yaml-cpp
)

# install(TARGETS common EXPORT Metal COMPONENT metalium-dev)
