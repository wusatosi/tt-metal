add_library(ttmetalium_hostdevcommon INTERFACE)
add_library(TT::Metalium::HostDevCommon ALIAS ttmetalium_hostdevcommon)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
    target_sources(
        ttmetalium_hostdevcommon
        PUBLIC
            FILE_SET api
            TYPE HEADERS
            BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/api
            FILES
                api/hostdevcommon/common_runtime_address_map.h
                api/hostdevcommon/common_values.hpp
                api/hostdevcommon/dprint_common.h
                api/hostdevcommon/kernel_structs.h
                api/hostdevcommon/profiler_common.h
    )
endif()

target_include_directories(ttmetalium_hostdevcommon INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>")
