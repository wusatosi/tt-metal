include(${PROJECT_SOURCE_DIR}/cmake/flatbuffers.cmake)

# Specify source dir and schema files to be used
set(LIGHTMETAL_SCHEMAS ${CMAKE_CURRENT_SOURCE_DIR}/binary.fbs)

if(NOT TARGET FlatBuffers::FlatBuffers)
    message(FATAL_ERROR "FlatBuffers library not found. Ensure dependencies are correctly included.")
endif()

# Add LightMetal library
add_library(lightmetal STATIC)

# Add source files for the library
target_sources(lightmetal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lightmetal.cpp)

# Generate headers for each FlatBuffer schema and add to the target
foreach(FBS_FILE ${LIGHTMETAL_SCHEMAS})
    GENERATE_FBS_HEADER(${FBS_FILE})
    message(STATUS "Generated header: ${FBS_GENERATED_HEADER}")
    target_sources(lightmetal PRIVATE ${FBS_GENERATED_HEADER})
endforeach()

# Link FlatBuffers library
target_link_libraries(lightmetal PRIVATE FlatBuffers::FlatBuffers)

# Add include directory for generated headers and include the directory containing lightmetal.hpp
target_include_directories(
    lightmetal
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR} # For lightmetal.hpp
        ${PROJECT_SOURCE_DIR} # For trace_buffer.hpp
        ${CMAKE_BINARY_DIR}/include # For fmt/color.h
)
