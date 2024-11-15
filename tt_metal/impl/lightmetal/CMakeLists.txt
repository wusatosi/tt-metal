include(${PROJECT_SOURCE_DIR}/cmake/flatbuffers.cmake)

# Specify source dir and schema files to be used
set(LIGHTMETAL_SCHEMAS ${CMAKE_CURRENT_SOURCE_DIR}/binary.fbs)

if(NOT TARGET FlatBuffers::FlatBuffers)
    message(FATAL_ERROR "FlatBuffers library not found. Ensure dependencies are correctly included.")
endif()

# Function to generate FlatBuffers headers
function(GENERATE_FBS_HEADER FBS_FILE)
    get_filename_component(FBS_FILE_NAME ${FBS_FILE} NAME)
    set(FBS_GENERATED_HEADER "${CMAKE_CURRENT_BINARY_DIR}/${FBS_FILE_NAME}_generated.h")
    add_custom_command(
        OUTPUT
            ${FBS_GENERATED_HEADER}
        COMMAND
            flatc --cpp -o "${CMAKE_CURRENT_BINARY_DIR}" ${FBS_FILE}
        DEPENDS
            flatc
            ${FBS_FILE}
        COMMENT "Building C++ header for ${FBS_FILE}"
    )
    set(FBS_GENERATED_HEADER ${FBS_GENERATED_HEADER} PARENT_SCOPE)
endfunction()

# Add LightMetal library
add_library(lightmetal STATIC)

# Add placeholder to determine linker language
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/dummy.cpp
    PROPERTIES
        LANGUAGE
            CXX
)
target_sources(lightmetal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dummy.cpp)

# Generate headers for each FlatBuffer schema and add to the target
foreach(FBS_FILE ${LIGHTMETAL_SCHEMAS})
    message(STATUS "KCM Processing schema file: ${FBS_FILE}")
    GENERATE_FBS_HEADER(${FBS_FILE})
    message(STATUS "Generated header: ${FBS_GENERATED_HEADER}")
    target_sources(lightmetal PRIVATE ${FBS_GENERATED_HEADER})
endforeach()

# Link FlatBuffers library
target_link_libraries(lightmetal PRIVATE FlatBuffers::FlatBuffers)

# Add include directory for generated headers
target_include_directories(lightmetal PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
