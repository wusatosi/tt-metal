# Built as outlined in Tracy documentation (pg.12)
set(TRACY_HOME ${PROJECT_SOURCE_DIR}/tt_metal/third_party/tracy)

if(NOT ENABLE_TRACY)
    # Stub Tracy::TracyClient to provide the headers which themselves provide stubs
    add_library(TracyClient INTERFACE)
    add_library(Tracy::TracyClient ALIAS TracyClient)
    target_include_directories(TracyClient SYSTEM INTERFACE ${TRACY_HOME}/public)
    return()
endif()

add_subdirectory(${TRACY_HOME})

set_target_properties(
    TracyClient
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
            "${PROJECT_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY
            "${PROJECT_BINARY_DIR}/lib"
        POSITION_INDEPENDENT_CODE
            ON # this is equivalent to adding -fPIC
        ADDITIONAL_CLEAN_FILES
            "${PROJECT_BINARY_DIR}/tools"
        OUTPUT_NAME
            "tracy"
)

target_compile_definitions(TracyClient PUBLIC TRACY_ENABLE)
target_compile_options(TracyClient PUBLIC -fno-omit-frame-pointer)
target_link_options(TracyClient PUBLIC -rdynamic)

# Our current fork of tracy does not have CMake support for these subdirectories
# Once we update, we can change this
include(ExternalProject)
#ExternalProject_Add(
#tracy-capture
#SOURCE_DIR ${TRACY_HOME}/capture
#CMAKE_ARGS
#-DCMAKE_BUILD_TYPE=Debug -DNO_FILESELECTOR=ON -DCMAKE_CXX_COMPILER=clang++-17 -DCMAKE_C_COMPILER=clang-17
#BINARY_DIR ${PROJECT_BINARY_DIR}/tools/profiler/capture
#STAMP_DIR ${PROJECT_BINARY_DIR}/tmp/tracy_stamp
#TMP_DIR ${PROJECT_BINARY_DIR}/tmp/tracy_tmp
#INSTALL_COMMAND
#${CMAKE_COMMAND} -E copy tracy-capture ${PROJECT_BINARY_DIR}/tools/profiler/bin/capture-release
#)
#ExternalProject_Add(
#tracy-csvexport
#SOURCE_DIR ${TRACY_HOME}/csvexport
#CMAKE_ARGS
#-DCMAKE_BUILD_TYPE=Debug -DNO_FILESELECTOR=ON -DCMAKE_CXX_COMPILER=clang++-17 -DCMAKE_C_COMPILER=clang-17
#BINARY_DIR ${PROJECT_BINARY_DIR}/tools/profiler/csvexport
#STAMP_DIR ${PROJECT_BINARY_DIR}/tmp/tracy_stamp
#TMP_DIR ${PROJECT_BINARY_DIR}/tmp/tracy_tmp
#INSTALL_COMMAND
#${CMAKE_COMMAND} -E copy tracy-csvexport ${PROJECT_BINARY_DIR}/tools/profiler/bin/csvexport-release
#)
add_custom_target(ALL)
