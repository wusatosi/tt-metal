get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(isMultiConfig)
    if(NOT "Profiler" IN_LIST CMAKE_CONFIGURATION_TYPES)
        list(APPEND CMAKE_CONFIGURATION_TYPES Profiler)
    endif()
    if(NOT "ProfilerWithDebInfo" IN_LIST CMAKE_CONFIGURATION_TYPES)
        list(APPEND CMAKE_CONFIGURATION_TYPES ProfilerWithDebInfo)
    endif()
endif()

set_property(
    GLOBAL
    APPEND
    PROPERTY
        DEBUG_CONFIGURATIONS
            ProfilerWithDebInfo
)

set(CMAKE_C_FLAGS_PROFILER "${CMAKE_C_FLAGS_RELEASE} -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_PROFILER "${CMAKE_CXX_FLAGS_RELEASE} -fno-omit-frame-pointer")
set(CMAKE_EXE_LINKER_FLAGS_PROFILER "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -rdynamic")
set(CMAKE_MODULE_LINKER_FLAGS_PROFILER "${CMAKE_MODULE_LINKER_FLAGS_RELEASE} -rdynamic")
set(CMAKE_SHARED_LINKER_FLAGS_PROFILER "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -rdynamic")
set(CMAKE_STATIC_LINKER_FLAGS_PROFILER "${CMAKE_STATIC_LINKER_FLAGS_RELEASE} -rdynamic")

# Mimic RelWithDebInfo, but avoid -DNDEBUG to keep TT_ASSERT.  Take Debug and increase the -O level.
set(CMAKE_C_FLAGS_PROFILERWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_PROFILERWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer")
set(CMAKE_EXE_LINKER_FLAGS_PROFILERWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -rdynamic")
set(CMAKE_MODULE_LINKER_FLAGS_PROFILERWITHDEBINFO "${CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO} -rdynamic")
set(CMAKE_SHARED_LINKER_FLAGS_PROFILERWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} -rdynamic")
set(CMAKE_STATIC_LINKER_FLAGS_PROFILERWITHDEBINFO "${CMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO} -rdynamic")

# Built as outlined in Tracy documentation (pg.12)
set(TRACY_HOME ${PROJECT_SOURCE_DIR}/tt_metal/third_party/tracy)

set(TRACY_ENABLE OFF) # We'll enable it conditionally
add_subdirectory(${TRACY_HOME} EXCLUDE_FROM_ALL SYSTEM)

set(USE_TRACY "$<OR:$<CONFIG:Profiler,ProfilerWithDebInfo>,$<BOOL:${ENABLE_TRACY}>>") # Keep ENABLE_TRACY for backwards compatibility for now
add_library(TracyWrapper INTERFACE)
add_library(Tracy::Wrapper ALIAS TracyWrapper)
# Propagate the include dirs from Tracy::TracyClient unconditionally; upstream headers will stub out the content.
target_include_directories(TracyWrapper SYSTEM INTERFACE
  $<TARGET_PROPERTY:Tracy::TracyClient,INTERFACE_INCLUDE_DIRECTORIES>
)
# Only link TracyClient if we're using it to prevent the linker from wanting it.
target_link_libraries(TracyWrapper INTERFACE $<${USE_TRACY}:Tracy::TracyClient>)

set_target_properties(
    TracyClient
    PROPERTIES
        EXCLUDE_FROM_ALL
            TRUE
        LIBRARY_OUTPUT_DIRECTORY
            "${PROJECT_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY
            "${PROJECT_BINARY_DIR}/lib"
        OUTPUT_NAME
            "tracy"
)

target_compile_definitions(TracyClient PUBLIC
  "$<${USE_TRACY}:TRACY_ENABLE>"
)
target_compile_options(TracyClient PUBLIC
  "$<$<BOOL:${ENABLE_TRACY}>:-fno-omit-frame-pointer>" # Backwards compatibility
)
target_link_options(TracyClient PUBLIC
  "$<$<BOOL:${ENABLE_TRACY}>:-rdynamic>" # Backwards compatibility
)

# Our current fork of tracy does not have CMake support for these subdirectories
# Once we update, we can change this
include(ExternalProject)
ExternalProject_Add(
    tracy_csv_tools
    PREFIX ${TRACY_HOME}/csvexport/build/unix
    SOURCE_DIR ${TRACY_HOME}/csvexport/build/unix
    BINARY_DIR ${PROJECT_BINARY_DIR}/tools/profiler/bin
    INSTALL_DIR ${PROJECT_BINARY_DIR}/tools/profiler/bin
    STAMP_DIR "${PROJECT_BINARY_DIR}/tmp/tracy_stamp"
    TMP_DIR "${PROJECT_BINARY_DIR}/tmp/tracy_tmp"
    DOWNLOAD_COMMAND
        ""
    CONFIGURE_COMMAND
        ""
    INSTALL_COMMAND
        bash -c "$<${USE_TRACY}:
          cp ${TRACY_HOME}/csvexport/build/unix/csvexport-release .
        >"
    BUILD_COMMAND
        bash -c "
          do_it() {   
            cd ${TRACY_HOME}/csvexport/build/unix &&
            CXX=g++ TRACY_NO_LTO=1 make -f ${TRACY_HOME}/csvexport/build/unix/Makefile
            $<SEMICOLON>
          } &&
          $<IF:${USE_TRACY},do_it,true>
        "
)
ExternalProject_Add(
    tracy_capture_tools
    PREFIX ${TRACY_HOME}/capture/build/unix
    SOURCE_DIR ${TRACY_HOME}/capture/build/unix
    BINARY_DIR ${PROJECT_BINARY_DIR}/tools/profiler/bin
    INSTALL_DIR ${PROJECT_BINARY_DIR}/tools/profiler/bin
    STAMP_DIR "${PROJECT_BINARY_DIR}/tmp/tracy_stamp"
    TMP_DIR "${PROJECT_BINARY_DIR}/tmp/tracy_tmp"
    DOWNLOAD_COMMAND
        ""
    CONFIGURE_COMMAND
        ""
    INSTALL_COMMAND
        bash -c " $<${USE_TRACY}:
          cp ${TRACY_HOME}/capture/build/unix/capture-release .
        >"
    BUILD_COMMAND
        bash -c "
          do_it() {   
            cd ${TRACY_HOME}/capture/build/unix && 
            CXX=g++ TRACY_NO_LTO=1 make -f ${TRACY_HOME}/capture/build/unix/Makefile
            $<SEMICOLON>
          } &&
          $<IF:${USE_TRACY},do_it,true>
        "

)
add_custom_target(
    tracy_tools
    ALL
    DEPENDS
        tracy_csv_tools
        tracy_capture_tools
)
