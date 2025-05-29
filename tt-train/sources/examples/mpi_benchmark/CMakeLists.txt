project(mpi_benchmark)

set(MAIN_SOURCES main.cpp)

if(NOT TARGET OpenMPI::MPI)
    set(ULFM_PREFIX /opt/openmpi-v5.0.7-ulfm)
    set(ULFM_LIB ${ULFM_PREFIX}/lib/libmpi.so.40)

    add_library(OpenMPI::MPI SHARED IMPORTED GLOBAL)

    set_target_properties(
        OpenMPI::MPI
        PROPERTIES
            IMPORTED_LOCATION
                "${ULFM_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES
                "${ULFM_PREFIX}/include"
            EXCLUDE_FROM_ALL
                TRUE
    )
endif()

add_executable(mpi_benchmark ${MAIN_SOURCES})
target_link_libraries(
    mpi_benchmark
    PRIVATE
        ttml
        benchmark::benchmark
        OpenMPI::MPI
)
