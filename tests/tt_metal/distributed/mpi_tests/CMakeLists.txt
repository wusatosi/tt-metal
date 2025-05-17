# Find MPI in this scope to make MPI::MPI_CXX available
find_package(MPI REQUIRED)

# Common function to set up MPI test targets
function(add_mpi_test test_name test_source)
    add_executable(${test_name} ${test_source})
    
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(
        ${test_name}
        PRIVATE
            MPI::MPI_CXX
            tt_metal
            test_common_libs
            gtest
            gtest_main
    )
    
    add_test(
        NAME mpirun_${test_name}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${test_name}> ${MPIEXEC_POSTFLAGS}
    )
endfunction()

# Add MPI tests
add_mpi_test(test_mpi_basic test_mpi_basic.cpp)
add_mpi_test(test_mesh_device test_mesh_device.cpp)
