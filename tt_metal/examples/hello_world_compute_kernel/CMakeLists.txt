cmake_minimum_required(VERSION 3.16..3.30)

project(MetalHelloWorld)

add_executable(metal_example_hello_world_compute_kernel)
target_sources(metal_example_hello_world_compute_kernel PRIVATE hello_world_compute_kernel.cpp)

find_package(TT-Metalium REQUIRED)
target_link_libraries(metal_example_hello_world_compute_kernel PRIVATE TT::Metalium)
