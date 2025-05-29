// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/examples/example/device/kernels/dataflow/utils.hpp"

#define USE_MOVER 1
// #define USE_MEMCPY 1

void kernel_main() {
    DPRINT << "BR starts" << ENDL();
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(0);
    uint32_t dest_buffer_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_from_compute = 1;
    constexpr uint32_t cb_with_buffer = 2;
    const auto tile_size_bytes = get_tile_size(cb_with_buffer);
    const auto total_write_bytes = tile_size_bytes * num_tiles_per_core;

    // Initialize Mover
    Mover mover{};

    cb_wait_front(cb_from_compute, num_tiles_per_core);
    cb_reserve_back(cb_with_buffer, num_tiles_per_core);

    auto l1_read_addr = get_read_ptr(cb_from_compute);
    auto l1_write_addr = get_write_ptr(cb_with_buffer);

    // ------------- write logic ---------------
#ifdef USE_MOVER
    {
        DeviceZoneScopedN("write through mover");
        mover.configure(l1_read_addr, l1_write_addr, total_write_bytes);
        mover.run();
        mover.wait();
    }
#elif USE_MEMCPY
    write_through_memcpy(l1_write_addr, l1_read_addr, total_write_bytes);
#else
    write_through_noc(l1_read_addr, l1_write_addr, total_write_bytes);
#endif
    // -----------------------------------------

    cb_push_back(cb_with_buffer, num_tiles_per_core);
    cb_pop_front(cb_from_compute, num_tiles_per_core);
    DPRINT << "BR ends" << ENDL();
}
