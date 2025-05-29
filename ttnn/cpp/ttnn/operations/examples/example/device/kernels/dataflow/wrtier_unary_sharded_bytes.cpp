// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/examples/example/device/kernels/dataflow/utils.hpp"

// #define USE_MOVER 1
// #define USE_MEMCPY 1

void kernel_main() {
    DPRINT << "BR starts" << ENDL();
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(0);
    uint32_t dest_buffer_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_from_compute = 1;
    constexpr uint32_t cb_with_buffer = 2;
    // Test variable.
    const auto tile_size_bytes = get_tile_size(cb_with_buffer);
    const auto total_write_bytes = tile_size_bytes * num_tiles_per_core;
    constexpr uint tile_els = 1024;
    constexpr uint face_els = 256;
    constexpr uint num_faces = 4;
    constexpr uint row_els = 16;
    constexpr uint num_rows = 16;
    constexpr uint byte_size = 4;
    constexpr uint one_row_bytes = row_els * byte_size;

    // MODIFY THIS VARIABLE to test each cases.
    bool test_unaligned_addr = true;

    auto l1_read_addr = get_read_ptr(cb_from_compute);
    auto l1_write_addr = get_write_ptr(cb_with_buffer);

    // ------------- write logic ---------------
    cb_wait_front(cb_from_compute, num_tiles_per_core);
    cb_reserve_back(cb_with_buffer, num_tiles_per_core);
#ifdef USE_MOVER
    // Initialize Mover
    Mover mover{};
    auto temp_bytes = one_row_bytes;
    {
        DeviceZoneScopedN("mover 64B");
        auto l1_read_addr = get_read_ptr(cb_from_compute);
        auto l1_write_addr = get_read_ptr(cb_with_buffer);
        for (uint j = 0; j < num_rows * 2; j++) {
            mover.configure(l1_read_addr, l1_write_addr, temp_bytes);
            mover.run();
            l1_read_addr += one_row_bytes;
            l1_write_addr += one_row_bytes;
        }
        mover.wait();
    }

    temp_bytes /= 2;
    {
        DeviceZoneScopedN("mover 32B");
        auto l1_read_addr = get_read_ptr(cb_from_compute);
        auto l1_write_addr = get_read_ptr(cb_with_buffer);
        for (uint j = 0; j < num_rows * 2; j++) {
            mover.configure(l1_read_addr, l1_write_addr, temp_bytes);
            mover.run();
            l1_read_addr += one_row_bytes;
            l1_write_addr += one_row_bytes;
        }
        mover.wait();
    }

    temp_bytes /= 2;
    {
        DeviceZoneScopedN("mover 16B");
        auto l1_read_addr = get_read_ptr(cb_from_compute);
        auto l1_write_addr = get_read_ptr(cb_with_buffer);
        for (uint j = 0; j < num_rows * 2; j++) {
            mover.configure(l1_read_addr, l1_write_addr, temp_bytes);
            mover.run();
            l1_read_addr += one_row_bytes;
            l1_write_addr += one_row_bytes;
        }
        mover.wait();
    }

    temp_bytes /= 2;
    {
        DeviceZoneScopedN("mover 8B");
        auto l1_read_addr = get_read_ptr(cb_from_compute);
        auto l1_write_addr = get_read_ptr(cb_with_buffer);
        for (uint j = 0; j < num_rows; j++) {
            mover.configure(l1_read_addr, l1_write_addr, temp_bytes);
            mover.run();
            l1_read_addr += one_row_bytes;
            l1_write_addr += one_row_bytes;
        }
        mover.wait();
    }

    // Write tile
    {
        DeviceZoneScopedN("mover tile");
        mover.configure(l1_read_addr, l1_write_addr, 4096);
        mover.run();
        mover.wait();
    }
#elif USE_MEMCPY
    if (!test_unaligned_addr) {
        auto temp_bytes = one_row_bytes;
        {
            DeviceZoneScopedN("memcpy 64B");
            auto l1_read_addr = get_read_ptr(cb_from_compute);
            auto l1_write_addr = get_read_ptr(cb_with_buffer);
            for (uint j = 0; j < num_rows; j++) {
                write_through_memcpy(l1_read_addr, l1_write_addr, temp_bytes);
                // add stride
                l1_read_addr += one_row_bytes;
                l1_write_addr += one_row_bytes;
            }
        }

        temp_bytes /= 2;
        {
            DeviceZoneScopedN("memcpy 32B");
            auto l1_read_addr = get_read_ptr(cb_from_compute);
            auto l1_write_addr = get_read_ptr(cb_with_buffer);
            for (uint j = 0; j < num_rows; j++) {
                write_through_memcpy(l1_read_addr, l1_write_addr, temp_bytes);
                // add stride
                l1_read_addr += one_row_bytes;
                l1_write_addr += one_row_bytes;
            }
        }

        temp_bytes /= 2;
        {
            DeviceZoneScopedN("memcpy 16B");
            auto l1_read_addr = get_read_ptr(cb_from_compute);
            auto l1_write_addr = get_read_ptr(cb_with_buffer);
            for (uint j = 0; j < num_rows; j++) {
                write_through_memcpy(l1_read_addr, l1_write_addr, temp_bytes);
                // add stride
                l1_read_addr += one_row_bytes;
                l1_write_addr += one_row_bytes;
            }
        }

        temp_bytes /= 2;
        {
            DeviceZoneScopedN("memcpy 8B");
            auto l1_read_addr = get_read_ptr(cb_from_compute);
            auto l1_write_addr = get_read_ptr(cb_with_buffer);
            for (uint j = 0; j < num_rows; j++) {
                write_through_memcpy(l1_read_addr, l1_write_addr, temp_bytes);
                // add stride
                l1_read_addr += one_row_bytes;
                l1_write_addr += one_row_bytes;
            }
        }

        temp_bytes /= 2;
        {
            DeviceZoneScopedN("memcpy 4B");
            auto l1_read_addr = get_read_ptr(cb_from_compute);
            auto l1_write_addr = get_read_ptr(cb_with_buffer);
            for (uint j = 0; j < num_rows; j++) {
                write_through_memcpy(l1_read_addr, l1_write_addr, temp_bytes);
                // add stride
                l1_read_addr += one_row_bytes;
                l1_write_addr += one_row_bytes;
            }
        }
    } else {
        {
            DeviceZoneScopedN("memcpy 60B unaligned");
            auto l1_read_addr = get_read_ptr(cb_from_compute);
            auto l1_write_addr = get_read_ptr(cb_with_buffer);
            for (uint j = 0; j < num_rows * 2; j++) {
                write_through_memcpy(l1_read_addr, l1_write_addr, 60);
                // add stride
                l1_read_addr += one_row_bytes;
                l1_write_addr += one_row_bytes;
            }
        }
        {
            DeviceZoneScopedN("memcpy 28B unaligned");
            auto l1_read_addr = get_read_ptr(cb_from_compute);
            auto l1_write_addr = get_read_ptr(cb_with_buffer);
            for (uint j = 0; j < num_rows * 2; j++) {
                write_through_memcpy(l1_read_addr, l1_write_addr, 28);
                // add stride
                l1_read_addr += one_row_bytes;
                l1_write_addr += one_row_bytes;
            }
        }
        {
            DeviceZoneScopedN("memcpy 4B unaligned");
            auto l1_read_addr = get_read_ptr(cb_from_compute);
            auto l1_write_addr = get_read_ptr(cb_with_buffer);
            for (uint j = 0; j < num_rows * 2; j++) {
                write_through_memcpy(l1_read_addr, l1_write_addr, 4);
                // add stride
                l1_read_addr += one_row_bytes;
                l1_write_addr += one_row_bytes;
            }
        }
    }

    // Write a tile
    {
        DeviceZoneScopedN("memcpy tile");
        auto l1_read_addr = get_read_ptr(cb_from_compute);
        auto l1_write_addr = get_write_ptr(cb_with_buffer);
        write_through_memcpy(l1_read_addr, l1_write_addr, tile_size_bytes);
    }
#else
    if (!test_unaligned_addr) {
        auto temp_bytes = one_row_bytes;
        {
            DeviceZoneScopedN("noc 64B");
            auto l1_read_addr = get_read_ptr(cb_from_compute);
            auto l1_write_addr = get_read_ptr(cb_with_buffer);
            auto noc_addr = get_noc_addr(l1_read_addr);
            for (uint j = 0; j < num_rows * 2; j++) {
                noc_async_read(noc_addr, l1_write_addr, temp_bytes);
                l1_read_addr += one_row_bytes;
                l1_write_addr += one_row_bytes;
                noc_addr = get_noc_addr(l1_read_addr);
            }
            noc_async_read_barrier();
        }

        temp_bytes /= 2;
        {
            DeviceZoneScopedN("noc 32B");
            auto l1_read_addr = get_read_ptr(cb_from_compute);
            auto l1_write_addr = get_read_ptr(cb_with_buffer);
            auto noc_addr = get_noc_addr(l1_read_addr);
            for (uint j = 0; j < num_rows * 2; j++) {
                noc_async_read(noc_addr, l1_write_addr, temp_bytes);
                l1_read_addr += one_row_bytes;
                l1_write_addr += one_row_bytes;
                noc_addr = get_noc_addr(l1_read_addr);
            }
            noc_async_read_barrier();
        }

        temp_bytes /= 2;
        {
            DeviceZoneScopedN("noc 16B");
            auto l1_read_addr = get_read_ptr(cb_from_compute);
            auto l1_write_addr = get_read_ptr(cb_with_buffer);
            auto noc_addr = get_noc_addr(l1_read_addr);
            for (uint j = 0; j < num_rows * 2; j++) {
                noc_async_read(noc_addr, l1_write_addr, temp_bytes);
                l1_read_addr += one_row_bytes;
                l1_write_addr += one_row_bytes;
                noc_addr = get_noc_addr(l1_read_addr);
            }
            noc_async_read_barrier();
        }

        temp_bytes /= 2;
        {
            DeviceZoneScopedN("noc 8B");
            auto l1_read_addr = get_read_ptr(cb_from_compute);
            auto l1_write_addr = get_read_ptr(cb_with_buffer);
            auto noc_addr = get_noc_addr(l1_read_addr);
            for (uint j = 0; j < num_rows * 2; j++) {
                noc_async_read(noc_addr, l1_write_addr, temp_bytes);
                l1_read_addr += one_row_bytes;
                l1_write_addr += one_row_bytes;
                noc_addr = get_noc_addr(l1_read_addr);
            }
            noc_async_read_barrier();
        }

        temp_bytes /= 2;
        {
            DeviceZoneScopedN("noc 4B");
            auto l1_read_addr = get_read_ptr(cb_from_compute);
            auto l1_write_addr = get_read_ptr(cb_with_buffer);
            auto noc_addr = get_noc_addr(l1_read_addr);
            for (uint j = 0; j < num_rows * 2; j++) {
                noc_async_read(noc_addr, l1_write_addr, temp_bytes);
                l1_read_addr += one_row_bytes;
                l1_write_addr += one_row_bytes;
                noc_addr = get_noc_addr(l1_read_addr);
            }
            noc_async_read_barrier();
        }
    } else {
        {
            DeviceZoneScopedN("noc 60B unaligned");
            auto l1_read_addr = get_read_ptr(cb_from_compute);
            auto l1_write_addr = get_read_ptr(cb_with_buffer);
            l1_read_addr += byte_size;
            l1_write_addr += byte_size;
            auto noc_addr = get_noc_addr(l1_read_addr);
            for (uint j = 0; j < num_rows; j++) {
                noc_async_read(noc_addr, l1_write_addr, 60);
                l1_read_addr += one_row_bytes;
                l1_write_addr += one_row_bytes;
                noc_addr = get_noc_addr(l1_read_addr);
            }
            noc_async_read_barrier();
        }
        {
            DeviceZoneScopedN("noc 28B unaligned");
            auto l1_read_addr = get_read_ptr(cb_from_compute);
            auto l1_write_addr = get_read_ptr(cb_with_buffer);
            l1_read_addr += byte_size;
            l1_write_addr += byte_size;
            auto noc_addr = get_noc_addr(l1_read_addr);
            for (uint j = 0; j < num_rows; j++) {
                noc_async_read(noc_addr, l1_write_addr, 28);
                l1_read_addr += one_row_bytes;
                l1_write_addr += one_row_bytes;
                noc_addr = get_noc_addr(l1_read_addr);
            }
            noc_async_read_barrier();
        }
        {
            auto l1_read_addr = get_read_ptr(cb_from_compute);
            auto l1_write_addr = get_read_ptr(cb_with_buffer);
            l1_read_addr += byte_size;
            l1_write_addr += byte_size;

            DeviceZoneScopedN("noc 4B unaligned");
            auto noc_addr = get_noc_addr(l1_read_addr);
            for (uint j = 0; j < num_rows; j++) {
                noc_async_read(noc_addr, l1_write_addr, 4);
                l1_read_addr += one_row_bytes;
                l1_write_addr += one_row_bytes;
                noc_addr = get_noc_addr(l1_read_addr);
            }
            noc_async_read_barrier();
        }
    }

    // Write a tile
    {
        DeviceZoneScopedN("noc tile");
        auto noc_addr = get_noc_addr(l1_read_addr);
        noc_async_read(noc_addr, l1_write_addr, tile_size_bytes);
        noc_async_read_barrier();
    }
#endif
    // -----------------------------------------

    cb_push_back(cb_with_buffer, num_tiles_per_core);
    cb_pop_front(cb_from_compute, num_tiles_per_core);
    DPRINT << "BR ends" << ENDL();
}
