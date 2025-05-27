// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/examples/example/device/kernels/dataflow/utils.hpp"

void kernel_main() {
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(0);
    uint32_t dest_buffer_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_from_compute = 1;
    constexpr uint32_t cb_with_buffer = 2;
    const auto tile_size_bytes = get_tile_size(cb_with_buffer);

    for (uint32_t i = 0; i < num_tiles_per_core; i++) {
        cb_wait_front(cb_from_compute, 1);
        cb_reserve_back(cb_with_buffer, 1);

        auto l1_read_addr = get_read_ptr(cb_from_compute);
        auto l1_write_addr = get_read_ptr(cb_with_buffer);

        std::memcpy(
            reinterpret_cast<uint32_t*>(l1_write_addr), reinterpret_cast<uint32_t*>(l1_read_addr), tile_size_bytes);

        cb_push_back(cb_with_buffer, 1);
        cb_pop_front(cb_from_compute, 1);
    }
}
