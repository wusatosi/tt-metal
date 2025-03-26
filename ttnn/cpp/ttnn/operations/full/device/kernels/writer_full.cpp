// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "dprint.h"

constexpr uint32_t onetile = 1;

void kernel_main() {
    // fill bfp8 value
    constexpr uint32_t cb_value = get_compile_time_arg_val(0);
    cb_reserve_back(cb_value, onetile);

    uint32_t write_addr = get_write_ptr(cb_value);

    auto ptr = reinterpret_cast<uint8_t*>(write_addr);

    if (false) {
        // 1
        // exp [0, 64)
        ptr[0] = 127;

        // sign and mantissa [64, ~]
        for (int i = 0; i < 16; i++) {
            ptr[64 + i] = 0x00;
        }
        ptr[64] = 0x40;
    }

    if (false) {
        // not assert but don't know why
        for (int i = 0; i < 16; i++) {
            ptr[64 + i] = 0x00;
        }
        ptr[64] = 0x10;
    }

    if (true) {
        ptr[0] = 1;
        for (int i = 0; i < 16; i++) {
            ptr[64 + i] = 0x10;
        }
    }

    cb_push_back(cb_value, 1);

    // write to dram
    uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t cb_page_size = get_tile_size(cb_value);
    const auto cb_data_format = get_dataformat(cb_value);

    DPRINT << "cb_page_size: " << cb_page_size << " cb_data_format: " << static_cast<uint32_t>(cb_data_format)
           << ENDL();

    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = output_addr, .page_size = cb_page_size, .data_format = cb_data_format};

    cb_wait_front(cb_value, 1);
    const auto cb_value_addr = get_read_ptr(cb_value);
    noc_async_write_tile(0, s, cb_value_addr);
    noc_async_write_barrier();
    cb_pop_front(cb_value, 1);
}
