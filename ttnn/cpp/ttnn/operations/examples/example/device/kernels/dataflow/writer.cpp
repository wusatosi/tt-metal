// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t output_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_output = tt::CBIndex::c_2;

    // single-tile ublocks
    const uint32_t cb_page_size = get_tile_size(cb_output);
    const auto cb_data_format = get_dataformat(cb_output);
    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = output_addr, .page_size = cb_page_size, .data_format = cb_data_format};

    cb_wait_front(cb_output, 1);
    uint32_t l1_read_addr = get_read_ptr(cb_output);
    const auto cb_value_addr = get_read_ptr(cb_output);
    noc_async_write_tile(0, s, cb_value_addr);
    noc_async_write_barrier();

    cb_pop_front(cb_output, 1);
}
