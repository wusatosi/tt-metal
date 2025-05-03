// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

inline void print_bf16_pages(uint32_t l1_addr, uint32_t elts_per_page, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * elts_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < elts_per_page; ++j, ++ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t input_width = get_arg_val<uint32_t>(1);
    uint32_t input_height = get_arg_val<uint32_t>(2);
    uint32_t stick_nbytes = get_arg_val<uint32_t>(3);
    constexpr bool dst_is_dram = true;
    constexpr uint32_t cb_id_out0 = 0;
    const DataFormat data_format = get_dataformat(cb_id_out0);

    // Create address generator for row-major tensor
    const InterleavedAddrGen<dst_is_dram> s = {.bank_base_address = dst_addr, .page_size = stick_nbytes};

    // Calculate total number of elements per row
    uint32_t elements_per_row = input_width;

    // Process each row
    for (uint32_t row = 0; row < input_height; row++) {
        // Calculate row offset in bytes
        uint32_t row_offset = row * elements_per_row * stick_nbytes;

        // Process each element in the row
        for (uint32_t col = 0; col < input_width; col++) {
            // Reserve space in circular buffer
            // cb_reserve_back(cb_id_out0, 1);
            // uint32_t src_addr = get_read_ptr(cb_id_out0);

            // Calculate column offset in bytes
            // uint32_t col_offset = col * stick_nbytes;

            // Calculate total offset for this element
            // uint32_t total_offset = row_offset + col_offset;

            // uint64_t dst_noc_addr = get_noc_addr(row * input_width + col, s);
            // Write the data using the address generator
            // noc_async_write(src_addr, dst_noc_addr , stick_nbytes);
            // noc_async_write_barrier();
            // print_bf16_pages(src_addr, stick_nbytes/2, 1);

            // Pop the data from the circular buffer
            // cb_pop_front(cb_id_out0, 1);
        }
    }

    // Wait for all writes to complete
    // noc_async_write_barrier();
}
