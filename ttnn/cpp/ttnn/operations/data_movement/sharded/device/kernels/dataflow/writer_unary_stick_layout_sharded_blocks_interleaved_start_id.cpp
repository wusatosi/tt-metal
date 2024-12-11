// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

inline void print_pages(uint32_t l1_addr, uint32_t pagelen, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * pagelen;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < pagelen; ++j, ++ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}

void kernel_main() {
    DPRINT << "HIT S2I 1" << ENDL();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t stick_size = get_arg_val<uint32_t>(1);
    const uint32_t block_height = get_arg_val<uint32_t>(2);
    const uint32_t block_width_bytes = get_arg_val<uint32_t>(3);
    const uint32_t padded_block_width_bytes = get_arg_val<uint32_t>(4);
    const uint32_t input_width_offset_bytes = get_arg_val<uint32_t>(5);
    const uint32_t start_id = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);

    constexpr bool dst0_is_dram = get_compile_time_arg_val(1) == 1;
#define dst_stick_size_is_pow2 get_compile_time_arg_val(2) == 1
#if (dst_stick_size_is_pow2)
    constexpr uint32_t dst_log_base_2_of_page_size = get_compile_time_arg_val(3);
    const InterleavedPow2AddrGen<dst0_is_dram> s0 = {
        .bank_base_address = dst_addr + input_width_offset_bytes,
        .log_base_2_of_page_size = dst_log_base_2_of_page_size  // TODO(AP): refactor
    };
#else
    const InterleavedAddrGen<dst0_is_dram> s0 = {
        .bank_base_address = dst_addr + input_width_offset_bytes, .page_size = stick_size};
#endif
    uint32_t stick_id = start_id;
    cb_wait_front(cb_id_out0, block_height);
    uint32_t l1_read_addr_base = get_read_ptr(cb_id_out0);
    uint32_t l1_read_addr = l1_read_addr_base;
    for (uint32_t h = 0; h < block_height; ++h) {
        uint64_t dst_noc_addr = get_noc_addr(stick_id, s0);
        noc_async_write(l1_read_addr, dst_noc_addr, block_width_bytes);
        stick_id++;
        l1_read_addr += padded_block_width_bytes;
        noc_async_write_barrier();
    }

    uint32_t ptr = get_read_ptr(cb_id_out0);
    print_pages(l1_read_addr_base, 64, 3);

    cb_pop_front(cb_id_out0, block_height);
}
