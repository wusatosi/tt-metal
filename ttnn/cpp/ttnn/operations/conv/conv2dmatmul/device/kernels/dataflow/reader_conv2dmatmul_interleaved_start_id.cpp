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
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);
    uint32_t batch_size = get_arg_val<uint32_t>(2);
    uint32_t input_width = get_arg_val<uint32_t>(3);
    uint32_t input_height = get_arg_val<uint32_t>(4);
    uint32_t kernel_height = get_arg_val<uint32_t>(5);
    uint32_t kernel_width = get_arg_val<uint32_t>(6);
    uint32_t stick_nbytes = get_arg_val<uint32_t>(7);
    constexpr bool src_is_dram = true;
    constexpr bool dst_is_dram = true;
    constexpr uint32_t cb_id_in0 = 0;
    const DataFormat data_format = get_dataformat(cb_id_in0);

    // Create address generator for row-major tensor
    const InterleavedAddrGen<src_is_dram> s_in = {
        .bank_base_address = src_addr,
        .page_size = stick_nbytes,
    };

    const InterleavedAddrGen<dst_is_dram> s_out = {
        .bank_base_address = dst_addr,
        .page_size = stick_nbytes,
    };
    uint32_t Oh = input_height / kernel_height;
    uint32_t Ow = input_width / kernel_width;
    uint32_t patch_size = kernel_height * kernel_width;
    uint32_t dst_row = batch_size * Oh * Ow;

    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t oh = 0; oh < Oh; oh++) {
            for (uint32_t ow = 0; ow < Ow; ow++) {
                for (uint32_t kh = 0; kh < kernel_height; kh++) {
                    for (uint32_t kw = 0; kw < kernel_width; kw++) {
                        int h = oh * kernel_height + kh;
                        int w = ow * kernel_width + kw;

                        int src_index = (b * input_height + h) * input_width + w;

                        int dst_row = (b * Oh + oh) * Ow + ow;
                        int dst_col = (kh * kernel_width + kw);
                        int dst_index = dst_row * patch_size + dst_col;
                        uint64_t src_noc_addr = get_noc_addr(src_index, s_in);
                        uint64_t dst_noc_addr = get_noc_addr(dst_index, s_out);
                        uint32_t l1_addr = get_write_ptr(cb_id_in0);
                        noc_async_read(src_noc_addr, l1_addr, stick_nbytes);
                        noc_async_read_barrier();
                        noc_async_write(l1_addr, dst_noc_addr, stick_nbytes);
                        noc_async_write_barrier();
                        // print_bf16_pages(l1_addr, stick_nbytes/2, 1);
                    }
                }
            }
        }
    }
    noc_async_read_barrier();
}
