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
    uint32_t input_height = get_arg_val<uint32_t>(3);
    uint32_t input_width = get_arg_val<uint32_t>(4);
    uint32_t kernel_height = get_arg_val<uint32_t>(5);
    uint32_t kernel_width = get_arg_val<uint32_t>(6);
    uint32_t stick_nbytes = get_arg_val<uint32_t>(7);
    uint32_t start_id = get_arg_val<uint32_t>(8);
    uint32_t num_tiles = get_arg_val<uint32_t>(9);
    uint32_t ntiles_per_row = get_arg_val<uint32_t>(10);
    constexpr bool src_is_dram = true;
    constexpr bool dst_is_dram = true;
    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    uint32_t onetile = ntiles_per_row;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);
    DPRINT << "stick_nbytes: " << stick_nbytes << ENDL();
    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGen<dst_is_dram> d = {.bank_base_address = dst_addr, .page_size = stick_nbytes};
    // Create address generator for row-major tensor
    uint32_t OH = input_height / kernel_height;
    uint32_t OW = input_width / kernel_width;
    uint32_t patch_size = kernel_height * kernel_width;
    const uint32_t W_PAD = 32;
    const uint32_t C_PAD = 64 * ntiles_per_row;

    uint32_t tile_cols = (input_width + W_PAD - 1) / W_PAD;
    uint32_t tiles_per_batch = input_height * tile_cols;

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(cb_id_in0, onetile);
        uint64_t l1_write_addr = get_write_ptr(cb_id_in0);
        for (uint32_t j = 0; j < onetile; ++j) {
            noc_async_read_tile(onetile * i + j, s, l1_write_addr);
            l1_write_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
        cb_wait_front(cb_id_in1, onetile);
        uint64_t l1_read_addr = get_read_ptr(cb_id_in1);

        // print_bf16_pages(l1_read_addr, tile_bytes / 2, 1);
        int b = i / tiles_per_batch;
        int bh_index = i % tiles_per_batch;
        int h = bh_index / tile_cols;
        int tile_col = bh_index % tile_cols;

        int w_start = tile_col * W_PAD;
        int w_end = (w_start + W_PAD > input_width) ? input_width : w_start + W_PAD;
        for (int w_local = 0; w_start + w_local < w_end; ++w_local) {
            int w = w_start + w_local;
            uint64_t src = l1_read_addr + w_local * C_PAD;

            int oh = h / kernel_height;
            int ow = w / kernel_width;
            int kh = h % kernel_height;
            int kw = w % kernel_width;

            int dst_row = b * OH * OW + oh * OW + ow;
            int dst_col = (kh * kernel_width + kw);

            uint64_t dst = dst_row * patch_size + dst_col;
            uint64_t dst_addr = get_noc_addr(dst, d);
            noc_async_write(src, dst_addr, stick_nbytes);
            // print_bf16_pages(l1_read_addr, stick_nbytes / 2, 1);
        }
        noc_async_write_barrier();

        cb_pop_front(cb_id_in1, onetile);
    }
}
