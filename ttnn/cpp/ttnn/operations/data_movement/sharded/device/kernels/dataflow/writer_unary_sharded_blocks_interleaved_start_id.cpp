// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#define ENABLE_DEBUG_PRINT 1

#if ENABLE_DEBUG_PRINT == 1
    #include "debug/dprint.h"

    inline void print_pages(uint32_t l1_addr, uint32_t pagelen, uint32_t npages, uint32_t start = 0) {
        volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * pagelen;
        for (uint32_t page = 0; page < npages; ++ page) {
            DPRINT << start + page << ": ";
            for (uint32_t j = 0; j < pagelen; ++ j, ++ ptr) {
                DPRINT << BF16(*ptr) << " ";
            }
            DPRINT << ENDL();
        }
    }

    SliceRange sr = SliceRange{ .h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 32, .ws = 4 };

#endif

void kernel_main() {
    const uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    const uint32_t block_height_tiles = get_arg_val<uint32_t>(1);
    const uint32_t block_width_tiles = get_arg_val<uint32_t>(2);
    const uint32_t unpadded_block_height_tiles = get_arg_val<uint32_t>(3);
    const uint32_t unpadded_block_width_tiles = get_arg_val<uint32_t>(4);
    const uint32_t output_width_tiles = get_arg_val<uint32_t>(5); // input width in tiles - block width in tiles
    const uint32_t block_num_tiles = get_arg_val<uint32_t>(6); // block_height_tiles * block_width_tiles
    const uint32_t start_id_offset = get_arg_val<uint32_t>(7);
    const uint32_t start_id_base = get_arg_val<uint32_t>(8);
    const uint32_t start_id = start_id_base + start_id_offset;

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;

    // single-tile ublocks
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    const uint32_t padded_width_diff = (block_width_tiles - unpadded_block_width_tiles) * tile_bytes;

    DPRINT << "HAHAHAHAHAHAHAHAHA" << ENDL();

    uint32_t row_start_tile_id = start_id;
    cb_wait_front(cb_id_out, block_num_tiles);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out);

    // DPRINT << TSLICE(cb_id_out, 0, sr, TSLICE_INPUT_CB, TSLICE_RD_PTR);

    for (uint32_t h = 0; h < unpadded_block_height_tiles; h++) {
        uint32_t tile_id = row_start_tile_id;
        for (uint32_t w = 0; w < unpadded_block_width_tiles; w++) {
            // print_pages(l1_read_addr, tile_bytes / 2, 1);
            noc_async_write_tile(tile_id, s, l1_read_addr);
            tile_id++;
            l1_read_addr += tile_bytes;
            noc_async_write_barrier();
        }
        l1_read_addr += padded_width_diff;
        row_start_tile_id += output_width_tiles;
    }
    cb_pop_front(cb_id_out, block_num_tiles);
}
