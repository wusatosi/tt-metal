// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "debug/dprint.h"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "////////////" << ENDL();
    for (uint16_t r = 0; r < 32; ++r) {
        DPRINT << (uint)r << " WRITER: "
               << TileSlice(
                      cb_id,
                      tile_id,
                      SliceRange{
                          .h0 = (uint8_t)r,
                          .h1 = (uint8_t)(r + 1),
                          .hs = (uint8_t)1,
                          .w0 = (uint8_t)0,
                          .w1 = (uint8_t)32,
                          .ws = (uint8_t)1},
                      true,
                      untilize)
               << ENDL();
    }
    DPRINT << "**********" << ENDL();
}

void kernel_main() {
    DPRINT << "WRITER" << ENDL();
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    DPRINT << "Writer args: " << ENDL() << "dst_addr: " << dst_addr << ENDL()
           << "dst_start_tile_id: " << dst_start_tile_id << ENDL() << "num_tiles: " << num_tiles << ENDL();

    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);

    const uint32_t ublock_size_tiles = get_compile_time_arg_val(1);

    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = dst_addr,
        .page_size = get_tile_size(cb_id_out0),
        .data_format = get_dataformat(cb_id_out0),
    };
    uint32_t dst_tile_id = dst_start_tile_id;
    uint32_t l1_read_addr = 0;
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        DPRINT << "Writer args: " << ENDL() << "dst_addr: " << dst_addr << ENDL();
        cb_wait_front(cb_id_out0, ublock_size_tiles);

        l1_read_addr = get_read_ptr(cb_id_out0);

        noc_async_write_tile(dst_tile_id, s, l1_read_addr);
        noc_async_read_barrier();

        cb_pop_front(cb_id_out0, ublock_size_tiles);
        DPRINT << "Writer done a bit" << ENDL();
        dst_tile_id += ublock_size_tiles;
    }

    DPRINT << "Writer done new line" << ENDL();
    print_full_tile(cb_id_out0);
    DPRINT << "Writer done" << ENDL();
    //}
}
