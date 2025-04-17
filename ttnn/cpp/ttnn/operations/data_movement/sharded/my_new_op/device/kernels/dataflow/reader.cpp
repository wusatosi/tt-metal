// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "debug/dprint.h"
// #include "debug/dprint_pages.h"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint16_t r = 0; r < 32; ++r) {
        DPRINT << (uint)r << " READER: "
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
    DPRINT << "++++++" << ENDL();
}

void kernel_main() {
    DPRINT << "READER" << ENDL();

    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t src1_addr = get_arg_val<uint32_t>(2);
    uint32_t src1_start_tile_id = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);

    DPRINT << "FULL tile READED" << ENDL();

    DPRINT << "Reader args: " << ENDL() << "src0_addr: " << src0_addr << ENDL()
           << "src0_start_tile_id: " << src0_start_tile_id << ENDL() << "src1_addr: " << src1_addr << ENDL()
           << "src1_start_tile_id: " << src1_start_tile_id << ENDL() << "num_tiles: " << num_tiles << ENDL();

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(2);
    uint32_t ublock_size_tiles = get_compile_time_arg_val(4);

    DPRINT << "cb_id_in0: " << cb_id_in0 << ENDL();
    DPRINT << "cb_id_in1: " << cb_id_in1 << ENDL();
    DPRINT << "ublock_size_tiles: " << ublock_size_tiles << ENDL();

    // single-tile ublocks
    uint32_t ublock_size_bytes_0 = get_tile_size(cb_id_in0);
    uint32_t ublock_size_bytes_1 = get_tile_size(cb_id_in1);

    DPRINT << "ublock_size_bytes_0 = " << ublock_size_bytes_0 << ENDL();
    DPRINT << "ublock_size_bytes_1 = " << ublock_size_bytes_1 << ENDL();

    // uint32_t num_tiles = src0_num_tiles > src1_num_tiles ? src0_num_tiles : src1_num_tiles;

    const InterleavedAddrGenFast<true> s0 = {
        .bank_base_address = src0_addr,
        .page_size = get_tile_size(cb_id_in0),
        .data_format = get_dataformat(cb_id_in0),
    };

    const InterleavedAddrGenFast<true> s1 = {
        .bank_base_address = src1_addr,
        .page_size = get_tile_size(cb_id_in1),
        .data_format = get_dataformat(cb_id_in1),
    };

    uint32_t src0_tile_id = src0_start_tile_id;
    uint32_t src1_tile_id = src1_start_tile_id;
    // read ublocks from src0/src1 to CB0/CB1, then push ublocks to compute (unpacker)
    DPRINT << "num_tiles: " << num_tiles << ENDL();
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        DPRINT << "i: " << i << ENDL();
        cb_reserve_back(cb_id_in0, ublock_size_tiles);
        DPRINT << "cb_reserve_back cb_id_in0: " << cb_id_in0 << ENDL();
        cb_reserve_back(cb_id_in1, ublock_size_tiles);
        DPRINT << "cb_reserve_back cb_id_in1: " << cb_id_in1 << ENDL();

        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        noc_async_read_tile(src0_tile_id, s0, l1_write_addr_in0);
        noc_async_read_tile(src1_tile_id, s1, l1_write_addr_in1);
        noc_async_read_barrier();

        cb_push_back(cb_id_in0, ublock_size_tiles);
        cb_push_back(cb_id_in1, ublock_size_tiles);

        src0_tile_id += ublock_size_tiles;
        src1_tile_id += ublock_size_tiles;
    }
    print_full_tile(cb_id_in0);

    // print_full_tile(cb_id_in0);

    DPRINT << "READER DONE" << ENDL();
}
