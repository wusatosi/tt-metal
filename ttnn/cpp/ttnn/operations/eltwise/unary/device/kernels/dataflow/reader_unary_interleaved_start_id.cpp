// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    // uint32_t dummy3 = get_arg_val<uint32_t>(3);
    // uint32_t dummy4 = get_arg_val<uint32_t>(4);
    // uint32_t dummy5 = get_arg_val<uint32_t>(5);
    // uint32_t dummy6 = get_arg_val<uint32_t>(6);
    // uint32_t dummy7 = get_arg_val<uint32_t>(7);
    // uint32_t dummy8 = get_arg_val<uint32_t>(8);
    // uint32_t dummy9 = get_arg_val<uint32_t>(9);
    // uint32_t dummy10 = get_arg_val<uint32_t>(10);
    // uint32_t dummy11 = get_arg_val<uint32_t>(11);
    // uint32_t dummy12 = get_arg_val<uint32_t>(12);

    // uint32_t dummy13 = get_arg_val<uint32_t>(13);
    // uint32_t dummy14 = get_arg_val<uint32_t>(14);
    // uint32_t dummy15 = get_arg_val<uint32_t>(15);
    // uint32_t dummy16 = get_arg_val<uint32_t>(16);
    // uint32_t dummy17 = get_arg_val<uint32_t>(17);
    // uint32_t dummy18 = get_arg_val<uint32_t>(18);
    // uint32_t dummy19 = get_arg_val<uint32_t>(19);
    // uint32_t dummy20 = get_arg_val<uint32_t>(20);
    // uint32_t dummy21 = get_arg_val<uint32_t>(21);
    // uint32_t dummy22 = get_arg_val<uint32_t>(22);

    // uint32_t dummy23 = get_arg_val<uint32_t>(23);
    // uint32_t dummy24 = get_arg_val<uint32_t>(24);
    // uint32_t dummy25 = get_arg_val<uint32_t>(25);
    // uint32_t dummy26 = get_arg_val<uint32_t>(26);
    // uint32_t dummy27 = get_arg_val<uint32_t>(27);
    // uint32_t dummy28 = get_arg_val<uint32_t>(28);
    // uint32_t dummy29 = get_arg_val<uint32_t>(29);
    // uint32_t dummy30 = get_arg_val<uint32_t>(30);
    // uint32_t dummy31 = get_arg_val<uint32_t>(31);
    // uint32_t dummy32 = get_arg_val<uint32_t>(32);

    uint32_t dummy0 = get_common_arg_val<uint32_t>(0);
    uint32_t dummy1 = get_common_arg_val<uint32_t>(1);
    uint32_t dummy2 = get_common_arg_val<uint32_t>(2);
    uint32_t dummy3 = get_common_arg_val<uint32_t>(3);
    uint32_t dummy4 = get_common_arg_val<uint32_t>(4);
    uint32_t dummy5 = get_common_arg_val<uint32_t>(5);
    uint32_t dummy6 = get_common_arg_val<uint32_t>(6);
    uint32_t dummy7 = get_common_arg_val<uint32_t>(7);
    uint32_t dummy8 = get_common_arg_val<uint32_t>(8);
    uint32_t dummy9 = get_common_arg_val<uint32_t>(9);

    uint32_t dummy10 = get_common_arg_val<uint32_t>(10);
    uint32_t dummy11 = get_common_arg_val<uint32_t>(11);
    uint32_t dummy12 = get_common_arg_val<uint32_t>(12);
    uint32_t dummy13 = get_common_arg_val<uint32_t>(13);
    uint32_t dummy14 = get_common_arg_val<uint32_t>(14);
    uint32_t dummy15 = get_common_arg_val<uint32_t>(15);
    uint32_t dummy16 = get_common_arg_val<uint32_t>(16);
    uint32_t dummy17 = get_common_arg_val<uint32_t>(17);
    uint32_t dummy18 = get_common_arg_val<uint32_t>(18);
    uint32_t dummy19 = get_common_arg_val<uint32_t>(19);

    uint32_t dummy20 = get_common_arg_val<uint32_t>(20);
    uint32_t dummy21 = get_common_arg_val<uint32_t>(21);
    uint32_t dummy22 = get_common_arg_val<uint32_t>(22);
    uint32_t dummy23 = get_common_arg_val<uint32_t>(23);
    uint32_t dummy24 = get_common_arg_val<uint32_t>(24);
    uint32_t dummy25 = get_common_arg_val<uint32_t>(25);
    uint32_t dummy26 = get_common_arg_val<uint32_t>(26);
    uint32_t dummy27 = get_common_arg_val<uint32_t>(27);
    uint32_t dummy28 = get_common_arg_val<uint32_t>(28);
    uint32_t dummy29 = get_common_arg_val<uint32_t>(29);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

// read a ublock of tiles from src to CB, and then push the ublock to unpacker
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_reserve_back(cb_id_in0, onetile);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read_tile(i, s, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
    }
}
