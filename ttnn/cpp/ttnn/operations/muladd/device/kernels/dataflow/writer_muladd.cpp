// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
// #include "debug/dprint_pages.h"
#include "debug/dprint.h"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint16_t r = 0; r < 32; ++r) {
        DPRINT << (uint)r << " : "
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
    uint32_t dst_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);

    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    cb_wait_front(cb_id_out, 1);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out);

    // print_full_tile(cb_id_out,0,false);

    noc_async_write_tile(0, s, l1_read_addr);
    noc_async_write_barrier();
    cb_pop_front(cb_id_out, 1);

    // noc_async_write(l1_read_addr,dst_addr,tile_bytes);
}
