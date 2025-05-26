// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"

// #include "debug/dprint.h"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    UNPACK(DPRINT << "===" << tile_id << "===" << ENDL());
    for (uint16_t r = 0; r < 32; ++r) {
        UNPACK(
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
                   << ENDL());
    }
    UNPACK(DPRINT << "++++++" << ENDL());
}

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    // UNPACK(( DPRINT << "Block count=" << uint32_t(per_core_block_cnt) << " tile count=" << per_core_block_tile_cnt <<
    // ENDL() ));
    tilize_init(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16);

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        cb_wait_front(tt::CBIndex::c_0, per_core_block_tile_cnt);
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_tile_cnt);

        for (uint32_t tile_idx = 0; tile_idx < 1; ++tile_idx) {
            print_full_tile(tt::CBIndex::c_0, 0, true);
        }
        tilize_block(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16);
        for (uint32_t tile_idx = 0; tile_idx < 1; ++tile_idx) {
            print_full_tile(tt::CBIndex::c_16, 0, false);
        }

        cb_push_back(tt::CBIndex::c_16, per_core_block_tile_cnt);
        cb_pop_front(tt::CBIndex::c_0, per_core_block_tile_cnt);
    }
}
}  // namespace NAMESPACE
