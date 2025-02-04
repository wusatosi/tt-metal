// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
// #include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    const uint32_t block_size_col = get_compile_time_arg_val(0);
    const uint32_t block_size_row = get_compile_time_arg_val(1);
    const uint32_t third_dim = get_compile_time_arg_val(2);

    // UNPACK (( DPRINT <<  "block_size_col: " << block_size_col << ENDL()));
    // UNPACK (( DPRINT <<  "block_size_row: " << block_size_row << ENDL()));
    // UNPACK (( DPRINT <<  "third_dim: " << third_dim << ENDL()));

    tilize_init(tt::CBIndex::c_0, block_size_row, tt::CBIndex::c_16);
    for (uint32_t b = 0; b < block_size_col * third_dim; ++b) {
        // UNPACK (( DPRINT << "tilizing block " << b << ENDL()));
        cb_wait_front(tt::CBIndex::c_0, block_size_row);
        cb_reserve_back(tt::CBIndex::c_16, block_size_row);

        // for (int32_t r = 0; r < 32; ++r) {
        // SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r+1), .hs = 1, .w0 = 0, .w1 = 64, .ws = 1};
        //     UNPACK(( DPRINT  << TSLICE(tt::CBIndex::c_0, 0, sr, false, false) << ENDL() ));
        // }

        tilize_block(tt::CBIndex::c_0, block_size_row, tt::CBIndex::c_16);

        // UNPACK (( DPRINT << "AFTER TILIZE BLOCK " << ENDL()));
        // for (int32_t r = 0; r < 32; ++r) {
        // SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r+1), .hs = 1, .w0 = 0, .w1 = 64, .ws = 1};
        //     UNPACK(( DPRINT  << TSLICE(tt::CBIndex::c_16, 0, sr) << ENDL() ));
        // }

        cb_push_back(tt::CBIndex::c_16, block_size_row);
        cb_pop_front(tt::CBIndex::c_0, block_size_row);
    }
    tilize_uninit(tt::CBIndex::c_0, tt::CBIndex::c_16);
}
}  // namespace NAMESPACE
