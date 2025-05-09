// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/transpose_wh.h"
#include "tt_metal/hw/inc/debug/dprint_pages.h"
#include "tt_metal/hw/inc/debug/dprint_tensix.h"

namespace NAMESPACE {
void MAIN {
    uint32_t NHtWt = get_arg_val<uint32_t>(0);

    transpose_wh_init(tt::CBIndex::c_0, tt::CBIndex::c_16);

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    UNPACK(DPRINT << "NHtWt " << NHtWt << ENDL());
    for (uint32_t n = 0; n < NHtWt; n++) {
        cb_wait_front(tt::CBIndex::c_0, 1);
        cb_reserve_back(tt::CBIndex::c_16, 1);

        UNPACK(DPRINT << "It " << n << ENDL());
        UNPACK(tt::compute::common::print_full_tile(tt::CBIndex::c_0, 0, true));
        UNPACK(DPRINT << ENDL());

        acquire_dst();
        transpose_wh_tile(tt::CBIndex::c_0, 0, 0);
        dprint_tensix_dest_reg(0);
        pack_tile(0, tt::CBIndex::c_16);
        release_dst();

        PACK(DPRINT << "It " << n << ENDL());
        PACK(tt::compute::common::print_full_tile(tt::CBIndex::c_16, 0, true));
        PACK(DPRINT << ENDL());

        cb_push_back(tt::CBIndex::c_16, 1);
        cb_pop_front(tt::CBIndex::c_0, 1);
    }
}
}  // namespace NAMESPACE
