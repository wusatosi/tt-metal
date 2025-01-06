// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "dprint.h"

namespace NAMESPACE {
void MAIN {
    DPRINT << "\n";

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_1);

    cb_reserve_back(tt::CBIndex::c_2, 1);
    tile_regs_acquire();

    // Pop tile after tile, copy to DST and pack
    cb_wait_front(tt::CBIndex::c_0, 1);
    cb_wait_front(tt::CBIndex::c_1, 1);

    UNPACK(
        DPRINT << "a " << TSLICE(tt::CBIndex::c_0, 0, SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1})
               << ENDL();
        DPRINT << "b " << TSLICE(tt::CBIndex::c_1, 0, SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1})
               << ENDL(););

    mul_tiles_init();
    mul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);

    tile_regs_commit();

    tile_regs_wait();

    pack_tile(0, tt::CBIndex::c_2);

    cb_pop_front(tt::CBIndex::c_0, 1);
    cb_pop_front(tt::CBIndex::c_1, 1);

    tile_regs_release();

    cb_push_back(tt::CBIndex::c_2, 1);

    cb_wait_front(tt::CBIndex::c_2, 1);
    UNPACK(DPRINT << "mul "
                  << TSLICE(tt::CBIndex::c_2, 0, SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1})
                  << ENDL(););
}
}  // namespace NAMESPACE
