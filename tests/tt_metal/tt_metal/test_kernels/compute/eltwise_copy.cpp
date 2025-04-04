// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);
    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        ckernel::acquire_dst();

        // Pop tile after tile, copy to DST and pack
        ckernel::cb_wait_front(tt::CBIndex::c_0, 1);
        ckernel::cb_reserve_back(tt::CBIndex::c_16, 1);
        ckernel:: copy_tile(tt::CBIndex::c_0, 0, 0);

        ckernel:: pack_tile(0, tt::CBIndex::c_16);

        ckernel::cb_pop_front(tt::CBIndex::c_0, 1);
        ckernel::cb_push_back(tt::CBIndex::c_16, 1);

        ckernel:: release_dst();
    }
}
}  // namespace NAMESPACE
