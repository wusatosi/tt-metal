// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    ckernel:: init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        ckernel::cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            ckernel:: tile_regs_acquire();

            // Pop tile after tile, copy to DST and pack
            ckernel::cb_wait_front(tt::CBIndex::c_0, 1);
            ckernel:: copy_tile(tt::CBIndex::c_0, 0, 0);

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
            ckernel:: tile_regs_commit();
            ckernel::tile_regs_wait();
            ckernel:: pack_tile(0, tt::CBIndex::c_16);

            ckernel::cb_pop_front(tt::CBIndex::c_0, 1);
            ckernel::tile_regs_release();
        }
        ckernel::cb_push_back(tt::CBIndex::c_16, per_core_block_dim);
    }
}
}  // namespace NAMESPACE
