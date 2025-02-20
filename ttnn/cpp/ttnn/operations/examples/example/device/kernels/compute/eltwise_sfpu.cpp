// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = 256;
    uint32_t start_id = 0;

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t block_size = 256;

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_1);

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; i += block_size) {
        cb_reserve_back(tt::CBIndex::c_1, block_size);
        cb_wait_front(tt::CBIndex::c_0, block_size);

        for (uint32_t tile_index = 0; tile_index < block_size; ++tile_index) {
            tile_regs_acquire();
            // reconfig_data_format_srca<true>(tt::CBIndex::c_0, tt::CBIndex::c_0); // TRISC kernel duration around
            // 182xx ~ 184xx reconfig_data_format_srca<true>(tt::CBIndex::c_0); // TRISC kernel duration around 176xx ~
            // 178xx
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);  // TRISC kernel duration around 176xx ~ 178xx
            copy_tile(tt::CBIndex::c_0, tile_index, 0);
            tile_regs_commit();

            tile_regs_wait();
            // pack_reconfig_data_format(tt::CBIndex::c_1);
            pack_reconfig_data_format(tt::CBIndex::c_1, tt::CBIndex::c_1);
            pack_tile(0, tt::CBIndex::c_1);
            tile_regs_release();
        }

        cb_pop_front(tt::CBIndex::c_0, block_size);
        cb_push_back(tt::CBIndex::c_1, block_size);
    }
}
}  // namespace NAMESPACE
