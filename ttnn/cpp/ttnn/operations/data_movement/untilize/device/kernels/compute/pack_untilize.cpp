// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"

#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);

    constexpr uint32_t num_blocks_per_col = 2;

    constexpr uint32_t block_ct_dim = per_core_block_tile_cnt / num_blocks_per_col;
    constexpr uint32_t full_ct_dim = per_core_block_tile_cnt;

    pack_untilize_init<block_ct_dim, full_ct_dim>(tt::CBIndex::c_0, tt::CBIndex::c_16);

    for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
        cb_wait_front(tt::CBIndex::c_0, block_ct_dim);
        cb_reserve_back(tt::CBIndex::c_16, block_ct_dim);

        pack_untilize_block<block_ct_dim, full_ct_dim>(tt::CBIndex::c_0, 1, tt::CBIndex::c_16);

        cb_push_back(tt::CBIndex::c_16, block_ct_dim);
        cb_pop_front(tt::CBIndex::c_0, block_ct_dim);
    }

    pack_untilize_uninit();
}
}  // namespace NAMESPACE
