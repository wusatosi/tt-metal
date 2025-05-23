// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(3);

    constexpr uint32_t num_blocks_per_col = 2;
    constexpr uint32_t block_ct_dim = per_core_block_tile_cnt / num_blocks_per_col;
    constexpr uint32_t full_ct_dim = per_core_block_tile_cnt;

    pack_untilize_init<block_ct_dim, full_ct_dim>(src_cb_id, out_cb_id);

    cb_reserve_back(out_cb_id, full_ct_dim);

    for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
        cb_wait_front(src_cb_id, block_ct_dim);
        pack_untilize_block<block_ct_dim, full_ct_dim>(src_cb_id, 1, out_cb_id, b);
        cb_pop_front(src_cb_id, block_ct_dim);
    }

    cb_push_back(out_cb_id, full_ct_dim);

    pack_untilize_uninit(out_cb_id);
}
}  // namespace NAMESPACE
