// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(3);

#ifdef SHORT_INIT
    unary_op_init_common(cb_id_in0, cb_id_out);
    pack_untilize_init_short<per_core_block_tile_cnt>(cb_id_in0, cb_id_out);
#else
    pack_untilize_init<per_core_block_tile_cnt>(cb_id_in0, cb_id_out);
#endif

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        cb_wait_front(cb_id_in0, per_core_block_tile_cnt);
        cb_reserve_back(cb_id_out, per_core_block_tile_cnt);

        pack_untilize_block<per_core_block_tile_cnt>(cb_id_in0, 1, cb_id_out);

        cb_push_back(cb_id_out, per_core_block_tile_cnt);
        cb_pop_front(cb_id_in0, per_core_block_tile_cnt);
    }

    pack_untilize_uninit(cb_id_out);
}
}  // namespace NAMESPACE
