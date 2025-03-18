// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/untilize.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    uint32_t third_dim = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(4);
    untilize_init(cb_id_in0, cb_id_out);

    uint32_t onetile = 1;
    for (uint32_t b = 0; b < per_core_block_cnt * per_core_block_tile_cnt * third_dim; ++b) {
        cb_wait_front(cb_id_in0, onetile);
        cb_reserve_back(cb_id_out, onetile);

        untilize_block(cb_id_in0, onetile, cb_id_out);

        cb_push_back(cb_id_out, onetile);
        cb_pop_front(cb_id_in0, onetile);
    }
}
}  // namespace NAMESPACE
