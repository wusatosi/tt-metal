// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_tile_cnt = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(1);

    unary_op_init_common(cb_id_in0, cb_id_out);
    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        acquire_dst();

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(cb_id_in0, 1);
        cb_reserve_back(cb_id_out, 1);
        copy_tile(cb_id_in0, 0, 0);

        pack_tile(0, cb_id_out);

        cb_pop_front(cb_id_in0, 1);
        cb_push_back(cb_id_out, 1);

        release_dst();
    }
}
}  // namespace NAMESPACE
