// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/transpose_wh.h"

namespace NAMESPACE {
void MAIN {
    uint32_t NHtWt = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(1);

    transpose_wh_init(cb_id_in0, cb_id_out);

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    for (uint32_t n = 0; n < NHtWt; n++) {
        cb_wait_front(cb_id_in0, 1);
        cb_reserve_back(cb_id_out, 1);

        acquire_dst();
        transpose_wh_tile(cb_id_in0, 0, 0);
        pack_tile(0, cb_id_out);
        release_dst();

        cb_push_back(cb_id_out, 1);
        cb_pop_front(cb_id_in0, 1);
    }
}
}  // namespace NAMESPACE
