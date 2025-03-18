// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
// #include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    const uint32_t block_size_col = get_compile_time_arg_val(0);
    const uint32_t block_size_row = get_compile_time_arg_val(1);
    const uint32_t third_dim = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(4);

    tilize_init(cb_id_in0, block_size_row, cb_id_out);
    for (uint32_t b = 0; b < block_size_col * third_dim; ++b) {
        cb_wait_front(cb_id_in0, block_size_row);
        cb_reserve_back(cb_id_out, block_size_row);

        tilize_block(cb_id_in0, block_size_row, cb_id_out);

        cb_push_back(cb_id_out, block_size_row);
        cb_pop_front(cb_id_in0, block_size_row);
    }
    tilize_uninit(cb_id_in0, cb_id_out);
}
}  // namespace NAMESPACE
