// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/binary_bitwise_sfpu.h"
#include "compute_kernel_api/binary_shift.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_unary/erf_erfc.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "test_op_common.hpp"

namespace NAMESPACE {

void MAIN {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_grad_in = tt::CBIndex::c_2;

    constexpr float kAlpha = 0.70710678118654752440f;  // 1 / sqrt(2)
    constexpr float kBeta = 0.3989422804014327f;       // 1 / sqrt(2π)

    unary_op_init_common(cb_grad_out, cb_grad_in);

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        cb_reserve_back(cb_grad_in, per_core_block_size);

        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            cb_wait_front(cb_grad_out, per_core_block_size);
            cb_wait_front(cb_input, per_core_block_size);

            tile_regs_acquire();
            tile_regs_wait();

            copy_tile(cb_grad_out, i, 0);  // grad => tile 0
            copy_tile(cb_input, i, 1);     // x => tile 1

            mul_binary_tile_init();

            // // Commented snippet is brokem
            // mul_binary_tile(1, 0);
            // pack_tile(1, cb_grad_in);

            mul_binary_tile(0, 1);
            pack_tile(0, cb_grad_in);

            tile_regs_commit();
            tile_regs_release();

            cb_pop_front(cb_grad_out, per_core_block_size);
            cb_pop_front(cb_input, per_core_block_size);
        }
        cb_push_back(cb_grad_in, per_core_block_size);
    }
}
}  // namespace NAMESPACE
