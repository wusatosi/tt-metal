// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_in2 = tt::CBIndex::c_2;
    constexpr auto cb_out0 = tt::CBIndex::c_3;
    constexpr auto cb_intermed = tt::CBIndex::c_4;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        // MULTIPLY
        binary_op_specific_init<false, ELWMUL>();

        cb_wait_front(cb_in0, per_core_block_size);
        cb_wait_front(cb_in1, per_core_block_size);
        cb_reserve_back(cb_intermed, per_core_block_size);

        tile_regs_acquire();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            mul_tiles(cb_in0, cb_in1, i, i, i);
        }

        tile_regs_commit();
        tile_regs_wait();

        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, cb_intermed);
        }
        tile_regs_release();

        cb_push_back(cb_intermed, per_core_block_size);
        cb_pop_front(cb_in0, per_core_block_size);
        cb_pop_front(cb_in1, per_core_block_size);

        // ADD
        binary_op_specific_init<false, ELWADD>();

        cb_wait_front(cb_in2, per_core_block_size);
        cb_wait_front(cb_intermed, per_core_block_size);
        cb_reserve_back(cb_out0, per_core_block_size);

        tile_regs_acquire();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            add_tiles(cb_in2, cb_intermed, i, i, i);
        }

        tile_regs_commit();
        tile_regs_wait();

        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, cb_out0);
        }
        tile_regs_release();

        cb_pop_front(cb_in2, per_core_block_size);
        cb_pop_front(cb_intermed, per_core_block_size);
        cb_push_back(cb_out0, per_core_block_size);
    }
}
}  // namespace NAMESPACE
