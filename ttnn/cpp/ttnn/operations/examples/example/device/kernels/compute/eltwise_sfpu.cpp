// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/matmul.h"
#include "dprint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = 128;
    uint32_t start_id = 0;

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;
    constexpr uint32_t cb_in2 = 2;
    constexpr uint32_t cb_tmp0 = 24;  // temp
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t block_size = 128;

    constexpr int dst0 = 0;
    constexpr int dst1 = 1;

    init_sfpu(cb_in0, cb_out);

    cb_wait_front(cb_in1, 1);
    cb_wait_front(cb_in2, 1);

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; i += block_size) {
        cb_reserve_back(cb_out, block_size);
        cb_wait_front(cb_in0, block_size);

        // out = (in0 + in1) * in2
        for (uint32_t tile_index = 0; tile_index < block_size; ++tile_index) {
            tile_regs_acquire();
            // reconfig_data_format(cb_in0, cb_in1);
            // reconfig_data_format(cb_tmp0, cb_in0, cb_in2, cb_in1);
            add_tiles_init(cb_in0, cb_in1);
            add_tiles(cb_in0, cb_in1, tile_index, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            cb_reserve_back(cb_tmp0, 1);
            pack_reconfig_data_format(cb_out, cb_tmp0);
            pack_tile(dst0, cb_tmp0);
            cb_push_back(cb_tmp0, 1);
            tile_regs_release();

            // mul
            tile_regs_acquire();
            cb_wait_front(cb_tmp0, 1);
            // reconfig_data_format(cb_tmp0, cb_in2);
            // reconfig_data_format(cb_in0, cb_tmp0, cb_in1, cb_in2);
            mul_tiles_init(cb_tmp0, cb_in2);
            mul_tiles(cb_tmp0, cb_in2, 0, 0, dst0);
            cb_pop_front(cb_tmp0, 1);
            tile_regs_commit();

            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_reconfig_data_format(cb_tmp0, cb_out);
            pack_tile(dst0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }

        cb_pop_front(cb_in0, block_size);
        cb_push_back(cb_out, block_size);
    }

    cb_pop_front(cb_in1, 1);
    cb_pop_front(cb_in2, 1);
}
}  // namespace NAMESPACE
