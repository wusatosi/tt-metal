// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_batch = 1;
    uint32_t batches = per_core_block_dim / tiles_per_batch;
    uint32_t remainder_tiles = per_core_block_dim - (batches * tiles_per_batch);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    // constexpr uint32_t one = 0x3f800000u;  // Represents 1.0f
    init_sfpu(cb_input, cb_output);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        for (uint32_t tile_index = 0; tile_index < batches; ++tile_index) {
            cb_reserve_back(cb_output, tiles_per_batch);
            cb_wait_front(cb_input, tiles_per_batch);
            tile_regs_acquire();

            // Pop tile after tile, copy to DST and pack
            for (uint32_t t = 0; t < tiles_per_batch; ++t) {
                copy_tile_to_dst_init_short(cb_input);
                copy_tile(cb_input, t, t);
                exp_tile_init<1u>();
                exp_tile<1u>(t);
                log_tile_init();
                log_tile(t);
                tanh_tile_init();
                tanh_tile(t);

                binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    cb_input);
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    cb_input, t, t);
            }

            tile_regs_commit();

            tile_regs_wait();
            for (uint32_t t = 0; t < tiles_per_batch; ++t) {
                pack_tile(t, cb_output, t);
            }

            tile_regs_release();

            cb_pop_front(cb_input, tiles_per_batch);
            cb_push_back(cb_output, tiles_per_batch);
        }

        if (remainder_tiles > 0) {
            cb_reserve_back(cb_output, remainder_tiles);
            cb_wait_front(cb_input, remainder_tiles);
            tile_regs_acquire();

            for (uint32_t t = 0; t < remainder_tiles; ++t) {
                copy_tile_to_dst_init_short(cb_input);
                copy_tile(cb_input, t, t);
                exp_tile_init<1u>();
                exp_tile<1u>(t);
                log_tile_init();
                log_tile(t);
                tanh_tile_init();
                tanh_tile(t);

                binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    cb_input);
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    cb_input, t, t);
            }

            tile_regs_commit();
            tile_regs_wait();

            for (uint32_t t = 0; t < remainder_tiles; ++t) {
                pack_tile(t, cb_output, batches * tiles_per_batch + t);
            }

            tile_regs_release();
            cb_pop_front(cb_input, remainder_tiles);
            cb_push_back(cb_output, remainder_tiles);
        }
    }
}
}  // namespace NAMESPACE
