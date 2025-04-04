// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {

void MAIN {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_mask = tt::CBIndex::c_1;
    constexpr auto cb_bcast_scaler = tt::CBIndex::c_2;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr auto cb_exps = tt::CBIndex::c_24;
    constexpr auto cb_recipsumexps = tt::CBIndex::c_25;
    constexpr auto cb_max = tt::CBIndex::c_26;
    constexpr auto cb_x_m_max = tt::CBIndex::c_27;
    constexpr auto cb_tmp = tt::CBIndex::c_28;

    binary_op_init_common(cb_in0, cb_bcast_scaler, cb_out0);

    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr uint32_t onetile = 1;

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);

    ckernel::cb_wait_front(cb_mask, onetile);
    ckernel::cb_wait_front(cb_bcast_scaler, onetile);

    for (uint32_t n = 0; n < N; ++n) {
        // find max value
        if (Wt == 1) {
            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, /*pop0=*/0, /*popm=*/0);

            reduce_tile_to_cb<false, PoolType::MAX, REDUCE_DIM>(
                cb_tmp, cb_bcast_scaler, cb_max, Wt, /*pop0=*/1, /*pop1=*/0);
        } else {
            reduce_tile_to_cb<false, PoolType::MAX, REDUCE_DIM>(
                cb_in0, cb_bcast_scaler, cb_max, Wt - 1, /*pop0=*/0, /*pop1=*/0);

            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, Wt - 1, 0, /*pop0=*/0, /*popm=*/0);

            ckernel::cb_wait_front(cb_max, 1);
            ckernel::cb_wait_front(cb_tmp, 1);

            ckernel:: tile_regs_acquire();
            copy_tile_init_with_dt(cb_max);
            ckernel:: copy_tile(cb_max, 0, dst0);

            constexpr uint32_t bcast_scaler0 = 0;  // 0th index from bcast_scaler CB
            reduce_init_delta_with_dt<false, PoolType::MAX, REDUCE_DIM>(cb_max, cb_tmp, cb_bcast_scaler);
            ckernel::reduce_tile<PoolType::MAX, REDUCE_DIM>(cb_tmp, cb_bcast_scaler, 0, bcast_scaler0, dst0);
            ckernel::reduce_revert_delta(cb_max);
            ckernel:: tile_regs_commit();

            ckernel::tile_regs_wait();
            pack_tile_with_dt(dst0, cb_max);
            ckernel::tile_regs_release();

            ckernel::cb_pop_front(cb_max, 1);
            ckernel::cb_pop_front(cb_tmp, 1);
            ckernel::cb_push_back(cb_max, 1);
        }

        // compute x - max(x)
        ckernel::cb_reserve_back(cb_x_m_max, Wt);
        ckernel::cb_wait_front(cb_in0, Wt);
        ckernel::cb_wait_front(cb_max, 1);

        for (uint32_t w = 0; w < Wt; ++w) {
            ckernel:: tile_regs_acquire();
            sub_bcast_cols_init_short_with_dt(cb_in0, cb_max);
            sub_tiles_bcast<BroadcastType::COL>(cb_in0, cb_max, w, 0, dst0);
            ckernel:: tile_regs_commit();

            ckernel::tile_regs_wait();
            pack_tile_with_dt(dst0, cb_x_m_max);
            ckernel::tile_regs_release();
        }
        ckernel::cb_pop_front(cb_max, 1);
        ckernel::cb_pop_front(cb_in0, Wt);
        ckernel::cb_push_back(cb_x_m_max, Wt);

        // compute exp(x - max(x))
        ckernel::cb_reserve_back(cb_exps, Wt);
        ckernel::cb_wait_front(cb_x_m_max, Wt);
        for (uint32_t w = 0; w < Wt; ++w) {
            ckernel:: tile_regs_acquire();
            copy_tile_init_with_dt(cb_x_m_max);
            ckernel:: copy_tile(cb_x_m_max, w, dst0);

#ifndef SOFTMAX
            negative_tile_init();
            negative_tile(dst0);
#endif

            exp_tile_init();
            exp_tile(dst0);

            if (w == Wt - 1) {
                copy_tile_init_with_dt(cb_mask);
                ckernel:: copy_tile(cb_mask, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
            ckernel:: tile_regs_commit();

            ckernel::tile_regs_wait();
            pack_tile_with_dt(dst0, cb_exps);
            ckernel::tile_regs_release();
        }
        ckernel::cb_push_back(cb_exps, Wt);

#ifdef LOG
        // log(sum)
        reduce_and_log_tile_to_cb<false, PoolType::SUM, REDUCE_DIM>(
            cb_exps, cb_bcast_scaler, cb_recipsumexps, Wt, /*pop0=*/Wt, /*pop1=*/0);
#else
        // 1/sum
        reduce_and_recip_tile_to_cb<false, PoolType::SUM, REDUCE_DIM>(
            cb_exps, cb_bcast_scaler, cb_recipsumexps, Wt, /*pop0=*/0, /*pop1=*/0);
#endif

        // compute final result
        ckernel::cb_reserve_back(cb_out0, Wt);
        ckernel::cb_wait_front(cb_x_m_max, Wt);
        ckernel::cb_wait_front(cb_recipsumexps, 1);

#ifndef LOG
        ckernel::cb_wait_front(cb_exps, Wt);
#endif

        for (uint32_t w = 0; w < Wt; w += onetile) {
#ifdef LOG
            // x - max - log(sum)
            ckernel:: tile_regs_acquire();
            sub_bcast_cols_init_short_with_dt(cb_x_m_max, cb_recipsumexps);
            sub_tiles_bcast<BroadcastType::COL>(cb_x_m_max, cb_recipsumexps, w, 0, dst0);
            ckernel:: tile_regs_commit();

            ckernel::tile_regs_wait();
            pack_tile_with_dt(dst0, cb_out0);
            ckernel::tile_regs_release();
#else
            // exp(x - max) / psum
            ckernel:: tile_regs_acquire();
            mul_bcast_cols_init_short_with_dt(cb_exps, cb_recipsumexps);
            mul_tiles_bcast_cols(cb_exps, cb_recipsumexps, w, 0, dst0);
            ckernel:: tile_regs_commit();

            ckernel::tile_regs_wait();
            pack_tile_with_dt(dst0, cb_out0);
            ckernel::tile_regs_release();
#endif
        }

        ckernel::cb_pop_front(cb_recipsumexps, 1);
        ckernel::cb_pop_front(cb_x_m_max, Wt);
        ckernel::cb_push_back(cb_out0, Wt);
#ifndef LOG
        ckernel::cb_pop_front(cb_exps, Wt);
#endif
    }
}
}  // namespace NAMESPACE
