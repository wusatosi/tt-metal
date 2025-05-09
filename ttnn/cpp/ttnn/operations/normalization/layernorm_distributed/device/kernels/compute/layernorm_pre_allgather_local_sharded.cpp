// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes partial layernorm statistics for a dimension shard.
 * For layernorm it computes partial E(x**2) and E(x) for a subset of dimension tiles.
 * For rmsnorm it only computes partial E(x**2).
 * It uses LLK APIs for all mathematical operations.
 */

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"
#include "debug/dprint.h"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

namespace NAMESPACE {
void MAIN {
    uint32_t NCHt = get_arg_val<uint32_t>(0);

    DPRINT << "CL_START" << ENDL();

    constexpr uint32_t tiles_per_dim_shard = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_1;

    constexpr uint32_t cb_x2 = tt::CBIndex::c_6;       // x**2
    constexpr uint32_t cb_partial = tt::CBIndex::c_7;  // Partial results buffer

    cb_wait_front(cb_reduce, 1);  // comes from the reader
    DPRINT << "CL_REDUCE_SCALAR_RECEIVED" << ENDL();

    binary_op_init_common(cb_inp, cb_reduce, cb_x2);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr int dst0 = 0;

        /*
         * Compute x**2
         */
        reconfig_data_format(cb_inp, cb_inp);
        pack_reconfig_data_format(cb_x2);
        mul_tiles_init(cb_inp, cb_inp);
        DPRINT << "CL_COMPUTING_X_SQUARED" << ENDL();

        for (uint32_t wt = 0; wt < tiles_per_dim_shard; wt += blk) {
            cb_wait_front(cb_inp, wt + blk);  // cumulative wait
            DPRINT << "CL_WAIT_INPUT" << wt + blk << "/" << tiles_per_dim_shard << ENDL();
            cb_reserve_back(cb_x2, blk);
            ACQ();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                mul_tiles(cb_inp, cb_inp, wt + wtr, wt + wtr, wtr);
                pack_tile(wtr, cb_x2, wt + wtr);
            }
            REL();
            cb_push_back(cb_x2, blk);
        }

        /*
         * Compute partial sum(x**2) for this dimension shard
         */
        DPRINT << "CL_COMPUTING_SUM_X_SQUARED" << ENDL();
        reconfig_data_format(cb_x2, cb_reduce);
        pack_reconfig_data_format(cb_partial);
        reduce_init_delta<false>(cb_x2, cb_reduce, cb_partial);
        cb_wait_front(cb_x2, tiles_per_dim_shard);
        cb_reserve_back(cb_partial, onetile);
        ACQ();
        for (uint32_t wtr = 0; wtr < tiles_per_dim_shard; wtr++) {
            reduce_tile(cb_x2, cb_reduce, wtr, 0, dst0);
        }
        pack_tile(dst0, cb_partial, 0);
        REL();
        cb_push_back(cb_partial, onetile);
        cb_pop_front(cb_x2, tiles_per_dim_shard);

        reduce_revert_delta(cb_partial);
        DPRINT << "CL_SUM_X_SQUARED_COMPLETE" << ENDL();

#ifndef RMSNORM
        /*
         * Compute partial sum(x) for this dimension shard
         */
        reconfig_data_format(cb_inp, cb_reduce);
        pack_reconfig_data_format(cb_partial);
        reduce_init_delta<false>(cb_inp, cb_reduce, cb_partial);
        cb_reserve_back(cb_partial, onetile);
        ACQ();
        for (uint32_t wtr = 0; wtr < tiles_per_dim_shard; wtr++) {
            reduce_tile(cb_inp, cb_reduce, wtr, 0, dst0);
        }
        pack_tile(dst0, cb_partial, 1);
        REL();
        cb_push_back(cb_partial, onetile);

        reduce_revert_delta(cb_partial);
#endif

        cb_pop_front(cb_inp, tiles_per_dim_shard);
        DPRINT << "CL_BATCH_COMPLETE" << ENDL();
    }

    cb_pop_front(cb_reduce, 1);
    DPRINT << "CL_COMPLETE" << ENDL();
}
}  // namespace NAMESPACE
