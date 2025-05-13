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
    DPRINT << "CL_KRNL_START" << ENDL();

    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t tiles_per_dim_shard = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    // DPRINT << "CL_ARGS NCHt=" << NCHt << " tiles_per_dim=" << tiles_per_dim_shard << " blk=" << blk << ENDL();

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_1;
    constexpr uint32_t cb_x2 = tt::CBIndex::c_6;       // x**2
    constexpr uint32_t cb_partial = tt::CBIndex::c_7;  // Partial results buffer

    // CB 1 should now be available
    DPRINT << "CL_WAIT_CB1" << ENDL();
    // cb_wait_front(cb_reduce, 1); // RE-ENABLED // <-- COMMENTED OUT: No longer needed, cb_reduce not populated by
    // reader
    DPRINT << "CL_CB1_READY" << ENDL();

    binary_op_init_common(cb_inp, cb_reduce, cb_x2);  // Check if cb_reduce is truly needed here by LLK

    // DPRINT << "CL_ENTERING_MAIN_LOOP NCHt=" << NCHt << ENDL();
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // DPRINT << "CL_LOOP_ITER ncht=" << ncht << ENDL();
        constexpr int dst0 = 0;

        // Compute x**2
        reconfig_data_format(cb_inp, cb_inp);
        pack_reconfig_data_format(cb_x2);
        mul_tiles_init(cb_inp, cb_inp);
        // DPRINT << "CL_START_X2_COMP" << ENDL();
        for (uint32_t wt = 0; wt < tiles_per_dim_shard; wt += blk) {
            // DPRINT << "CL_X2_WAIT_INP blk=" << (wt + blk) << ENDL();
            cb_wait_front(cb_inp, wt + blk);
            // DPRINT << "CL_X2_INP_READY blk=" << (wt + blk) << ENDL();
            cb_reserve_back(cb_x2, blk);
            ACQ();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                mul_tiles(cb_inp, cb_inp, wt + wtr, wt + wtr, wtr);
            }
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                pack_tile(wtr, cb_x2, wt + wtr);
            }
            REL();
            cb_push_back(cb_x2, blk);
        }
        // DPRINT << "CL_END_X2_COMP" << ENDL();

        // Compute partial sum(x**2)
        reconfig_data_format(cb_x2, cb_reduce);
        pack_reconfig_data_format(cb_partial);
        reduce_init_delta<false>(cb_x2, cb_reduce, cb_partial);
        // DPRINT << "CL_WAIT_X2_FOR_REDUCE target=" << tiles_per_dim_shard << ENDL();
        cb_wait_front(cb_x2, tiles_per_dim_shard);
        // DPRINT << "CL_X2_READY_FOR_REDUCE" << ENDL();
        cb_reserve_back(cb_partial, onetile);
        ACQ();
        for (uint32_t wtr = 0; wtr < tiles_per_dim_shard; wtr++) {
            reduce_tile(cb_x2, cb_reduce, wtr, 0, dst0);
        }
        pack_tile(dst0, cb_partial, 0);
        REL();
        DPRINT << "CL_PUSH_SUMX2" << ENDL();  // Output for writer
        cb_push_back(cb_partial, onetile);
        // DPRINT << "CL_POP_X2 target=" << tiles_per_dim_shard << ENDL();
        cb_pop_front(cb_x2, tiles_per_dim_shard);
        reduce_revert_delta(cb_partial);
        // DPRINT << "CL_SUMX2_DONE" << ENDL();

#ifndef RMSNORM
        // Compute partial sum(x)
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
        DPRINT << "CL_PUSH_SUMX" << ENDL();  // Output for writer
        cb_push_back(cb_partial, onetile);
        reduce_revert_delta(cb_partial);
        // DPRINT << "CL_SUMX_DONE" << ENDL();
#endif

        // DPRINT << "CL_POP_INPUT target=" << tiles_per_dim_shard << ENDL();
        cb_pop_front(cb_inp, tiles_per_dim_shard);
        // DPRINT << "CL_BATCH_COMPLETE ncht=" << ncht << ENDL();
    }
    // DPRINT << "CL_MAIN_LOOP_COMPLETE" << ENDL();

    // CB 1 should now be available to pop
    DPRINT << "CL_POP_CB1" << ENDL();
    // cb_pop_front(cb_reduce, 1); // RE-ENABLED // <-- COMMENTED OUT: Not needed, didn't wait/use
    DPRINT << "CL_KRNL_END" << ENDL();
}
}  // namespace NAMESPACE
