// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "dprint.h"

namespace NAMESPACE {

void MAIN {
    constexpr auto cb_input = tt::CBIndex::c_0;        // float32
    constexpr auto cb_scaler = tt::CBIndex::c_1;       // bfloat16
    constexpr auto cb_fp32_scaler = tt::CBIndex::c_2;  // float32
    constexpr auto cb_output = tt::CBIndex::c_3;       // bfloat16
    constexpr auto cb_fp32_output = tt::CBIndex::c_4;  // float32

    binary_op_init_common(cb_input, cb_scaler);

    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr uint32_t onetile = 1;

    constexpr bool to_from_int8 = true;

    cb_wait_front(cb_input, onetile);
    cb_wait_front(cb_scaler, onetile);
    cb_wait_front(cb_fp32_scaler, onetile);

    DPRINT << FIXED() << SETPRECISION(10);
    UNPACK(DPRINT << "DST_ACCUM_MODE " << static_cast<uint32_t>(DST_ACCUM_MODE) << "\n";)

    UNPACK(DPRINT << "AAAA cb_input "
                  << TSLICE(cb_input, 0, SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1}) << ENDL();)
    UNPACK(DPRINT << "AAAA cb_scaler "
                  << TSLICE(cb_scaler, 0, SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1}) << ENDL();)

    // float32 output
    {
        tile_regs_acquire();
        cb_wait_front(cb_input, onetile);

        // golden sum: 4.998046875

        // this return 4.9687500000 <--- 16 bits truncated but why?. this is problem.
        // input is fp32, dst register is fp32. no need to truncate this.
        reconfig_data_format<to_from_int8>(cb_input, cb_scaler);
        reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_fp32_output, cb_input, cb_scaler);
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_input, cb_scaler, 0, 0, dst0);

        // this return 4.9960937500 <--- 19 bits truncated but this is ok.
        // reconfig_data_format<to_from_int8>(cb_input, cb_fp32_scaler);
        // reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_fp32_output, cb_input, cb_fp32_scaler);
        // reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_input, cb_fp32_scaler, 0, 0, dst0);

        reduce_revert_delta<ReduceDim::REDUCE_ROW>(cb_fp32_output);
        tile_regs_commit();

        tile_regs_wait();
        pack_reconfig_data_format(cb_fp32_output);
        pack_tile(dst0, cb_fp32_output);
        tile_regs_release();

        cb_push_back(cb_fp32_output, onetile);

        cb_wait_front(cb_fp32_output, onetile);
        UNPACK(DPRINT << "AAAA cb_fp32_output "
                      << TSLICE(cb_fp32_output, 0, SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1})
                      << ENDL();)
    }

    // bfloat16 output
    {
        tile_regs_acquire();
        cb_wait_front(cb_input, onetile);

        // golden sum: 4.998046875

        // this return 4.9687500000 <--- truncated why? this also problem.
        reconfig_data_format<to_from_int8>(cb_input, cb_scaler);
        reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_output, cb_input, cb_scaler);
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_input, cb_scaler, 0, 0, dst0);

        // this return 5.0000000000 <--- ok
        // reconfig_data_format<to_from_int8>(cb_input, cb_fp32_scaler);
        // reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_output, cb_input, cb_fp32_scaler);
        // reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_input, cb_fp32_scaler, 0, 0, dst0);

        reduce_revert_delta<ReduceDim::REDUCE_ROW>(cb_output);
        tile_regs_commit();

        tile_regs_wait();
        pack_reconfig_data_format(cb_output);
        pack_tile(dst0, cb_output);
        tile_regs_release();

        cb_push_back(cb_output, onetile);

        cb_wait_front(cb_output, onetile);
        UNPACK(DPRINT << "AAAA cb_output "
                      << TSLICE(cb_output, 0, SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1})
                      << ENDL();)
    }
}
}  // namespace NAMESPACE
