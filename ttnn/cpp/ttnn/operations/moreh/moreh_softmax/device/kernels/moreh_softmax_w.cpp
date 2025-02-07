// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "dprint.h"

namespace NAMESPACE {

void MAIN {
    constexpr auto cb_input = tt::CBIndex::c_0;   // float32
    constexpr auto cb_scaler = tt::CBIndex::c_1;  // bfloat16
    constexpr auto cb_scaler_fp32 = tt::CBIndex::c_2;  // float32
    constexpr auto cb_output = tt::CBIndex::c_3;  // bfloat16
    constexpr auto cb_output_fp32 = tt::CBIndex::c_4;  // float32
    constexpr auto cb_bfp8 = tt::CBIndex::c_5;    // bfloat16

    binary_op_init_common(cb_input, cb_output);

    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr uint32_t onetile = 1;
    constexpr bool flag_to_from_int8 = false;

    cb_wait_front(cb_input, onetile);

    DPRINT << FIXED() << SETPRECISION(10);
    UNPACK(DPRINT << "DST_ACCUM_MODE " << static_cast<uint32_t>(DST_ACCUM_MODE) << "\n";)

    UNPACK(DPRINT << "AAAA cb_input "
                  << TSLICE(cb_input, 0, SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1}) << ENDL();)

    tile_regs_acquire();
    cb_wait_front(cb_scaler_fp32, onetile);

    reconfig_data_format_srca<true>(cb_scaler_fp32);
    copy_tile_init(cb_scaler_fp32);
    copy_tile(cb_scaler_fp32, 0, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_scaler);
    pack_tile(dst0, cb_scaler);
    tile_regs_release();

    cb_push_back(cb_scaler, onetile);

    cb_wait_front(cb_scaler, onetile);
    cb_wait_front(cb_scaler_fp32, onetile);
    UNPACK(DPRINT << "AAAA cb_scaler "
                  << TSLICE(cb_scaler, 0, SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 3, .ws = 1}) << ENDL();)
    UNPACK(DPRINT << "AAAA cb_scaler_fp32 "
                  << TSLICE(cb_scaler_fp32, 0, SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 3, .ws = 1})
                  << ENDL();)

    tile_regs_acquire();
    cb_wait_front(cb_input, onetile);
    reconfig_data_format<true>(cb_input, cb_scaler_fp32);
    reduce_init_delta<false, REDUCE_OP, REDUCE_DIM>(cb_output_fp32, cb_input, cb_scaler_fp32);
    reduce_tile<REDUCE_OP, REDUCE_DIM>(cb_input, cb_scaler_fp32, 0, 0, dst0);

    reduce_revert_delta(cb_output_fp32);
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_output_fp32);
    pack_tile(dst0, cb_output_fp32);
    tile_regs_release();

    cb_push_back(cb_output_fp32, onetile);

    cb_wait_front(cb_output_fp32, onetile);
    UNPACK(DPRINT << "AAAA cb_output_fp32 "
                  << TSLICE(cb_output_fp32, 0, SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 3, .ws = 1})
                  << ENDL();)
}
}  // namespace NAMESPACE
