// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {

void MAIN {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_mask = tt::CBIndex::c_1;
    constexpr auto cb_bcast_scaler = tt::CBIndex::c_2;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    binary_op_init_common(cb_in0, cb_bcast_scaler);

    constexpr int dst0 = 0;
    constexpr int dst_mask = 1;

    cb_reserve_back(cb_out0, 1);
    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_mask, 1);

    tile_regs_acquire();

    copy_tile_init_with_dt(cb_in0);
    copy_tile(cb_in0, 0, dst0);

    copy_tile_init_with_dt(cb_mask);
    copy_tile(cb_mask, 0, dst_mask);

    mask_tile_init();
    mask_tile(dst0, dst_mask);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_out0);
    tile_regs_release();

    cb_push_back(cb_out0, 1);

}
}  // namespace NAMESPACE
