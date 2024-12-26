// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "dprint.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_2;
    constexpr auto cb_one = tt::CBIndex::c_3;
    constexpr auto cb_param = tt::CBIndex::c_4;
    constexpr auto cb_tmp1 = tt::CBIndex::c_5;
    constexpr auto cb_tmp2 = tt::CBIndex::c_6;
    constexpr auto cb_tmp3 = tt::CBIndex::c_16;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    constexpr uint32_t onetile = 1;

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        tile_regs_acquire();
        sub_tiles_to_cb(cb_one, cb_param, cb_tmp1, 0, tile_id, 1, 0);  // 1 - momentum
        mul_tiles_to_cb(cb_param, cb_in0, cb_tmp2, 0, 0, 1, 1);        // momentum * running stats
        mul_tiles_to_cb(cb_tmp1, cb_in1, cb_tmp3, tile_id, 0, 0, 1);   // cb_tmp1 * batch stat
        add_tiles_to_cb(cb_tmp2, cb_tmp3, cb_out0, 0, 0, 1, 1);        // cb_tmp2 * cb_tmp3
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out0);
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
