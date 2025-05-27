// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "debug/dprint_tensix.h"
// #include "ttnn/cpp/ttnn/operations/examples/example/device/kernels/compute/utils.hpp"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    DPRINT << "per_core_block_cnt : " << per_core_block_cnt << ENDL();
    DPRINT << "per_core_block_dim : " << per_core_block_dim << ENDL();

    constexpr tt::CBIndex cb_in = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_1;

    init_sfpu(cb_in, cb_out);
    // uint32_t copy_count = 0;
    // uint32_t pack_count = 0;
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_in, 1);
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_in);
            copy_tile(cb_in, 0, 0);
            tile_regs_commit();
            cb_pop_front(cb_in, 1);

            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }
    }

    DPRINT << "TR ends" << ENDL();
}
}  // namespace NAMESPACE
