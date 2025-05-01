// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "debug/dprint.h"
#include "debug/dprint_tensix.h"
#include "debug/dprint_pages.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_single_transfer = get_compile_time_arg_val(1);
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(3);

    constexpr uint32_t outer_loop = num_tiles / num_single_transfer;

    unary_op_init_common(in_cb_id, out_cb_id);
    // Parity with SDPA case. Use only Packer 0/1
    PACK((
        llk_pack_reduce_config_v2<ReduceDim::REDUCE_COL, false /*at_start*/, false /*revert*/, false /*DEST_ACCUM_EN*/>(
            out_cb_id)));
    // Run the outer loop
    for (uint32_t b = 0; b < outer_loop; ++b) {
        // Wait for num_single_transfer tiles to be available in in_cb
        cb_wait_front(in_cb_id, num_single_transfer);
        // Acquire DEST reg for MATH/PACK
        acquire_dst();
        // Reserve out_cb space for num_single_transfer tiles
        cb_reserve_back(out_cb_id, num_single_transfer);

        // Copy num_single_transfer tiles from in_cb to DEST
        for (uint32_t t = 0; t < num_single_transfer; ++t) {
            copy_tile(in_cb_id, t, t);
            dprint_tensix_dest_reg(t);
        }
        // Pack num_single_transfer tiles to out_cb
        pack_tile_compact(0 /* start tile idx from DEST */, out_cb_id /* out_cb_id */, 0 /* output tile idx */);
        // for (uint32_t t = 0; t < num_single_transfer; ++t) {
        //     pack_tile(t, out_cb_id);
        // }
        UNPACK(tt::compute::common::print_full_tile(out_cb_id, 0, false));
        // Release DEST reg marking compute/pack complete
        release_dst();
        // Move rd ptr from in_cb by num_single_transfer places
        cb_pop_front(in_cb_id, num_single_transfer);
        // Move wr prt from out_cb by num_single_transfer places
        cb_push_back(out_cb_id, num_single_transfer);
    }
}
}  // namespace NAMESPACE
