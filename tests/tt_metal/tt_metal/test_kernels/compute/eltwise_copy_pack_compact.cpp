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
    // I think this is redundant now
    PACK((llk_pack_init<
          false /*untilize*/,
          false /*zero_output*/,
          false /*tilize*/,
          true /*compact*/,
          num_single_transfer /*block_ct_dim*/>(out_cb_id)));
    // Parity with SDPA case. Use only Packer 0/1
    PACK((llk_pack_reduce_config_v2<
          ReduceDim::REDUCE_COL,
          false /*at_start*/,
          false /*revert*/,
          false /*DEST_ACCUM_EN*/,
          true /*compact*/>(out_cb_id)));
    // Run the outer loop
    for (uint32_t b = 0; b < outer_loop; ++b) {
        // Wait for num_single_transfer tiles to be available in in_cb
        cb_wait_front(in_cb_id, num_single_transfer);
        // Reserve out_cb space for num_single_transfer tiles
        cb_reserve_back(out_cb_id, num_single_transfer);
        // Initializes the packer reads for tile row, assumes same address for the next num_single_transfer calls
        pack_init_compact(0 /* start tile idx from DEST */, out_cb_id /* out_cb_id */, 0 /* output tile idx */);
        for (uint32_t t = 0; t < num_single_transfer; ++t) {
            // Copy num_single_transfer tiles from in_cb to DEST
            tile_regs_acquire();
            copy_tile(in_cb_id, t, 0);
            dprint_tensix_dest_reg(0);
            tile_regs_commit();

            // Pack num_single_transfer tiles to out_cb
            tile_regs_wait();
            pack_tile(0 /* start tile idx from DEST */, out_cb_id /* out_cb_id */, 0 /* output tile idx */);
            if (t == num_single_transfer - 1) {
                // Fill out with zeroes if needed
                pack_tile_last<num_single_transfer>();
            }
            tile_regs_release();
        }
        UNPACK(tt::compute::common::print_full_tile(out_cb_id, 0, false));
        // Move rd ptr from in_cb by num_single_transfer places
        cb_pop_front(in_cb_id, num_single_transfer);
        // Move wr prt from out_cb by num_single_transfer places
        cb_push_back(out_cb_id, num_single_transfer);
    }
    // Revert the changes back
    reduce_revert_delta(out_cb_id);
}
}  // namespace NAMESPACE
