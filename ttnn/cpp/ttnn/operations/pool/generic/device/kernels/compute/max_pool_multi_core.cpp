// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/pack_untilize.h"

#define DEBUG_PRINT 1

#if DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "tt_metal/hw/inc/debug/dprint_tensix.h"
#include "tt_metal/hw/inc/circular_buffer.h"

inline void print_pages(uint32_t l1_addr, uint32_t pagelen, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * pagelen;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < pagelen; ++j, ++ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}
#endif

template <
    uint32_t num_output_tiles,
    bool is_partial_tile,
    uint32_t split_reader,
    uint32_t unpA_face_r_dim,
    uint32_t in_nblocks_c>
inline void reduce_h_fused(
    const uint32_t in_cb_id, const uint32_t in_scalar_cb_id, const uint32_t in_stick_index, const uint32_t out_cb_id) {
    constexpr uint32_t num_faces_in_tile = is_partial_tile ? 1 : 2;
    constexpr uint32_t num_out_rows = 1;
    for (uint32_t b_i = 0; b_i < in_nblocks_c; ++b_i) {
        cb_reserve_back(out_cb_id, 1);
        const uint32_t curr_in_cb_id = split_reader ? (in_cb_id + (in_stick_index & 0x1)) : in_cb_id;
        cb_wait_front(curr_in_cb_id, 1);
        tile_regs_acquire();
        unpack_tilizeA_B_block(
            curr_in_cb_id,
            in_scalar_cb_id,
            num_output_tiles,
            0 /*tile idx for Src b is 0 because only 1 tile of constants is loaded*/,
            num_faces_in_tile /* unpack 1 or 2 faces ) */,
            unpA_face_r_dim);
        for (uint32_t c_i = 0; c_i < num_output_tiles; ++c_i) {
            reduce_tile_math(c_i, num_faces_in_tile /* reduce 1 or 2 faces */);
        }

        // dprint_tensix_dest_reg(0);

        cb_pop_front(curr_in_cb_id, 1);
        tile_regs_wait();
        tile_regs_commit();
        pack_untilize_dst<num_output_tiles>(
            out_cb_id, 1 /*out_subblock_h*/, 0, num_out_rows, num_faces_in_tile); /* pack 1 row (1x16 or 1x32) */
        tile_regs_release();

        if (curr_in_cb_id == in_cb_id) {
            PACK(uint32_t ptr = get_local_cb_interface(out_cb_id).fifo_rd_ptr;);
            PACK(print_pages(ptr, 64, 1););
        }

        cb_push_back(out_cb_id, 1);
    }
}

namespace NAMESPACE {

void MAIN {
    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet.
    constexpr uint32_t in_ntiles_hw = get_compile_time_arg_val(0);
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(1);
    DPRINT << "COMPUTE in_ntiles_hw: " << in_ntiles_hw << ", in_ntiles_c: " << in_ntiles_c << ENDL();
    constexpr uint32_t window_size_hw = get_compile_time_arg_val(3);
    constexpr uint32_t out_h = get_compile_time_arg_val(4);
    constexpr uint32_t out_w = get_compile_time_arg_val(5);

    constexpr uint32_t split_reader = get_compile_time_arg_val(12);

    constexpr uint32_t nsticks_per_core = get_compile_time_arg_val(13);
    constexpr uint32_t in_c = get_compile_time_arg_val(14);
    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(15);
    DPRINT << "COMPUTE in_c: " << in_c << ", in_nblocks_c: " << in_nblocks_c << ENDL();

    constexpr uint32_t in_cb_id = tt::CBIndex::c_0;  // and tt::CBIndex::c_1 for split reader
    constexpr uint32_t in_scalar_cb_id = tt::CBIndex::c_4;
    constexpr uint32_t in_tiled_cb_id = tt::CBIndex::c_24;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_16;

    constexpr bool is_partial_tile = in_c < 32;
    static_assert((!is_partial_tile || (in_c == 16)), "Partial tile must have c_dim 16");
    constexpr uint32_t num_faces_in_tile = is_partial_tile ? 1 : 2;
    constexpr uint32_t num_out_rows = 1;

    constexpr uint32_t num_output_tiles = in_ntiles_c / in_nblocks_c;
    DPRINT << "COMPUTE num_output_tiles: " << num_output_tiles << ENDL();
    tilizeA_B_reduce_init<true>(
        in_cb_id, in_scalar_cb_id, num_output_tiles, out_cb_id, num_faces_in_tile, window_size_hw);
    pack_untilize_dst_init_short<in_ntiles_c>(
        out_cb_id, num_out_rows, num_faces_in_tile); /* pack 1 row (1x16 or 1x32) */

    cb_wait_front(in_scalar_cb_id, 1);
    for (uint32_t i = 0; i < nsticks_per_core; ++i) {
        reduce_h_fused<num_output_tiles, is_partial_tile, split_reader, window_size_hw, in_nblocks_c>(
            in_cb_id, in_scalar_cb_id, i, out_cb_id);
    }
    cb_pop_front(in_scalar_cb_id, 1);
}

}  // namespace NAMESPACE
