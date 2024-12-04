// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "mod_div_lib.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "debug/dprint.h"

#ifdef FUSE_BIAS
#include "compute_kernel_api/bcast.h"
#endif

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#define DEBUG_PRINT 0

#include "debug/dprint_tensix.h"

// #include "debug_macros.h"
    inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
        PACK(( DPRINT << "======" << ENDL() ));
        for (uint16_t r = 0; r < 32; ++ r) {
            SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r+1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
            PACK(( DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL() ));
        }
        PACK(( DPRINT << "++++++" << ENDL() ));
    }
// SliceRange srt = SliceRange{.h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 8, .ws = 1};
// SliceRange srr = SliceRange{.h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1};
// SliceRange srr1 = SliceRange{.h0 = 1, .h1 = 2, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1};
// SliceRange src = SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1};

inline void tilize_in(
    uint32_t in_cb_id,
    uint32_t in_subblock_h,
    uint32_t in_block_w,
    uint32_t in_num_subblocks,
    uint32_t out_cb_id
) {
    tensix_sync();
    //PACK((DPRINT << "Reconfigure for Tilize" << ENDL()));
    tensix_sync();
    tilize_init_short(in_cb_id, in_block_w);
    for (uint32_t in_subblock = 0; in_subblock < in_num_subblocks; ++in_subblock) {
        for (uint32_t h = 0; h < in_subblock_h; ++h) {
            cb_wait_front(in_cb_id, in_block_w);
            cb_reserve_back(out_cb_id, in_block_w);
            tilize_block(in_cb_id, in_block_w, out_cb_id);
            cb_push_back(out_cb_id, in_block_w);
            cb_pop_front(in_cb_id, in_block_w);
        }
    }
    tilize_uninit(in_cb_id);
} // tilize_in()

template <uint32_t out_subblock_w, uint32_t out_block_w, bool is_non_tile_height>
inline void reblock_and_untilize(
    uint32_t num_out_subblocks_in_col,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h,
    uint32_t output_rows_h,
    uint32_t interm_cb_id,
    uint32_t out_cb_id) {
    constexpr bool is_non_tile_height_= is_non_tile_height;
    uint32_t TILE_SIZE = is_non_tile_height_ ? 32 : out_block_w;
    uint32_t num_tiles_in_row_of_subblocks = mulsi3(out_subblock_num_tiles, num_out_subblocks_in_col);
    cb_wait_front(interm_cb_id, num_tiles_in_row_of_subblocks);
    uint32_t within_block_index = 0;
    for (uint32_t h = 0; h < out_subblock_h; h++) {
        uint32_t block_offset = 0;
        uint32_t out_sub_block_rows_h = output_rows_h <= TILE_SIZE ? output_rows_h : TILE_SIZE;
        uint32_t rows_to_copy = is_non_tile_height_ ? out_sub_block_rows_h : 16;
        cb_reserve_back(out_cb_id, out_sub_block_rows_h);
        for (uint32_t n = 0; n < num_out_subblocks_in_col; n++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < out_subblock_w; w++) {
                uint32_t tile_index = block_offset + within_block_index + w;
                copy_tile(interm_cb_id, tile_index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dst<out_subblock_w, out_block_w>(out_cb_id, 1, n, rows_to_copy);

            // dprint_tensix_dest_reg();

            tile_regs_release();
            block_offset += out_subblock_num_tiles;
        }
        cb_push_back(out_cb_id, out_sub_block_rows_h);
        output_rows_h -= out_sub_block_rows_h;
        within_block_index += out_subblock_w;
    }
    cb_pop_front(interm_cb_id, num_tiles_in_row_of_subblocks);
}

namespace NAMESPACE {
void MAIN {

    constexpr uint32_t in0_block_w            = get_compile_time_arg_val(0); // inner block size in tiles
    constexpr uint32_t in0_num_subblocks      = get_compile_time_arg_val(1); // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles    = get_compile_time_arg_val(2); // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    constexpr uint32_t in0_subblock_h         = get_compile_time_arg_val(4);
    constexpr uint32_t in1_num_subblocks      = get_compile_time_arg_val(5); // outer column block size (in inner column blocks)
    constexpr uint32_t in1_block_num_tiles    = get_compile_time_arg_val(6); //out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_block_w            = get_compile_time_arg_val(7); // out_subblock_w*in1_num_subblocks
    // if these are not defined as volatile, it causes code size for TRISC2 to be too large if num_blocks > 1
    constexpr uint32_t in0_num_blocks_h       = get_compile_time_arg_val(8);
    constexpr uint32_t in0_num_blocks_w       = get_compile_time_arg_val(9);
    constexpr uint32_t in1_num_blocks_w       = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_h         = get_compile_time_arg_val(11); // inner row block size in tiles
    constexpr uint32_t out_subblock_w         = get_compile_time_arg_val(12); // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(13); // out_subblock_h * out_subblock_w;
    constexpr bool tilize_in0                 = get_compile_time_arg_val(14);
    constexpr bool untilize_out               = get_compile_time_arg_val(15);
    uint32_t output_rows_h                    = get_compile_time_arg_val(17);
    constexpr bool is_non_tile_height         = get_compile_time_arg_val(18);

    #ifdef WIDTH_SHARDED
    constexpr uint32_t in0_nblocks_w_tilize   = get_compile_time_arg_val(19);
    #endif

    constexpr uint32_t out_block_num_tiles    = in0_num_subblocks * in1_num_subblocks * out_subblock_num_tiles;
    constexpr uint32_t out_block_w = in1_block_w;
    constexpr bool spill = in0_num_blocks_w > 1;

    // CB indices
    constexpr uint32_t in0_cb_id                                = tt::CB::c_in0;
    constexpr uint32_t in1_cb_id                                = tt::CB::c_in1;
    constexpr uint32_t in0_pretilize_cb_id                      = tt::CB::c_in6;
    constexpr uint32_t in0_cb_second_reader_id                  = tt::CB::c_in7;
    constexpr uint32_t matmul_partials_cb                       = tt::CB::c_intermed0;
    constexpr uint32_t tilized_in0_cb_id                        = tt::CB::c_intermed1;
    //constexpr uint32_t untilize_mode_reblock_cb                 = tt::CB::c_intermed2;
    constexpr uint32_t out_cb_id                                = tt::CB::c_out0;

    constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? matmul_partials_cb : out_cb_id;

    constexpr uint32_t mm_out_cb_id = untilize_mode_out_cb_id;
    //DPRINT << "Tilize: " << (uint)tilize_in0 << ENDL();
    constexpr uint32_t mm_in0_cb_id = tilize_in0 ? tilized_in0_cb_id : in0_cb_id;
    constexpr uint32_t in0_num_subblocks_read = in0_num_subblocks;

    mm_block_init(mm_in0_cb_id, in1_cb_id, out_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
    #ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
    #endif
    // in1 num blocks w is the outer loop. Output blocks are computed in col major order.
    for(uint32_t in1_block_w_i = 0; in1_block_w_i < in1_num_blocks_w; ++in1_block_w_i) {

        for(uint32_t in0_block_h_i = 0; in0_block_h_i < in0_num_blocks_h; ++in0_block_h_i) {
            bool enable_reload = false;

            UNPACK( const uint32_t partials_cb_read_ptr = cb_interface[matmul_partials_cb].fifo_rd_ptr );
            PACK( const uint32_t partials_cb_write_ptr = cb_interface[matmul_partials_cb].fifo_wr_ptr );
            uint32_t curr_matmul_out_cb = matmul_partials_cb;
            for(uint32_t in0_block_w_i = 0; in0_block_w_i < in0_num_blocks_w; ++in0_block_w_i) {
                bool last_out = (in0_block_w_i == in0_num_blocks_w - 1);
                if constexpr (tilize_in0) {

                    reconfig_data_format_srca(in1_cb_id, in0_cb_id);

                    tilize_in(in0_cb_id, in0_subblock_h, in0_block_w, in0_num_subblocks_read, tilized_in0_cb_id);


                    mm_block_init_short_with_dt(mm_in0_cb_id, in1_cb_id, /*srca_old_operand=*/in0_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
                }
                cb_wait_front(mm_in0_cb_id, in0_block_num_tiles);
                cb_wait_front(in1_cb_id, in1_block_num_tiles);

                if (last_out) {
                    #ifndef FUSE_BIAS
                    curr_matmul_out_cb = mm_out_cb_id;
                    #endif
                }

                uint32_t in0_index_subblock_offset = 0;
                for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                    uint32_t in1_index_subblock_offset = 0;
                    for (uint32_t in1_subblock_i = 0; in1_subblock_i < in1_num_subblocks; ++in1_subblock_i) {
                        if (enable_reload) {
                            // Reconfigure input
                            copy_tile_to_dst_init_short_with_dt(in1_cb_id, matmul_partials_cb);
                            cb_wait_front(matmul_partials_cb, out_subblock_num_tiles);
                            tile_regs_acquire();

                            uint32_t start_dst_index = 0;
                            uint32_t start_tile_index = 0;
                            copy_block_matmul_partials(matmul_partials_cb, start_tile_index, start_dst_index, out_subblock_num_tiles);

                            cb_pop_front(matmul_partials_cb, out_subblock_num_tiles);
                            // Reconfigure srcA back
                            mm_block_init_short_with_dt(mm_in0_cb_id, in1_cb_id, matmul_partials_cb, false, out_subblock_w, out_subblock_h, in0_block_w);
                        } else {
                            // just acquire
                            tile_regs_acquire();
                        }

                        // Compute output sub-block
                        uint32_t dst_index = 0; // start at 0, each call to matmul_block internally increments dst_index
                        uint32_t in0_index = in0_index_subblock_offset; // offset into in0 block
                        uint32_t in1_index = in1_index_subblock_offset; // offset into in1 block
                        // inner dim that we accumulate is the inner dim of in0/in1, which is in0_block_w
                        for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; inner_dim_idx++) {
                            // matmul outer product of (out_subblock_h x out_subblock_w) tiles that fill dst
                            // accumulation is done by iterating matmul_block across inner dim
                            // in0_block_w is passed as innder dim (kt) to matmul_block, interally used to stride in0
                            matmul_block(mm_in0_cb_id, in1_cb_id, in0_index, in1_index, dst_index, false, out_subblock_w, out_subblock_h, in0_block_w);
                            in0_index ++;  // stride right by 1
                            in1_index += in1_block_w; // to stride down by 1 need to stride by in_per_core_w (should be called in1_block_w)
                        }

                        tile_regs_commit();

                        //dprint_tensix_dest_reg();
                        tensix_sync();
                        if (in0_block_w_i == 0)
                        {
                            //dprint_tensix_dest_reg<true>(0);
                            //DPRINT << "in0 subblocks: " << in0_num_blocks_w << ", in1 subblocks: " << in1_num_blocks_w << ", out_subblock_num_tiles: " << out_subblock_num_tiles << ENDL();
                        }
                        cb_reserve_back(curr_matmul_out_cb, out_subblock_num_tiles);
                        tile_regs_wait();


                        uint32_t start_dst_index = 0;

                        matmul_pack_tile(start_dst_index, curr_matmul_out_cb, out_subblock_num_tiles);

                        tile_regs_release();

                        //SliceRange sr = SliceRange{ .h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 32, .ws = 4 };
                        //UNPACK(( DPRINT << TSLICE(curr_matmul_out_cb, 0, sr) ));

                        tensix_sync();
                        if (in0_block_w_i == 0)
                        {
                            //print_full_tile(curr_matmul_out_cb);
                        }

                        cb_push_back(curr_matmul_out_cb, out_subblock_num_tiles);
                        in1_index_subblock_offset += out_subblock_w;
                    } // for in1_num_subblocks
                    in0_index_subblock_offset += in0_subblock_num_tiles;
                }

                    if constexpr (spill) {
                        //DPRINT << "SPILL" << ENDL();
                        enable_reload = true;

                        if (!last_out) {
                            UNPACK( cb_interface[matmul_partials_cb].fifo_rd_ptr = partials_cb_read_ptr );
                        }
                        if (in0_block_w_i < in0_num_blocks_w - 2) {
                            PACK( cb_interface[matmul_partials_cb].fifo_wr_ptr = partials_cb_write_ptr );
                        }

                     }

                cb_pop_front(mm_in0_cb_id, in0_block_num_tiles);
                cb_pop_front(in1_cb_id, in1_block_num_tiles);
            } // for in0_num_blocks_w
            if constexpr(matmul_partials_cb == mm_out_cb_id) {
                UNPACK( cb_interface[matmul_partials_cb].fifo_rd_ptr = partials_cb_read_ptr );
            }

            if constexpr((in1_num_blocks_w > 1 || in0_num_blocks_h > 1)) {
                reconfig_data_format_srca(matmul_partials_cb, in1_cb_id);

                if constexpr (!tilize_in0) {
                    mm_block_init_short(mm_in0_cb_id, in1_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
                }
            }
        } // for in0_num_blocks_h

    } // for in1_num_blocks_w
} // MAIN
} // NAMESPACE
