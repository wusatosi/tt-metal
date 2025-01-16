// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "mod_div_lib.h"

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
// #include "tools/profiler/kernel_profiler.hpp"
#include "debug/dprint.h"

namespace NAMESPACE {

FORCE_INLINE void reload_from_cb_to_dst(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t mm_partials_cb_id,
    bool in1_transpose_tile,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_w,
    uint32_t out_subblock_h,
    uint32_t in0_block_w) {
    // Reconfigure input
    copy_tile_to_dst_init_short_with_dt(in1_cb_id, mm_partials_cb_id);
    cb_wait_front(mm_partials_cb_id, out_subblock_num_tiles);

    uint32_t start_dst_index = 0;
    uint32_t start_tile_index = 0;
    copy_block_matmul_partials(mm_partials_cb_id, start_tile_index, start_dst_index, out_subblock_num_tiles);

    cb_pop_front(mm_partials_cb_id, out_subblock_num_tiles);
    // Reconfigure srcA back
    mm_block_init_short_with_dt(
        in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
}

void MAIN {
    // Runtime args
    uint32_t rt_args_idx = 0;
    uint32_t ring_idx = get_arg_val<uint32_t>(rt_args_idx++);
    bool master_reducer = (bool)get_arg_val<uint32_t>(rt_args_idx++);

    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);        // inner block size in tiles
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);  // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles =
        get_compile_time_arg_val(2);  // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    constexpr uint32_t in1_num_subblocks =
        get_compile_time_arg_val(4);  // outer column block size (in inner column blocks)
    constexpr uint32_t in1_block_num_tiles =
        get_compile_time_arg_val(5);                                  // out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_per_core_w = get_compile_time_arg_val(6);  // out_subblock_w*in1_num_subblocks
    constexpr uint32_t num_blocks = get_compile_time_arg_val(7);      // outer inner dim (in inner dim blocks)
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(8);  // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(9);  // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(10);  // out_subblock_h * out_subblock_w;
    constexpr uint32_t batch = get_compile_time_arg_val(11);                   // batch dim
    constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(12);     // number of tiles in out_block
    constexpr bool untilize_out = get_compile_time_arg_val(13);                // untilize output
    constexpr uint32_t num_reducer_partials = get_compile_time_arg_val(14);

    constexpr uint32_t out_block_w = out_subblock_w * in1_num_subblocks;
    constexpr uint32_t in1_width_in_tiles = in1_per_core_w * batch;

    constexpr uint32_t in0_cb_id = tt::CB::c_in0;
    constexpr uint32_t in1_cb_id = tt::CB::c_in1;
    constexpr uint32_t in2_cb_id = tt::CB::c_in2;
    constexpr uint32_t out_cb_id = tt::CB::c_in3;
    constexpr uint32_t mm_partials_cb_id = tt::CB::c_in4;
    constexpr uint32_t reducer_cb_id = tt::CB::c_in5;
    constexpr uint32_t partial_cb_id = tt::CB::c_in6;

    const uint32_t mm_out_cb_id = num_reducer_partials > 1 ? partial_cb_id : out_cb_id;
    // The only case where we don't want to pack_out is when the master
    // reducer partial can stay in dst and can be accumulated on top of
    uint32_t max_dst_tiles = 4;  // TODO: Make this dynamic
    const bool pack_out = num_reducer_partials == 1 || num_reducer_partials % 2 == 0 || !master_reducer ||
                          out_block_num_tiles > max_dst_tiles;

#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif

#ifdef IN1_TRANSPOSE_TILE
    constexpr uint32_t in1_transpose_tile = true;
#else
    constexpr uint32_t in1_transpose_tile = false;
#endif

    constexpr bool spill = num_blocks > 1 && (out_block_num_tiles / out_subblock_num_tiles) > 1;

    mm_block_init(
        in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);

    // Wait for the entire local shard
    cb_wait_front(in1_cb_id, in1_block_num_tiles * num_blocks * batch);
    // PACK(DPRINT << "in1 " << "\n" << TSLICE(in1_cb_id, 0, SliceRange::hw041()) << ENDL());
    for (uint32_t b = 0; b < batch; b++) {
        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;
        // DeviceZoneScopedN("batch")
#ifdef PACK_RELU
        // for each batch we start we relu disabled so that intermediate results are not relu'd
        if constexpr (batch > 1) {
            PACK((llk_pack_relu_config(ReluType::NO_RELU)));
        }
#endif

        if constexpr (batch > 1) {
            PACK((pack_reconfig_data_format(mm_partials_cb_id)));
        }

        // PACK(DPRINT << "out_block_num_tiles * b: " << out_block_num_tiles * b << ENDL());
        for (uint32_t block = 0; block < num_blocks; block++) {
            // DeviceZoneScopedN("block")
            const uint32_t input_cb_id = block == 0 ? in0_cb_id : in2_cb_id;
            bool last_out = block == (num_blocks - 1);
// Configure packer once for pack out without Bias
#if not defined FUSE_BIAS and defined PACK_RELU
            if (last_out) {
                // if last block we pack the final result with relu enabled
                PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
            }
#endif

            cb_wait_front(input_cb_id, in0_block_num_tiles);

            // Offset used to select the correct shard in K and N dimension,
            // Based on the ring_idx and chunk number, respectively
            uint32_t in1_batch_offset = in1_per_core_w * b;
            uint32_t in1_ring_offset = in0_block_w * ((ring_idx + block) % num_blocks);

            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = in1_batch_offset + (in1_ring_offset * in1_width_in_tiles);
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    tile_regs_acquire();
                    if (enable_reload) {
                        reload_from_cb_to_dst(
                            input_cb_id,
                            in1_cb_id,
                            mm_partials_cb_id,
                            in1_transpose_tile,
                            out_subblock_num_tiles,
                            out_subblock_w,
                            out_subblock_h,
                            in0_block_w);
                    }
                    // {DeviceZoneScopedN("start compute")};

#ifndef SKIP_COMPUTE
                    // Compute output sub-block
                    uint32_t dst_index = 0;  // start at 0, each call to matmul_block internally increments dst_index
                    uint32_t in0_index = in0_index_subblock_offset;  // offset into in0 block
                    uint32_t in1_index = in1_index_subblock_offset;  // offset into in1 block
                    // inner dim that we accumualte is the inner dim of in0/in1, which is in0_block_w
                    for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
                        // matmul outer product of (out_subblock_h x out_subblock_w) tiles that fill dst
                        // accumulation is done by iterating matmul_block across inner dim
                        // in0_block_w is passed as innder dim (kt) to matmul_block, interally used to stride in0
                        matmul_block(
                            input_cb_id,
                            in1_cb_id,
                            in0_index,
                            in1_index,
                            dst_index,
                            in1_transpose_tile,
                            out_subblock_w,
                            out_subblock_h,
                            in0_block_w);
                        in0_index++;                  // stride right by 1
                        in1_index += in1_width_in_tiles;  // to stride down by 1 need to stride by in_per_core_w (should
                                                          // be called in1_block_w)
                    }
                    // {DeviceZoneScopedN("end compute")};

#endif  // SKIP_COMPUTE

                    if (last_out) {
// If we fuse bias, we will pack out and run bias + optional sfpu in a separate loop
#if not defined FUSE_BIAS and defined SFPU_OP_INIT_ACTIVATION
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            SFPU_OP_FUNC_ACTIVATION
                        }
#endif

                        // Pack out to output buffer
                        cb_reserve_back(mm_out_cb_id, out_subblock_num_tiles);
                        if (pack_out) {
                            tile_regs_commit();
                            tile_regs_wait();
                        }
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                        PACK((pack_reconfig_data_format(mm_out_cb_id)));
#endif

#ifdef PACKER_L1_ACC

                        PACK((llk_pack_reconfig_l1_acc(0)));
#endif

                        uint32_t start_dst_index = 0;

                        if (pack_out) {
                            matmul_pack_tile(start_dst_index, mm_out_cb_id, out_subblock_num_tiles);
                            tile_regs_release();
                        }
                        cb_push_back(mm_out_cb_id, out_subblock_num_tiles);

                    } else if (spill) {
                        tile_regs_commit();
                        // Wait for tiles in output buffer to be written out since interm and output share memory
                        if (block == 0) {
                            cb_reserve_back(out_cb_id, out_num_tiles_to_wait);
                            out_num_tiles_to_wait += out_subblock_num_tiles;
                        }
                        // Move partial result to interm buffer
                        cb_reserve_back(mm_partials_cb_id, out_subblock_num_tiles);
                        tile_regs_wait();

#ifdef PACKER_L1_ACC
                        if (block == 0) {  // no accumulation for first iteration
                            PACK((llk_pack_reconfig_l1_acc(0)));
                        } else if (block == 1) {
                            PACK((llk_pack_reconfig_l1_acc(1)));
                        }
#endif

                        uint32_t start_dst_index = 0;
                        matmul_pack_tile(start_dst_index, mm_partials_cb_id, out_subblock_num_tiles);

                        tile_regs_release();
                        cb_push_back(mm_partials_cb_id, out_subblock_num_tiles);
                    }

                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

#ifdef PACKER_L1_ACC

            // Last iteration does spill and reload to output buffer
            if (block < num_blocks - 2 && spill) {
                cb_wait_front(mm_partials_cb_id, out_block_num_tiles);
                cb_pop_front(mm_partials_cb_id, out_block_num_tiles);
            }
            if (block == num_blocks - 2 && spill) {
                enable_reload = true;
            }  // reload when last iteration
#else
            if constexpr (spill) {
                enable_reload = true;
            }
#endif

            cb_pop_front(input_cb_id, in0_block_num_tiles);
        }

        if (master_reducer && num_reducer_partials > 1) {
            // DeviceZoneScopedN("reduce");
            /*
            1. wait front on the reducer cb for output_size * reducer_factor
            2. Perform reduction
            3. push back to output cb

            Assumptions:
                - out_block_num_tiles fits fully in dst
            */

            // The only required call from binary_op_init_common, that doesn't lead to packer overflow, is:
            UNPACK((llk_unpack_AB_hw_configure_disaggregated<DST_ACCUM_MODE>(reducer_cb_id, reducer_cb_id)));
            // binary_op_init_common(reducer_cb_id, reducer_cb_id, out_cb_id);

            cb_wait_front(partial_cb_id, out_block_num_tiles);  // wait for the local partial
            cb_wait_front(
                reducer_cb_id, out_block_num_tiles * (num_reducer_partials - 1));  // wait for the remote partials

            uint32_t num_pack_iters = (out_block_num_tiles + max_dst_tiles - 1) / max_dst_tiles;  // ceil division
            uint32_t out_block_tile_cnt = 0;

            for (uint32_t i = 0; i < num_pack_iters; i++) {
                uint32_t num_tiles_to_pack = std::min(max_dst_tiles, out_block_num_tiles - out_block_tile_cnt);

                if (pack_out || i > 0) {
                    tile_regs_acquire();
                }

                uint32_t cb_a = reducer_cb_id;
                uint32_t cb_b = reducer_cb_id;

                uint32_t idx_a = out_block_tile_cnt;
                uint32_t idx_b = out_block_tile_cnt;

                uint32_t p = 0;

                if (pack_out) {  // local partial in CB
                    cb_a = partial_cb_id;
                    if (num_reducer_partials % 2 == 0) {
                        // Add the local partial
                        add_tiles_init(cb_a, cb_b, false);
                        for (uint32_t t = 0; t < num_tiles_to_pack; t++) {
                            add_tiles(cb_a, cb_b, idx_a + t, idx_b + t, t);
                        }
                        p = 1;  // Already done with the first partial in reducer cb
                    } else {
                        // Copy to dst
                        copy_tile_to_dst_init_short(cb_a);
                        for (uint32_t t = 0; t < num_tiles_to_pack; t++) {
                            copy_tile(cb_a, idx_a + t, t);
                        }
                    }
                    cb_a = reducer_cb_id;
                }

                if (num_reducer_partials > 2) {  // More than 2 partials
                    add_tiles_init(cb_a, cb_b, true);
                }
                for (; p < num_reducer_partials - 1; p += 2) {
                    uint32_t partial_idx_a = idx_a + out_block_num_tiles * p;
                    uint32_t partial_idx_b = idx_b + out_block_num_tiles * (p + 1);

                    for (uint32_t t = 0; t < num_tiles_to_pack; t++) {
                        add_tiles(cb_a, cb_b, partial_idx_a + t, partial_idx_b + t, t);
                    }
                }
                tile_regs_commit();
                cb_reserve_back(out_cb_id, num_tiles_to_pack);
                // {DeviceZoneScopedN("commit")};

                // Pack out
                tile_regs_wait();
                // {DeviceZoneScopedN("wait")};
                matmul_pack_tile(0, out_cb_id, num_tiles_to_pack);
                tile_regs_release();
                cb_push_back(out_cb_id, num_tiles_to_pack);
                // {DeviceZoneScopedN("pushed")};

                out_block_tile_cnt += num_tiles_to_pack;
            }
            cb_pop_front(partial_cb_id, out_block_num_tiles);  // Done with the local partial
            // (DPRINT << "b: " << b  << "\n" << TSLICE(out_cb_id, out_block_num_tiles, SliceRange::hw041()) << ENDL());
        }

        if constexpr (batch > 1) {
            // reconfigure init for matmul
            mm_block_init_short(in0_cb_id, in1_cb_id, 0, out_subblock_w, out_subblock_h, in0_block_w);

            // reconfigure unpacker df for src A
            reconfig_data_format_srca(mm_partials_cb_id, in1_cb_id);
        }
    }
    cb_pop_front(in1_cb_id, in1_block_num_tiles * num_blocks * batch);
}
}  // namespace NAMESPACE
