// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"

template <uint32_t num_tiles>
void max_block_inplace(uint32_t in0, uint32_t in1) {
    DeviceZoneScopedN("MAX_BLOCK_INPLACE");
    // inputs come in full, outputs go out full
    copy_tile_to_dst_init_short(in0);
    max_tile_init();

    constexpr uint32_t dst_reg_0 = 0;
    constexpr uint32_t dst_reg_1 = 1;
    cb_wait_front(in0, num_tiles);
    cb_wait_front(in1, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        copy_tile(in0, i, dst_reg_0);
        copy_tile(in1, i, dst_reg_1);
        max_tile(dst_reg_0, dst_reg_1, (int)VectorMode::C);
        pack_tile(dst_reg_0, in0);
        release_dst();
    }
    cb_pop_front(in0, num_tiles);
    cb_reserve_back(in0, num_tiles);
    cb_push_back(in0, num_tiles);
}

template <PoolType pool_type, ReduceDim reduce_dim, uint32_t in0_cb, uint32_t scale_cb, uint32_t rows, uint32_t cols>
void reduce_cols(uint32_t out_cb) {
    DeviceZoneScopedN("REDUCE_C");
    // Precondition: in0_cb has rows*cols produced. in0_cb has tiles in row-major order
    // Precondition: scale_cb has 1 produced
    // Precondition: out_cb has rows free
    // Postcondition: in0_cb has rows*cols produced
    // Precondition: scale_cb has 1 produced
    // Postcondition: out_cb has rows produced

    reduce_init_delta<false, pool_type, reduce_dim>(in0_cb, scale_cb, out_cb);

    constexpr uint32_t num_tiles = rows * cols;
    cb_wait_front(scale_cb, 1);
    cb_wait_front(in0_cb, num_tiles);
    cb_reserve_back(out_cb, cols);

    constexpr uint32_t reduce_dst_idx = 0;

    for (uint32_t j = 0; j < cols; j++) {
        acquire_dst();
        for (uint32_t i = 0; i < rows; i++) {
            reduce_tile<pool_type, reduce_dim>(in0_cb, scale_cb, i * cols + j, 0, reduce_dst_idx);
        }

        pack_tile(reduce_dst_idx, out_cb);
        release_dst();
    }

    cb_push_back(out_cb, cols);
    reduce_revert_delta<reduce_dim>(out_cb);
}

void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {
    DeviceZoneScopedN("RECIP_BLOCK_INPLACE");
    // Precondition: in_cb has num_tiles produced
    // Postcondition: in_cb has num_tiles produced
    copy_tile_to_dst_init_short(in_cb);
    recip_tile_init();
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(in_cb);

    cb_wait_front(in_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        copy_tile(in_cb, i, 0);
        recip_tile(0, (int)VectorMode::C);
        pack_tile(0, in_cb);
        release_dst();
    }
    cb_pop_front(in_cb, num_tiles);
    cb_reserve_back(in_cb, num_tiles);
    cb_push_back(in_cb, num_tiles);
}

template <uint32_t in0_cb, uint32_t rows, uint32_t cols>
void sub_exp_block_bcast_rows_inplace(uint32_t in1_cb) {
    DeviceZoneScopedN("SUB_EXP_BLOCK_BCAST_COLS_INPLACE");
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows produced
    sub_bcast_cols_init_short(in0_cb, in1_cb);
    exp_tile_init<true, true>();
    cb_wait_front(in0_cb, rows * cols);
    cb_wait_front(in1_cb, cols);

    constexpr uint32_t dst_tiles = SUB_EXP_GRANULARITY;
    constexpr uint32_t granularity = rows >> LOG2_SUB_EXP_GRANULARITY;
    for (uint32_t i = 0; i < cols; ++i) {
        uint32_t in0_index = i;
        uint32_t out_index = i;
        for (uint32_t u = 0; u < granularity; u++) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sub_tiles_bcast_cols(in0_cb, in1_cb, in0_index, i, j);
                exp_tile<true, true>(j);
                in0_index += cols;
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                pack_tile<true>(j, in0_cb, out_index);
                out_index += cols;
            }
            tile_regs_release();
        }
    }
    cb_pop_front(in0_cb, rows * cols);
    cb_reserve_back(in0_cb, rows * cols);
    cb_push_back(in0_cb, rows * cols);
}

template <uint32_t rows, uint32_t cols>
void mul_block_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, bool pack_accumulate = false) {
    DeviceZoneScopedN("MUL_BLOCK_BCAST_COLS_INPLACE");
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Precondition: out_cb has rows*cols produced
    // Postcondition: in0_cb empty
    // Postcondition: in1_cb empty
    // Postcondition: out_cb has rows*cols produced

    constexpr uint32_t num_tiles = rows * cols;
    constexpr uint32_t dst_tiles = DHT_GRANULARITY;
    constexpr uint32_t granularity = cols >> LOG2_DHT_GRANULARITY;
    mul_bcast_cols_init_short(in0_cb, in1_cb);
    PACK((llk_pack_reconfig_l1_acc(pack_accumulate)));
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, rows);
    if (!pack_accumulate) {
        cb_reserve_back(out_cb, num_tiles);
    }
    uint32_t in0_index = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; ++u) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                mul_tiles_bcast_cols(in0_cb, in1_cb, in0_index, i, j);
                in0_index++;
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                pack_tile(j, out_cb);
            }
            tile_regs_release();
        }
    }
    PACK((llk_pack_reconfig_l1_acc(false)));
    cb_pop_front(in1_cb, rows);
    cb_pop_front(in0_cb, num_tiles);
    if (!pack_accumulate) {
        cb_push_back(out_cb, num_tiles);
    } else {
        cb_pop_front(out_cb, num_tiles);
        cb_reserve_back(out_cb, num_tiles);
        cb_push_back(out_cb, num_tiles);
    }
}

template <uint32_t in0_cb, uint32_t in1_scalar_cb, uint32_t num_tiles>
void mul_block_bcast_scalar_inplace() {
    DeviceZoneScopedN("MUL_BLOCK_BCAST_SCALAR_INPLACE");
    // Precondition: in0_cb has num_tiles produced
    // Precondition: in1_scalar_cb has 1 produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_scalar_cb has 1 produced

    constexpr uint32_t dst_tiles = MUL_BCAST_GRANULARITY;
    constexpr uint32_t granularity = num_tiles >> LOG2_MUL_BCAST_GRANULARITY;
    reconfig_data_format(in0_cb, in1_scalar_cb);
    mul_tiles_bcast_scalar_init_short(in0_cb, in1_scalar_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_scalar_cb, 1);
    uint32_t in0_index = 0;
    for (uint32_t g = 0; g < granularity; ++g) {
        acquire_dst();
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            mul_tiles_bcast_scalar(in0_cb, in1_scalar_cb, in0_index, 0, i);
            in0_index++;
        }
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            pack_tile(i, in0_cb);
        }
        release_dst();
    }
    cb_pop_front(in0_cb, num_tiles);
    cb_reserve_back(in0_cb, num_tiles);
    cb_push_back(in0_cb, num_tiles);
}

void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    DeviceZoneScopedN("ADD_BLOCK_INPLACE");
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles consumed

    add_tiles_init(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        add_tiles(in0_cb, in1_cb, i, i, 0);
        pack_tile(0, in0_cb);
        release_dst();
    }

    cb_pop_front(in1_cb, num_tiles);
    cb_pop_front(in0_cb, num_tiles);
    cb_reserve_back(in0_cb, num_tiles);
    cb_push_back(in0_cb, num_tiles);
}

void mul_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    DeviceZoneScopedN("MUL_BLOCK_INPLACE");
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles produced

    mul_tiles_init(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        mul_tiles(in0_cb, in1_cb, 0, i, 0);
        cb_pop_front(in0_cb, 1);
        cb_reserve_back(in0_cb, 1);
        pack_tile(0, in0_cb);
        cb_push_back(in0_cb, 1);
        release_dst();
    }
}

void sub_exp_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    DeviceZoneScopedN("SUB_EXP_BLOCK");
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: out_cb has num_tiles produced
    // Postcondition: in0_cb and in1_cb has num_tiles produced

    sub_tiles_init(in0_cb, in1_cb);
    exp_tile_init<true, false>();
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();

        sub_tiles(in0_cb, in1_cb, i, i, 0);

        exp_tile<true, false>(0, (int)VectorMode::C);

        pack_tile(0, out_cb);

        cb_push_back(out_cb, 1);
        release_dst();
    }
}

void copy_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    DeviceZoneScopedN("COPY_BLOCK");
    // Precondition: in_cb has num_tiles produced
    // Precondition: out_cb has num_tiles free
    // Postcondition: in_cb has num_tiles consumed
    // Postcondition: out_cb has num_tiles produced
    copy_tile_to_dst_init_short(in_cb);
    cb_wait_front(in_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
#pragma GCC unroll 0
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        copy_tile(in_cb, i, 0 /*dst*/);
        pack_tile(0, out_cb);
        cb_push_back(out_cb, 1);
        release_dst();
    }
    cb_pop_front(in_cb, num_tiles);
}

void matmul_blocks(
    const uint32_t& in0_cb,
    const uint32_t& in1_cb,
    const uint32_t& out_cb,
    const uint32_t& M,
    const uint32_t& N,
    const uint32_t& K,
    const uint32_t& num_blocks,
    const uint32_t& in0_num_subblocks,
    const uint32_t& in1_num_subblocks,
    const uint32_t& in0_block_w,
    const uint32_t& subblock_h,
    const uint32_t& subblock_w,
    const bool& transpose) {
    DeviceZoneScopedN("MATMUL_BLOCKS");
    // precondition: in0_cb has M*K produced
    // preconditino: in1_cb has K*N produced
    // postcondition: in0_cb is full, in1_cb is empty
    // postcondition: out_cb has M*N produced
    mm_block_init_short(
        in0_cb, in1_cb, transpose /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);

    uint32_t output_num_tiles = M * N;
    uint32_t out_subblock_num_tiles = subblock_h * subblock_w;
    uint32_t in0_index_offset = 0;

    reconfig_data_format(in1_cb, in0_cb);
    cb_wait_front(in1_cb, K * N);
    cb_reserve_back(out_cb, output_num_tiles);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        uint32_t in1_index_offset = 0;
        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                matmul_block(
                    in0_cb, in1_cb, in0_index, in1_index, dst_index, transpose, subblock_w, subblock_h, in0_block_w);
                in0_index++;
                in1_index += N;
            }
            tile_regs_commit();

            tile_regs_wait();
            uint32_t dst_idx = 0;
            uint32_t out_col_offset = in1_subblock * subblock_w;
            for (uint32_t r = 0; r < subblock_h; r++) {
                uint32_t out_row_offset = (r + subblock_h * in0_subblock) * N;
                for (uint32_t c = 0; c < subblock_w; c++) {
                    pack_tile<true>(dst_idx, out_cb, out_row_offset + out_col_offset + c);
                    dst_idx++;
                }
            }
            tile_regs_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * in0_block_w;
    }
    cb_pop_front(in1_cb, K * N);
    cb_push_back(out_cb, output_num_tiles);
}

template <uint32_t in0_cb, uint32_t in1_cb, uint32_t K, uint32_t N>
void matmul_reduce_cols(const uint32_t& out_cb) {
    DeviceZoneScopedN("MATMUL_REDUCE");
    /**
     * This function takes an input of shape (K x N) and does reduce_sum on K to produce a 1xN output.
     * in0 is the CB to reduce, in1 is the single tile CB with a row of ones.
     *
     * The first stage does binary addition to reduce (K x N) to (1 x N) tiles. This then follows to a matmul
     * to reduce each column of a single tile to a single value.
     *
     */
    // precondition: in0_cb has M*K produced
    // preconditino: in1_cb has K*N produced
    // postcondition: in0_cb is full, in1_cb is empty
    // postcondition: out_cb has M*N produced

    constexpr uint32_t M = 1;  // Result of reduce is 1 column
    constexpr uint32_t in0_block_w = M;
    constexpr uint32_t subblock_h = M;
    // Reuse the Sq_chunk_t granularity chosen for sub_exp_block
    constexpr uint32_t subblock_w = STATS_GRANULARITY;
    constexpr uint32_t in1_num_subblocks = N >> LOG2_STATS_GRANULARITY;

    /**
     * Use DST accumulation to add tiles together to get an Mx1 output.
     */
    add_tiles_init(in0_cb, in0_cb, true);
    cb_wait_front(in0_cb, K * N);
    cb_reserve_back(out_cb, N);

    uint32_t in0_base_index = 0;
    for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
        acquire_dst();
        for (uint32_t col = 0; col < subblock_w; col++) {
            uint32_t in0_index = in0_base_index + col;
            for (uint32_t row = 0; row < K; row += 2) {
                add_tiles(in0_cb, in0_cb, in0_index, in0_index + N, col);
                in0_index += 2 * N;
            }
            pack_tile(col, out_cb);
        }
        release_dst();
        in0_base_index += subblock_w;
    }
    cb_push_back(out_cb, N);

    /**
     * Use matmul on 1xN input to reduce rows within tile to produce 1xN output.
     */

    mm_block_init_short(
        in1_cb, out_cb, 0 /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);

    constexpr uint32_t output_num_tiles = M * N;
    constexpr uint32_t out_subblock_num_tiles = subblock_h * subblock_w;

    reconfig_data_format(out_cb, in1_cb);
    cb_wait_front(in1_cb, M);
    cb_wait_front(out_cb, N);

    for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
        tile_regs_acquire();

        uint32_t dst_index = 0;
        uint32_t in0_index = 0;
        uint32_t in1_index = 0;

        matmul_block(in1_cb, out_cb, in0_index, in1_index, dst_index, 0, subblock_w, subblock_h, in0_block_w);

        tile_regs_commit();
        cb_pop_front(out_cb, subblock_w);

        tile_regs_wait();
        for (uint32_t i = 0; i < subblock_w; i++) {
            pack_tile(i, out_cb);
        }
        tile_regs_release();
        cb_push_back(out_cb, subblock_w);
    }
}
