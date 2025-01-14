// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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
#include "debug/waypoint.h"

/******************************************************************************
 *                                                                             *
 *                   Common Functions for Compute Kernels                      *
 *                                                                             *
 ******************************************************************************/

/******************************************************************************
 *                   Generic Compute Functions                                 *
 ******************************************************************************/
void max_block_inplace(uint32_t in0, uint32_t in1, uint32_t num_tiles) {
    WAYPOINT("CCAV");
    // inputs come in full, outputs go out full WAYPOINT("CCAW");
    copy_tile_to_dst_init_short(in0);
    WAYPOINT("CCAX");
    max_tile_init();
    WAYPOINT("CCAY");
    // WAYPOINT("CCAZ");
    constexpr uint32_t dst_reg_0 = 0;
    WAYPOINT("CCBA");
    constexpr uint32_t dst_reg_1 = 1;
    WAYPOINT("CCBB");
    cb_wait_front(in0, num_tiles);
    WAYPOINT("CCBC");
    cb_wait_front(in1, num_tiles);
    WAYPOINT("CCBD");
    for (uint32_t i = 0; i < num_tiles; ++i) {
        WAYPOINT("CCBE");
        acquire_dst();
        WAYPOINT("CCBF");
        copy_tile(in0, 0, dst_reg_0);
        WAYPOINT("CCBG");
        copy_tile(in1, i, dst_reg_1);
        WAYPOINT("CCBH");
        cb_pop_front(in0, 1);
        WAYPOINT("CCBI");
        cb_reserve_back(in0, 1);
        WAYPOINT("CCBJ");
        max_tile(dst_reg_0, dst_reg_1);
        WAYPOINT("CCBK");
        pack_tile(dst_reg_0, in0);
        WAYPOINT("CCBL");
        cb_push_back(in0, 1);
        WAYPOINT("CCBM");
        release_dst();
        WAYPOINT("CCBN");
    }
}

void max_block(uint32_t in0, uint32_t in1, uint32_t out_cb, uint32_t num_tiles) {
    WAYPOINT("CCBQ");
    // inputs come in full, outputs go out full WAYPOINT("CCBR");
    copy_tile_to_dst_init_short(in0);
    WAYPOINT("CCBS");
    max_tile_init();
    WAYPOINT("CCBT");
    // WAYPOINT("CCBU");
    constexpr uint32_t dst_reg_0 = 0;
    WAYPOINT("CCBV");
    constexpr uint32_t dst_reg_1 = 1;
    WAYPOINT("CCBW");
    cb_wait_front(in0, num_tiles);
    WAYPOINT("CCBX");
    cb_wait_front(in1, num_tiles);
    WAYPOINT("CCBY");
    cb_reserve_back(out_cb, num_tiles);
    WAYPOINT("CCBZ");
    for (uint32_t i = 0; i < num_tiles; ++i) {
        WAYPOINT("CCCA");
        acquire_dst();
        WAYPOINT("CCCB");
        copy_tile(in0, i, dst_reg_0);
        WAYPOINT("CCCC");
        copy_tile(in1, i, dst_reg_1);
        WAYPOINT("CCCD");
        max_tile(dst_reg_0, dst_reg_1);
        WAYPOINT("CCCE");
        pack_tile(dst_reg_0, out_cb, i);
        WAYPOINT("CCCF");
        release_dst();
        WAYPOINT("CCCG");
    }
    cb_push_back(out_cb, num_tiles);
    WAYPOINT("CCCI");
}

template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t in0_cb,
    uint32_t scale_cb,
    uint32_t out_cb,
    uint32_t rows,
    uint32_t cols>
void reduce_c() {
    WAYPOINT("CCCK");
    // Precondition: in0_cb has rows*cols produced. in0_cb has tiles in row-major order WAYPOINT("CCCT");
    // Precondition: scale_cb has 1 produced WAYPOINT("CCCU");
    // Precondition: out_cb has rows free WAYPOINT("CCCV");
    // Postcondition: in0_cb has rows*cols produced WAYPOINT("CCCW");
    // Precondition: scale_cb has 1 produced WAYPOINT("CCCX");
    // Postcondition: out_cb has rows produced WAYPOINT("CCCY");
    // WAYPOINT("CCCZ");
    reduce_init_delta<false, pool_type, reduce_dim>(in0_cb, scale_cb, out_cb);
    WAYPOINT("CCDA");
    // WAYPOINT("CCDB");
    const uint32_t num_tiles = rows * cols;
    WAYPOINT("CCDC");
    cb_wait_front(scale_cb, 1);
    WAYPOINT("CCDD");
    cb_wait_front(in0_cb, num_tiles);
    WAYPOINT("CCDE");
    cb_reserve_back(out_cb, rows);
    WAYPOINT("CCDF");
    // WAYPOINT("CCDG");
    constexpr uint32_t reduce_dst_idx = 0;
    WAYPOINT("CCDH");
    // WAYPOINT("CCDI");
    for (uint32_t i = 0; i < rows; i++) {
        WAYPOINT("CCDJ");
        acquire_dst();
        WAYPOINT("CCDK");
        for (uint32_t j = 0; j < cols; j++) {
            WAYPOINT("CCDL");
            reduce_tile<pool_type, reduce_dim>(in0_cb, scale_cb, i * cols + j, 0, reduce_dst_idx);
            WAYPOINT("CCDM");
        }
        // WAYPOINT("CCDO");
        cb_reserve_back(out_cb, 1);
        WAYPOINT("CCDP");
        pack_tile(reduce_dst_idx, out_cb);
        WAYPOINT("CCDQ");
        cb_push_back(out_cb, 1);
        WAYPOINT("CCDR");
        release_dst();
        WAYPOINT("CCDS");
    }
    // WAYPOINT("CCDU");
    reduce_revert_delta<reduce_dim>(out_cb);
    WAYPOINT("CCDV");
}

void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {
    WAYPOINT("CCDX");
    // Precondition: in_cb has num_tiles produced WAYPOINT("CCDY");
    // Postcondition: in_cb has num_tiles produced WAYPOINT("CCDZ");
    copy_tile_to_dst_init_short(in_cb);
    WAYPOINT("CCEA");
    recip_tile_init();
    WAYPOINT("CCEB");
    // WAYPOINT("CCEC");
    cb_wait_front(in_cb, num_tiles);
    WAYPOINT("CCED");
    for (uint32_t i = 0; i < num_tiles; ++i) {
        WAYPOINT("CCEE");
        acquire_dst();
        WAYPOINT("CCEF");
        copy_tile(in_cb, 0, 0);
        WAYPOINT("CCEG");
        cb_pop_front(in_cb, 1);
        WAYPOINT("CCEH");
        recip_tile(0);
        WAYPOINT("CCEI");
        cb_reserve_back(in_cb, 1);
        WAYPOINT("CCEJ");
        pack_tile(0, in_cb);
        WAYPOINT("CCEK");
        cb_push_back(in_cb, 1);
        WAYPOINT("CCEL");
        release_dst();
        WAYPOINT("CCEM");
    }
}

void sub_exp_block_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t rows, uint32_t cols) {
    WAYPOINT("CCEP");
    // Precondition: in0_cb has rows*cols produced WAYPOINT("CCEQ");
    // Precondition: in1_cb has rows produced WAYPOINT("CCER");
    // Postcondition: in0_cb has rows*cols produced WAYPOINT("CCES");
    // Postcondition: in1_cb has rows produced WAYPOINT("CCET");
    // WAYPOINT("CCEU");
    sub_bcast_cols_init_short(in0_cb, in1_cb);
    WAYPOINT("CCEV");
    exp_tile_init<true>();
    WAYPOINT("CCEW");
    cb_wait_front(in0_cb, rows * cols);
    WAYPOINT("CCEX");
    cb_wait_front(in1_cb, rows);
    WAYPOINT("CCEY");
    // WAYPOINT("CCEZ");
    constexpr uint32_t dst_tiles = SUB_EXP_GRANULARITY;
    WAYPOINT("CCFA");
    uint32_t granularity = cols >> LOG2_SUB_EXP_GRANULARITY;
    WAYPOINT("CCFB");
    for (uint32_t i = 0; i < rows; ++i) {
        WAYPOINT("CCFC");
        for (uint32_t u = 0; u < granularity; u++) {
            WAYPOINT("CCFD");
            tile_regs_acquire();
            WAYPOINT("CCFE");
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                WAYPOINT("CCFF");
                sub_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);
                WAYPOINT("CCFG");
                exp_tile<true>(j);
                WAYPOINT("CCFH");
            }
            tile_regs_commit();
            WAYPOINT("CCFJ");
            cb_pop_front(in0_cb, dst_tiles);
            WAYPOINT("CCFK");
            cb_reserve_back(in0_cb, dst_tiles);
            WAYPOINT("CCFL");
            tile_regs_wait();
            WAYPOINT("CCFM");
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                WAYPOINT("CCFN");
                pack_tile(j, in0_cb);
                WAYPOINT("CCFO");
            }
            cb_push_back(in0_cb, dst_tiles);
            WAYPOINT("CCFQ");
            tile_regs_release();
            WAYPOINT("CCFR");
        }
    }
}

void mul_block_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t rows, uint32_t cols) {
    WAYPOINT("CCFV");
    // Precondition: in0_cb has rows*cols produced WAYPOINT("CCFW");
    // Precondition: in1_cb has rows produced WAYPOINT("CCFX");
    // Postcondition: in0_cb has rows*cols produced WAYPOINT("CCFY");
    // Postcondition: in1_cb has rows consumed WAYPOINT("CCFZ");
    // WAYPOINT("CCGA");
    uint32_t num_tiles = rows * cols;
    WAYPOINT("CCGB");
    mul_bcast_cols_init_short(in0_cb, in1_cb);
    WAYPOINT("CCGC");
    cb_wait_front(in0_cb, num_tiles);
    WAYPOINT("CCGD");
    cb_wait_front(in1_cb, rows);
    WAYPOINT("CCGE");
    for (uint32_t i = 0; i < rows; ++i) {
        WAYPOINT("CCGF");
        for (uint32_t j = 0; j < cols; ++j) {
            WAYPOINT("CCGG");
            acquire_dst();
            WAYPOINT("CCGH");
            mul_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
            WAYPOINT("CCGI");
            cb_pop_front(in0_cb, 1);
            WAYPOINT("CCGJ");
            cb_reserve_back(in0_cb, 1);
            WAYPOINT("CCGK");
            pack_tile(0, in0_cb);
            WAYPOINT("CCGL");
            cb_push_back(in0_cb, 1);
            WAYPOINT("CCGM");
            release_dst();
            WAYPOINT("CCGN");
        }
    }
    cb_pop_front(in1_cb, rows);
    WAYPOINT("CCGQ");
}

void mul_block_bcast_scalar_inplace(uint32_t in0_cb, uint32_t in1_scalar_cb, uint32_t num_tiles) {
    WAYPOINT("CCGS");
    // Precondition: in0_cb has num_tiles produced WAYPOINT("CCGT");
    // Precondition: in1_scalar_cb has 1 produced WAYPOINT("CCGU");
    // Postcondition: in0_cb has num_tiles produced WAYPOINT("CCGV");
    // Postcondition: in1_scalar_cb has 1 produced WAYPOINT("CCGW");
    // WAYPOINT("CCGX");
    constexpr uint32_t dst_tiles = MUL_BCAST_GRANULARITY;
    WAYPOINT("CCGY");
    uint32_t granularity = num_tiles >> LOG2_MUL_BCAST_GRANULARITY;
    WAYPOINT("CCGZ");
    reconfig_data_format(in0_cb, in1_scalar_cb);
    WAYPOINT("CCHA");
    mul_tiles_bcast_scalar_init_short();
    WAYPOINT("CCHB");
    cb_wait_front(in0_cb, num_tiles);
    WAYPOINT("CCHC");
    cb_wait_front(in1_scalar_cb, 1);
    WAYPOINT("CCHD");
    for (uint32_t g = 0; g < granularity; ++g) {
        WAYPOINT("CCHE");
        acquire_dst();
        WAYPOINT("CCHF");
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            WAYPOINT("CCHG");
            mul_tiles_bcast_scalar(in0_cb, in1_scalar_cb, i, 0, i);
            WAYPOINT("CCHH");
        }
        cb_pop_front(in0_cb, dst_tiles);
        WAYPOINT("CCHJ");
        cb_reserve_back(in0_cb, dst_tiles);
        WAYPOINT("CCHK");
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            WAYPOINT("CCHL");
            pack_tile(i, in0_cb);
            WAYPOINT("CCHM");
        }
        cb_push_back(in0_cb, dst_tiles);
        WAYPOINT("CCHO");
        release_dst();
        WAYPOINT("CCHP");
    }
}

template <bool pop_in1>
void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    WAYPOINT("CCHT");
    // Precondition: in0_cb and in1_cb have num_tiles produced WAYPOINT("CCHU");
    // Postcondition: in0_cb has num_tiles produced WAYPOINT("CCHV");
    // Postcondition: in1_cb has num_tiles consumed WAYPOINT("CCHW");
    // WAYPOINT("CCHX");
    add_tiles_init();
    WAYPOINT("CCHY");
    cb_wait_front(in0_cb, num_tiles);
    WAYPOINT("CCHZ");
    cb_wait_front(in1_cb, num_tiles);
    WAYPOINT("CCIA");
    for (uint32_t i = 0; i < num_tiles; i++) {
        WAYPOINT("CCIB");
        acquire_dst();
        WAYPOINT("CCIC");
        add_tiles(in0_cb, in1_cb, 0, i, 0);
        WAYPOINT("CCID");
        cb_pop_front(in0_cb, 1);
        WAYPOINT("CCIE");
        cb_reserve_back(in0_cb, 1);
        WAYPOINT("CCIF");
        pack_tile(0, in0_cb);
        WAYPOINT("CCIG");
        cb_push_back(in0_cb, 1);
        WAYPOINT("CCIH");
        release_dst();
        WAYPOINT("CCII");
    }
    if (pop_in1) {
        WAYPOINT("CCIK");
        cb_pop_front(in1_cb, num_tiles);
        WAYPOINT("CCIL");
    }
}

void add_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    WAYPOINT("CCIO");
    // Precondition: in0_cb and in1_cb have num_tiles produced WAYPOINT("CCIP");
    // Postcondition: in0_cb has num_tiles produced WAYPOINT("CCIQ");
    // Postcondition: in1_cb has num_tiles consumed WAYPOINT("CCIR");
    // WAYPOINT("CCIS");
    add_tiles_init();
    WAYPOINT("CCIT");
    cb_wait_front(in0_cb, num_tiles);
    WAYPOINT("CCIU");
    cb_wait_front(in1_cb, num_tiles);
    WAYPOINT("CCIV");
    cb_reserve_back(out_cb, num_tiles);
    WAYPOINT("CCIW");
    for (uint32_t i = 0; i < num_tiles; i++) {
        WAYPOINT("CCIX");
        acquire_dst();
        WAYPOINT("CCIY");
        add_tiles(in0_cb, in1_cb, i, i, 0);
        WAYPOINT("CCIZ");
        pack_tile(0, out_cb, i);
        WAYPOINT("CCJA");
        release_dst();
        WAYPOINT("CCJB");
    }
    cb_push_back(out_cb, num_tiles);
    WAYPOINT("CCJD");
    // WAYPOINT("CCJE");
    cb_pop_front(in0_cb, num_tiles);
    WAYPOINT("CCJF");
    cb_pop_front(in1_cb, num_tiles);
    WAYPOINT("CCJG");
}

void mul_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    WAYPOINT("CCJI");
    // Precondition: in0_cb and in1_cb have num_tiles produced WAYPOINT("CCJJ");
    // Postcondition: in0_cb has num_tiles produced WAYPOINT("CCJK");
    // Postcondition: in1_cb has num_tiles produced WAYPOINT("CCJL");
    // WAYPOINT("CCJM");
    mul_tiles_init();
    WAYPOINT("CCJN");
    cb_wait_front(in0_cb, num_tiles);
    WAYPOINT("CCJO");
    cb_wait_front(in1_cb, num_tiles);
    WAYPOINT("CCJP");
    for (uint32_t i = 0; i < num_tiles; i++) {
        WAYPOINT("CCJQ");
        acquire_dst();
        WAYPOINT("CCJR");
        mul_tiles(in0_cb, in1_cb, 0, i, 0);
        WAYPOINT("CCJS");
        cb_pop_front(in0_cb, 1);
        WAYPOINT("CCJT");
        cb_reserve_back(in0_cb, 1);
        WAYPOINT("CCJU");
        pack_tile(0, in0_cb);
        WAYPOINT("CCJV");
        cb_push_back(in0_cb, 1);
        WAYPOINT("CCJW");
        release_dst();
        WAYPOINT("CCJX");
    }
}

void sub_exp_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    WAYPOINT("CCKA");
    // Precondition: in0_cb and in1_cb have num_tiles produced WAYPOINT("CCKB");
    // Postcondition: out_cb has num_tiles produced WAYPOINT("CCKC");
    // Postcondition: in0_cb and in1_cb has num_tiles produced WAYPOINT("CCKD");
    sub_tiles_init();
    WAYPOINT("CCKE");
    exp_tile_init<EXP_APPROX_MODE>();
    WAYPOINT("CCKF");
    cb_wait_front(in0_cb, num_tiles);
    WAYPOINT("CCKG");
    cb_wait_front(in1_cb, num_tiles);
    WAYPOINT("CCKH");
    cb_reserve_back(out_cb, num_tiles);
    WAYPOINT("CCKI");

    for (uint32_t i = 0; i < num_tiles; i++) {
        WAYPOINT("CCKJ");
        acquire_dst();
        WAYPOINT("CCKK");
        sub_tiles(in0_cb, in1_cb, i, i, 0);
        WAYPOINT("CCKL");
        exp_tile<EXP_APPROX_MODE>(0);
        WAYPOINT("CCKM");
        pack_tile(0, out_cb);
        WAYPOINT("CCKN");
        cb_push_back(out_cb, 1);
        WAYPOINT("CCKO");
        release_dst();
        WAYPOINT("CCKP");
    }
}

void copy_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    WAYPOINT("CCKS");
    // Precondition: in_cb has num_tiles produced WAYPOINT("CCKT");
    // Precondition: out_cb has num_tiles free WAYPOINT("CCKU");
    // Postcondition: in_cb has num_tiles consumed WAYPOINT("CCKV");
    // Postcondition: out_cb has num_tiles produced WAYPOINT("CCKW");
    // WAYPOINT("CCKX");
    copy_tile_to_dst_init_short(in_cb);
    WAYPOINT("CCKY");
    // WAYPOINT("CCKZ");
    cb_wait_front(in_cb, num_tiles);
    WAYPOINT("CCLA");
    cb_reserve_back(out_cb, num_tiles);
    WAYPOINT("CCLB");
    // WAYPOINT("CCLC");
#pragma GCC unroll 0 WAYPOINT("CCLD");
    for (uint32_t i = 0; i < num_tiles; i++) {
        WAYPOINT("CCLE");
        acquire_dst();
        WAYPOINT("CCLF");
        copy_tile(in_cb, i, 0 /*dst*/);
        WAYPOINT("CCLG");
        pack_tile(0, out_cb);
        WAYPOINT("CCLH");
        cb_push_back(out_cb, 1);
        WAYPOINT("CCLI");
        release_dst();
        WAYPOINT("CCLJ");
    }
    cb_pop_front(in_cb, num_tiles);
    WAYPOINT("CCLL");
}

ALWI void cb_matmul_blocks(
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
    WAYPOINT("CCLZ");
    // precondition: in0_cb has M*K produced WAYPOINT("CCMA");
    // preconditino: in1_cb has K*N produced WAYPOINT("CCMC");
    // postcondition: in0_cb is full, in1_cb is empty WAYPOINT("CCMD");
    // postcondition: out_cb has M*N produced WAYPOINT("CCME");
    // WAYPOINT("CCMF");
    mm_block_init_short(
        in0_cb, in1_cb, transpose /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);
    WAYPOINT("CCMH");
    // WAYPOINT("CCMI");
    reconfig_data_format(in1_cb, in0_cb);
    WAYPOINT("CCMJ");
    cb_wait_front(in1_cb, K * N);
    WAYPOINT("CCMK");
    // WAYPOINT("CCML");
    uint32_t output_num_tiles = M * N;
    WAYPOINT("CCMM");
    uint32_t out_subblock_num_tiles = subblock_h * subblock_w;
    WAYPOINT("CCMN");
    uint32_t in0_index_offset = 0;
    WAYPOINT("CCMO");
    // WAYPOINT("CCMP");
    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        WAYPOINT("CCMQ");
        uint32_t in1_index_offset = 0;
        WAYPOINT("CCMR");
        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            WAYPOINT("CCMS");
            tile_regs_acquire();
            WAYPOINT("CCMT");
            // WAYPOINT("CCMU");
            uint32_t dst_index = 0;
            WAYPOINT("CCMV");
            uint32_t in0_index = in0_index_offset;
            WAYPOINT("CCMW");
            uint32_t in1_index = in1_index_offset;
            WAYPOINT("CCMX");
            // WAYPOINT("CCMY");
            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                WAYPOINT("CCMZ");
                matmul_block(
                    in0_cb, in1_cb, in0_index, in1_index, dst_index, transpose, subblock_w, subblock_h, in0_block_w);
                WAYPOINT("CCNB");
                in0_index++;
                WAYPOINT("CCNC");
                in1_index += N;
                WAYPOINT("CCND");
            }
            tile_regs_commit();
            WAYPOINT("CCNF");
            // WAYPOINT("CCNG");
            cb_reserve_back(out_cb, out_subblock_num_tiles);
            WAYPOINT("CCNH");
            tile_regs_wait();
            WAYPOINT("CCNI");
            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                WAYPOINT("CCNJ");
                pack_tile(i, out_cb);
                WAYPOINT("CCNK");
            }
            tile_regs_release();
            WAYPOINT("CCNM");
            cb_push_back(out_cb, out_subblock_num_tiles);
            WAYPOINT("CCNN");
            // in1_index_offset += in1_subblock * subblock_w; WAYPOINT("CCNO");
            // in1_index_offset = (in1_subblock+1) * subblock_w; WAYPOINT("CCNP");
            in1_index_offset += subblock_w;
            WAYPOINT("CCNQ");
        }
        in0_index_offset += subblock_h * in0_block_w;
        WAYPOINT("CCNS");
    }
    cb_pop_front(in1_cb, K * N);
    WAYPOINT("CCNU");
}

/******************************************************************************
 *                   Flash Decode Functions                                    *
 ******************************************************************************/

/**
 * Flash attention computation loop
 *
 * Template Parameters:
 * @tparam St - Total sequence length in tiles
 * @tparam DHt - Head dimension in tiles
 * @tparam Sq_chunk_t - Query chunk size in tiles
 * @tparam Sk_chunk_t - Key chunk size in tiles
 * @tparam qk_in0_block_w - QK matmul block width
 * @tparam qk_subblock_w - QK matmul subblock width
 * @tparam qk_subblock_h - QK matmul subblock height
 * @tparam qk_in0_num_subblocks - QK input0 subblocks
 * @tparam qk_in1_num_subblocks - QK input1 subblocks
 * @tparam qk_num_blocks - QK number of blocks
 * @tparam out_in0_block_w - Output matmul block width
 * @tparam out_subblock_w - Output matmul subblock width
 * @tparam out_subblock_h - Output matmul subblock height
 * @tparam out_in0_num_subblocks - Output input0 subblocks
 * @tparam out_in1_num_subblocks - Output input1 subblocks
 * @tparam out_num_blocks - Output number of blocks
 * @tparam is_causal - Whether to use causal attention (if mask is applied)
 * @tparam use_attention_mask - Whether to use attention mask for non-causal attention
 *
 * Circular Buffer Parameters:
 * @tparam cb_q_in - Query input buffer
 * @tparam cb_k_in - Key input buffer
 * @tparam cb_v_in - Value input buffer
 * @tparam cb_mask_in - Mask input buffer
 * @tparam cb_scale_in - Scale input buffer
 * @tparam cb_identity_scale_in - Identity scale buffer
 * @tparam cb_qk_im - QK intermediate buffer
 * @tparam cb_out_im - Output intermediate buffer
 * @tparam cb_out_accumulate_im - Output accumulate buffer
 * @tparam cb_cur_max - Current max buffer
 * @tparam cb_prev_max - Previous max buffer
 * @tparam cb_cur_sum - Current sum buffer
 * @tparam cb_prev_sum - Previous sum buffer
 * @tparam cb_exp_max_diff - Exp max diff buffer
 * @tparam cb_out_o - Output O buffer
 * @tparam cb_out_m - Output M buffer
 * @tparam cb_out_l - Output L buffer
 *
 * Runtime Parameters:
 * @param k_chunk_start - Start index of key chunk
 * @param k_chunk_end - End index of key chunk
 * @param do_reduce - Whether to perform reduction
 * @param qk_chunk_tiles - Number of QK chunk tiles
 * @param out_chunk_tiles - Number of output chunk tiles
 */
template <
    // Compile-time dimension parameters
    uint32_t St,
    uint32_t DHt,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t qk_chunk_tiles,
    uint32_t out_chunk_tiles,
    // QK matmul block parameters
    uint32_t qk_in0_block_w,
    uint32_t qk_subblock_w,
    uint32_t qk_subblock_h,
    uint32_t qk_in0_num_subblocks,
    uint32_t qk_in1_num_subblocks,
    uint32_t qk_num_blocks,
    // Output matmul block parameters
    uint32_t out_in0_block_w,
    uint32_t out_subblock_w,
    uint32_t out_subblock_h,
    uint32_t out_in0_num_subblocks,
    uint32_t out_in1_num_subblocks,
    uint32_t out_num_blocks,
    // Attention parameters
    bool is_causal,
    bool use_attention_mask,
    // Circular buffer indices
    uint32_t cb_q_in,
    uint32_t cb_k_in,
    uint32_t cb_v_in,
    uint32_t cb_mask_in,
    uint32_t cb_scale_in,
    uint32_t cb_identity_scale_in,
    uint32_t cb_qk_im,
    uint32_t cb_out_im,
    uint32_t cb_out_accumulate_im,
    uint32_t cb_cur_max,
    uint32_t cb_prev_max,
    uint32_t cb_cur_sum,
    uint32_t cb_prev_sum,
    uint32_t cb_exp_max_diff,
    uint32_t cb_out_o,
    uint32_t cb_out_m,
    uint32_t cb_out_l>
void flash_attention_loop(
    // Runtime parameters
    uint32_t k_chunk_start,
    uint32_t k_chunk_end,
    bool do_reduce,
    bool apply_mask_at_last_chunk  // for causal mode, optionally apply mask at the last chunk
) {
    for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
        /* QK = Q_CHUNK @ K_CHUNK */
        reconfig_data_format(cb_q_in, cb_k_in);  // DEBUG
        pack_reconfig_data_format(cb_qk_im);
        cb_matmul_blocks(
            cb_q_in,
            cb_k_in,
            cb_qk_im,
            Sq_chunk_t,
            Sk_chunk_t,
            DHt,
            qk_num_blocks,
            qk_in0_num_subblocks,
            qk_in1_num_subblocks,
            qk_in0_block_w,
            qk_subblock_h,
            qk_subblock_w,
            true /*transpose*/);

        /* QK *= SCALE */
        mul_block_bcast_scalar_inplace(cb_qk_im, cb_scale_in, qk_chunk_tiles);

        if constexpr (is_causal) {
            // For decode, we only apply mask at the last chunk for causal mode
            if (k_chunk == k_chunk_end - 1 && apply_mask_at_last_chunk) {
                /* QK += MASK */
                reconfig_data_format(cb_qk_im, cb_mask_in);
                add_block_inplace<false>(cb_qk_im, cb_mask_in, qk_chunk_tiles);
            }
        } else {
            if constexpr (use_attention_mask) {
                reconfig_data_format(cb_qk_im, cb_mask_in);
                add_block_inplace<true>(cb_qk_im, cb_mask_in, qk_chunk_tiles);
            }
        }

        reconfig_data_format(cb_qk_im, cb_identity_scale_in);
        pack_reconfig_data_format(cb_cur_max);
        reduce_c<
            PoolType::MAX,
            ReduceDim::REDUCE_ROW,
            cb_qk_im,
            cb_identity_scale_in,
            cb_cur_max,
            Sq_chunk_t,
            Sk_chunk_t>();

        if (k_chunk > k_chunk_start) {
            reconfig_data_format(cb_cur_max, cb_prev_max);
            max_block_inplace(cb_cur_max, cb_prev_max, Sq_chunk_t);
        }
        /* QK -= cb_cur_max */
        /* QK = exp(QK)*/
        reconfig_data_format(cb_qk_im, cb_cur_max);
        pack_reconfig_data_format(cb_qk_im);
        sub_exp_block_bcast_cols_inplace(cb_qk_im, cb_cur_max, Sq_chunk_t, Sk_chunk_t);

        /* cb_cur_sum = sum(cb_qk_im, dim=-1) */
        reconfig_data_format(cb_qk_im, cb_identity_scale_in);
        pack_reconfig_data_format(cb_cur_sum);
        reduce_c<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_qk_im,
            cb_identity_scale_in,
            cb_cur_sum,
            Sq_chunk_t,
            Sk_chunk_t>();

        /* OUT_IM = QK @ V_CHUNK */
        reconfig_data_format(cb_qk_im, cb_v_in);  // DEBUG
        pack_reconfig_data_format(cb_out_im);
        cb_matmul_blocks(
            cb_qk_im,
            cb_v_in,
            cb_out_im,
            Sq_chunk_t,
            DHt,
            Sk_chunk_t,
            out_num_blocks,
            out_in0_num_subblocks,
            out_in1_num_subblocks,
            out_in0_block_w,
            out_subblock_h,
            out_subblock_w,
            false /*transpose*/);
        reconfig_data_format_srca(cb_out_im);
        cb_pop_front(cb_qk_im, qk_chunk_tiles);

        /* OUT_ACC += OUT_IM */
        if (k_chunk == k_chunk_start) {
            reconfig_data_format_srca(cb_out_im);
            pack_reconfig_data_format(cb_out_accumulate_im);
            copy_block(cb_out_im, cb_out_accumulate_im, out_chunk_tiles);
        } else {
            reconfig_data_format(cb_prev_max, cb_cur_max);  // DEBUG
            pack_reconfig_data_format(cb_exp_max_diff);
            /* cb_exp_max_diff = torch.exp(cb_prev_max - cb_cur_max) */
            sub_exp_block(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);
            cb_pop_front(cb_prev_max, Sq_chunk_t);

            /* cb_prev_sum *= cb_exp_max_diff */
            mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);

            /* cb_out_accumulate_im *= cb_exp_max_diff */
            reconfig_data_format(cb_out_accumulate_im, cb_exp_max_diff);  // DEBUG
            pack_reconfig_data_format(cb_out_accumulate_im);
            mul_block_bcast_cols_inplace(cb_out_accumulate_im, cb_exp_max_diff, Sq_chunk_t, DHt);

            /* cb_cur_sum += cb_prev_sum */
            reconfig_data_format(cb_cur_sum, cb_prev_sum);  // DEBUG
            pack_reconfig_data_format(cb_cur_sum);
            add_block_inplace<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);

            /* cb_out_accumulate_im += cb_out_im */
            reconfig_data_format(cb_out_accumulate_im, cb_out_im);  // DEBUG
            pack_reconfig_data_format(cb_out_accumulate_im);
            add_block_inplace<true>(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
        }

        if (k_chunk < k_chunk_end - 1 || do_reduce) {
            // Set cb_prev_sum and cb_prev_max
            reconfig_data_format(cb_cur_max, cb_cur_max);  // DEBUG
            pack_reconfig_data_format(cb_prev_max);
            copy_block(cb_cur_max, cb_prev_max, Sq_chunk_t);
            copy_block(cb_cur_sum, cb_prev_sum, Sq_chunk_t);

        } else {
            // Write o, m, l into cb_out
            copy_block(cb_out_accumulate_im, cb_out_o, out_chunk_tiles);
            copy_block(cb_cur_max, cb_out_m, Sq_chunk_t);
            copy_block(cb_cur_sum, cb_out_l, Sq_chunk_t);
        }
    }
}
