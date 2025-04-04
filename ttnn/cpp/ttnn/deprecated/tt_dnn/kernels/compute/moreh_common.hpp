/*
 * SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#ifndef REDUCE_OP
#define REDUCE_OP PoolType::SUM
#endif

#ifndef REDUCE_DIM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#endif

#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"

// Deprecated
ALWI void ACQ() { ckernel::acquire_dst(); }
ALWI void REL() { ckernel:: release_dst(); }

namespace ckernel {

ALWI void pack_tile_with_dt(uint32_t ifrom_dst, uint32_t icb) {
#if defined FP32_DEST_ACC_EN
    pack_reconfig_data_format(icb);
#endif
    ckernel:: pack_tile(ifrom_dst, icb);
}

ALWI void copy_tile_init_with_dt(uint32_t icb, uint32_t transpose = 0) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format_srca(icb);
#endif
    copy_tile_to_dst_init_short(icb, transpose);
}

ALWI void add_tiles_init_with_dt(uint32_t icb0 = 0, uint32_t icb1 = 1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    add_tiles_init(icb0, icb1);
}

ALWI void add_bcast_rows_init_short_with_dt(uint32_t icb0 = 0, uint32_t icb1 = 1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    add_bcast_rows_init_short(icb0, icb1);
}

ALWI void add_bcast_cols_init_short_with_dt(uint32_t icb0 = 0, uint32_t icb1 = 1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    add_bcast_cols_init_short(icb0, icb1);
}

ALWI void add_bcast_scalar_init_short_with_dt(uint32_t icb0 = 0, uint32_t icb1 = 1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    add_bcast_scalar_init_short(icb0, icb1);
}

ALWI void sub_tiles_init_with_dt(uint32_t icb0 = 0, uint32_t icb1 = 1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    sub_tiles_init(icb0, icb1);
}

ALWI void sub_bcast_rows_init_short_with_dt(uint32_t icb0 = 0, uint32_t icb1 = 1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    MATH((llk_math_eltwise_binary_init<ELWSUB, BroadcastType::ROW, MATH_FIDELITY>()));  // TODO(AP)
    // FIXME: API Update needed in compute kernel?
    UNPACK((llk_unpack_AB_init<BroadcastType::ROW>(icb0, icb1)));
}

ALWI void sub_bcast_cols_init_short_with_dt(uint32_t icb0 = 0, uint32_t icb1 = 1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    sub_bcast_cols_init_short(icb0, icb1);
}

ALWI void sub_tiles_bcast_scalar_init_short_with_dt(uint32_t icb0 = 0, uint32_t icb1 = 1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    sub_tiles_bcast_scalar_init_short(icb0, icb1);
}

ALWI void mul_tiles_init_with_dt(uint32_t icb0 = 0, uint32_t icb1 = 1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    mul_tiles_init(icb0, icb1);
}

ALWI void mul_bcast_rows_init_short_with_dt(uint32_t icb0 = 0, uint32_t icb1 = 1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    mul_bcast_rows_init_short(icb0, icb1);
}

ALWI void mul_bcast_cols_init_short_with_dt(uint32_t icb0 = 0, uint32_t icb1 = 1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    mul_bcast_cols_init_short(icb0, icb1);
}

ALWI void mul_tiles_bcast_scalar_init_short_with_dt(uint32_t icb0 = 0, uint32_t icb1 = 1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    mul_tiles_bcast_scalar_init_short(icb0, icb1);
}

template <bool at_start, PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
ALWI void reduce_init_delta_with_dt(uint32_t ocb = 16, uint32_t icb0 = 0, uint32_t icb1 = 1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    reduce_init_delta<at_start, reduce_type, reduce_dim>(icb0, icb1, ocb);
}

class ArgFetcher {
private:
    int arg_idx = 0;

public:
    template <typename T>
    T get_next_arg_val() {
        return get_arg_val<T>(arg_idx++);
    }
};

ALWI void mul_tiles_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ckernel::cb_reserve_back(ocb, onetile);
    ckernel::cb_wait_front(icb0, itile0 + 1);
    ckernel::cb_wait_front(icb1, itile1 + 1);

    ckernel:: tile_regs_acquire();
    mul_tiles_init_with_dt(icb0, icb1);
    mul_tiles(icb0, icb1, itile0, itile1, dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    if (pop0) {
        ckernel::cb_pop_front(icb0, pop0);
    }
    if (pop1) {
        ckernel::cb_pop_front(icb1, pop1);
    }

    ckernel::cb_push_back(ocb, onetile);
}

ALWI void mul_tiles_and_negative_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ckernel::cb_reserve_back(ocb, onetile);
    ckernel::cb_wait_front(icb0, itile0 + 1);
    ckernel::cb_wait_front(icb1, itile1 + 1);

    ckernel:: tile_regs_acquire();
    mul_tiles_init_with_dt(icb0, icb1);
    mul_tiles(icb0, icb1, itile0, itile1, dst0);

    negative_tile_init();
    negative_tile(dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    if (pop0) {
        ckernel::cb_pop_front(icb0, pop0);
    }
    if (pop1) {
        ckernel::cb_pop_front(icb1, pop1);
    }

    ckernel::cb_push_back(ocb, onetile);
}

ALWI void mul_tiles_and_mask_tile_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t maskcb,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t mtile = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1,
    uint32_t popm = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ckernel::cb_reserve_back(ocb, onetile);
    ckernel::cb_wait_front(icb0, itile0 + 1);
    ckernel::cb_wait_front(icb1, itile1 + 1);
    ckernel::cb_wait_front(maskcb, mtile + 1);

    ckernel:: tile_regs_acquire();
    mul_tiles_init_with_dt(icb0, icb1);
    mul_tiles(icb0, icb1, itile0, itile1, dst0);

    constexpr int dst_mask = 1;
    copy_tile_init_with_dt(maskcb);
    ckernel:: copy_tile(maskcb, mtile, dst_mask);

    mask_tile_init();
    mask_tile(dst0, dst_mask);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    if (pop0) {
        ckernel::cb_pop_front(icb0, pop0);
    }
    if (pop1) {
        ckernel::cb_pop_front(icb1, pop1);
    }
    if (popm) {
        ckernel::cb_pop_front(maskcb, popm);
    }

    ckernel::cb_push_back(ocb, onetile);
}

ALWI void mul_tiles_log_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ckernel::cb_reserve_back(ocb, onetile);
    ckernel::cb_wait_front(icb0, itile0 + 1);
    ckernel::cb_wait_front(icb1, itile1 + 1);

    ckernel:: tile_regs_acquire();
    mul_tiles_init_with_dt(icb0, icb1);
    mul_tiles(icb0, icb1, itile0, itile1, dst0);

    log_tile_init();
    log_tile(dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    if (pop0) {
        ckernel::cb_pop_front(icb0, pop0);
    }
    if (pop1) {
        ckernel::cb_pop_front(icb1, pop1);
    }

    ckernel::cb_push_back(ocb, onetile);
}

ALWI void mul_tiles_bcast_rows_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ckernel::cb_reserve_back(ocb, onetile);

    ckernel::cb_wait_front(icb0, itile0 + 1);
    ckernel::cb_wait_front(icb1, itile1 + 1);

    ckernel:: tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    mul_bcast_rows_init_short(icb0, icb1);
    mul_tiles_bcast_rows(icb0, icb1, itile0, itile1, dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    if (pop0) {
        ckernel::cb_pop_front(icb0, pop0);
    }
    if (pop1) {
        ckernel::cb_pop_front(icb1, pop1);
    }

    ckernel::cb_push_back(ocb, onetile);
}

ALWI void mul_tiles_bcast_rows_log_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ckernel::cb_reserve_back(ocb, onetile);

    ckernel::cb_wait_front(icb0, itile0 + 1);
    ckernel::cb_wait_front(icb1, itile1 + 1);

    ckernel:: tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    mul_bcast_rows_init_short(icb0, icb1);
    mul_tiles_bcast_rows(icb0, icb1, itile0, itile1, dst0);

    log_tile_init();
    log_tile(dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    if (pop0) {
        ckernel::cb_pop_front(icb0, pop0);
    }
    if (pop1) {
        ckernel::cb_pop_front(icb1, pop1);
    }

    ckernel::cb_push_back(ocb, onetile);
}

ALWI void mul_tiles_bcast_cols_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ckernel::cb_reserve_back(ocb, onetile);

    ckernel::cb_wait_front(icb0, itile0 + 1);
    ckernel::cb_wait_front(icb1, itile1 + 1);

    ckernel:: tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    mul_bcast_cols_init_short(icb0, icb1);
    mul_tiles_bcast_cols(icb0, icb1, itile0, itile1, dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    if (pop0) {
        ckernel::cb_pop_front(icb0, pop0);
    }
    if (pop1) {
        ckernel::cb_pop_front(icb1, pop1);
    }

    ckernel::cb_push_back(ocb, onetile);
}

ALWI void mul_tiles_bcast_cols_log_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ckernel::cb_reserve_back(ocb, onetile);

    ckernel::cb_wait_front(icb0, itile0 + 1);
    ckernel::cb_wait_front(icb1, itile1 + 1);

    ckernel:: tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    mul_bcast_cols_init_short(icb0, icb1);
    mul_tiles_bcast_cols(icb0, icb1, itile0, itile1, dst0);

    log_tile_init();
    log_tile(dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    if (pop0) {
        ckernel::cb_pop_front(icb0, pop0);
    }
    if (pop1) {
        ckernel::cb_pop_front(icb1, pop1);
    }

    ckernel::cb_push_back(ocb, onetile);
}

ALWI void copy_tile_to_cb(uint32_t icb, uint32_t ocb, uint32_t itile = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ckernel::cb_reserve_back(ocb, onetile);
    ckernel::cb_wait_front(icb, itile + 1);

    ckernel:: tile_regs_acquire();
    copy_tile_init_with_dt(icb);
    ckernel:: copy_tile(icb, itile, dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    if (pop) {
        ckernel::cb_pop_front(icb, pop);
    }
    ckernel::cb_push_back(ocb, onetile);
}

ALWI void sign_tile_to_cb(uint32_t icb, uint32_t ocb, uint32_t itile = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ckernel::cb_reserve_back(ocb, onetile);
    ckernel::cb_wait_front(icb, itile + 1);

    ckernel:: tile_regs_acquire();
    copy_tile_init_with_dt(icb);
    ckernel:: copy_tile(icb, itile, dst0);

    sign_tile_init();
    sign_tile(dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    if (pop) {
        ckernel::cb_pop_front(icb, pop);
    }
    ckernel::cb_push_back(ocb, onetile);
}

ALWI void add_tiles_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ckernel::cb_reserve_back(ocb, onetile);
    ckernel::cb_wait_front(icb0, itile0 + 1);
    ckernel::cb_wait_front(icb1, itile1 + 1);

    ckernel:: tile_regs_acquire();
    add_tiles_init_with_dt(icb0, icb1);
    add_tiles(icb0, icb1, itile0, itile1, dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    if (pop0) {
        ckernel::cb_pop_front(icb0, pop0);
    }
    if (pop1) {
        ckernel::cb_pop_front(icb1, pop1);
    }

    ckernel::cb_push_back(ocb, onetile);
}

ALWI void mask_tile_to_cb(
    uint32_t icb,
    uint32_t maskcb,
    uint32_t ocb,
    uint32_t itile = 0,
    uint32_t mtile = 0,
    uint32_t pop = 1,
    uint32_t popm = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;
    constexpr int dst_mask = 1;

    ckernel::cb_reserve_back(ocb, onetile);
    ckernel::cb_wait_front(icb, itile + 1);
    ckernel::cb_wait_front(maskcb, mtile + 1);

    ckernel:: tile_regs_acquire();
    copy_tile_init_with_dt(icb);
    ckernel:: copy_tile(icb, itile, dst0);

    copy_tile_init_with_dt(maskcb);
    ckernel:: copy_tile(maskcb, mtile, dst_mask);

    mask_tile_init();
    mask_tile(dst0, dst_mask);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    if (pop) {
        ckernel::cb_pop_front(icb, pop);
    }
    if (popm) {
        ckernel::cb_pop_front(maskcb, popm);
    }

    ckernel::cb_push_back(ocb, onetile);
}

template <bool at_start, PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
ALWI void reduce_tile_to_cb(
    uint32_t icb0, uint32_t icb1, uint32_t ocb, uint32_t size, uint32_t pop0 = 1, uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ckernel::cb_reserve_back(ocb, onetile);

    ckernel:: tile_regs_acquire();
    ckernel::cb_wait_front(icb1, onetile);

    reduce_init_delta_with_dt<at_start, reduce_type, reduce_dim>(ocb, icb0, icb1);
    for (uint32_t x = 0; x < size; ++x) {
        ckernel::cb_wait_front(icb0, x + 1);  // must be a cumulative wait for correctness

        constexpr uint32_t bcast_scaler0 = 0;  // 0th index from bcast_scaler CB
        ckernel::reduce_tile<reduce_type, reduce_dim>(icb0, icb1, x, bcast_scaler0, dst0);
    }
    ckernel::reduce_revert_delta(ocb);
    ckernel:: tile_regs_commit();

    if (pop0) {
        ckernel::cb_pop_front(icb0, pop0);
    }
    if (pop1) {
        ckernel::cb_pop_front(icb1, pop1);
    }

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    ckernel::cb_push_back(ocb, onetile);
}

ALWI void sub_tiles_bcast_cols_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ckernel::cb_reserve_back(ocb, onetile);

    ckernel::cb_wait_front(icb0, itile0 + 1);
    ckernel::cb_wait_front(icb1, itile1 + 1);

    ckernel:: tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    sub_bcast_cols_init_short(icb0, icb1);
    sub_tiles_bcast<BroadcastType::COL>(icb0, icb1, itile0, itile1, dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    if (pop0) {
        ckernel::cb_pop_front(icb0, pop0);
    }
    if (pop1) {
        ckernel::cb_pop_front(icb1, pop1);
    }

    ckernel::cb_push_back(ocb, onetile);
}

ALWI void sub_tiles_bcast_rows_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ckernel::cb_reserve_back(ocb, onetile);

    ckernel::cb_wait_front(icb0, itile0 + 1);
    ckernel::cb_wait_front(icb1, itile1 + 1);

    ckernel:: tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    // sub_bcast_rows_init_short();
    {
        MATH((llk_math_eltwise_binary_init<ELWSUB, BroadcastType::ROW, MATH_FIDELITY>()));
        UNPACK((llk_unpack_AB_init<BroadcastType::ROW>(0, 1)));
    }
    sub_tiles_bcast<BroadcastType::ROW>(icb0, icb1, itile0, itile1, dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    if (pop0) {
        ckernel::cb_pop_front(icb0, pop0);
    }
    if (pop1) {
        ckernel::cb_pop_front(icb1, pop1);
    }

    ckernel::cb_push_back(ocb, onetile);
}

ALWI void sub_tiles_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ckernel::cb_reserve_back(ocb, onetile);
    ckernel::cb_wait_front(icb0, itile0 + 1);
    ckernel::cb_wait_front(icb1, itile1 + 1);

    ckernel:: tile_regs_acquire();
    sub_tiles_init_with_dt(icb0, icb1);
    sub_tiles(icb0, icb1, itile0, itile1, dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    if (pop0) {
        ckernel::cb_pop_front(icb0, pop0);
    }
    if (pop1) {
        ckernel::cb_pop_front(icb1, pop1);
    }

    ckernel::cb_push_back(ocb, onetile);
}

ALWI void exp_tile_to_cb(uint32_t icb, uint32_t ocb, uint32_t itile = 0, uint32_t dst = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;

    ckernel::cb_reserve_back(ocb, onetile);
    ckernel::cb_wait_front(icb, itile + 1);

    ckernel:: tile_regs_acquire();
    copy_tile_init_with_dt(icb);
    ckernel:: copy_tile(icb, itile, dst);

    exp_tile_init();
    exp_tile(dst);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst, ocb);
    ckernel::tile_regs_release();

    if (pop) {
        ckernel::cb_pop_front(icb, pop);
    }
    ckernel::cb_push_back(ocb, onetile);
}

ALWI void rexp_tile_to_cb(uint32_t icb, uint32_t ocb, uint32_t itile = 0, uint32_t dst = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;

    ckernel::cb_reserve_back(ocb, onetile);
    ckernel::cb_wait_front(icb, itile + 1);

    ckernel:: tile_regs_acquire();
    copy_tile_init_with_dt(icb);
    ckernel:: copy_tile(icb, itile, dst);

    negative_tile_init();
    negative_tile(dst);

    exp_tile_init();
    exp_tile(dst);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst, ocb);
    ckernel::tile_regs_release();

    if (pop) {
        ckernel::cb_pop_front(icb, pop);
    }
    ckernel::cb_push_back(ocb, onetile);
}

ALWI void exp_tile_and_mask_tile_to_cb(
    uint32_t icb,
    uint32_t maskcb,
    uint32_t ocb,
    uint32_t itile = 0,
    uint32_t mtile = 0,
    uint32_t pop = 1,
    uint32_t popm = 1,
    uint32_t dst = 0,
    uint32_t dst_mask = 1) {
    constexpr uint32_t onetile = 1;

    ckernel::cb_reserve_back(ocb, onetile);
    ckernel::cb_wait_front(icb, itile + 1);
    ckernel::cb_wait_front(maskcb, mtile + 1);

    ckernel:: tile_regs_acquire();
    copy_tile_init_with_dt(icb);
    ckernel:: copy_tile(icb, itile, dst);

    if (pop) {
        ckernel::cb_pop_front(icb, pop);
    }

    exp_tile_init();
    exp_tile(dst);

    copy_tile_init_with_dt(maskcb);
    ckernel:: copy_tile(maskcb, mtile, dst_mask);

    mask_tile_init();
    mask_tile(dst, dst_mask);
    ckernel:: tile_regs_commit();

    if (popm) {
        ckernel::cb_pop_front(maskcb, popm);
    }

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst, ocb);
    ckernel::tile_regs_release();

    ckernel::cb_push_back(ocb, onetile);
}

ALWI void rexp_tile_and_mask_tile_to_cb(
    uint32_t icb,
    uint32_t maskcb,
    uint32_t ocb,
    uint32_t itile = 0,
    uint32_t mtile = 0,
    uint32_t pop = 1,
    uint32_t popm = 1,
    uint32_t dst = 0,
    uint32_t dst_mask = 1) {
    constexpr uint32_t onetile = 1;

    ckernel::cb_reserve_back(ocb, onetile);
    ckernel::cb_wait_front(icb, itile + 1);
    ckernel::cb_wait_front(maskcb, mtile + 1);

    ckernel:: tile_regs_acquire();
    copy_tile_init_with_dt(icb);
    ckernel:: copy_tile(icb, itile, dst);

    if (pop) {
        ckernel::cb_pop_front(icb, pop);
    }

    negative_tile_init();
    negative_tile(dst);

    exp_tile_init();
    exp_tile(dst);

    copy_tile_init_with_dt(maskcb);
    ckernel:: copy_tile(maskcb, mtile, dst_mask);

    mask_tile_init();
    mask_tile(dst, dst_mask);
    ckernel:: tile_regs_commit();

    if (popm) {
        ckernel::cb_pop_front(maskcb, popm);
    }

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst, ocb);
    ckernel::tile_regs_release();

    ckernel::cb_push_back(ocb, onetile);
}

ALWI void recip_tile_to_cb(uint32_t icb, uint32_t ocb, uint32_t itile = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ckernel::cb_reserve_back(ocb, onetile);
    ckernel::cb_wait_front(icb, itile + 1);

    ckernel:: tile_regs_acquire();
    copy_tile_init_with_dt(icb);
    ckernel:: copy_tile(icb, itile, dst0);

    recip_tile_init();
    recip_tile(dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    if (pop) {
        ckernel::cb_pop_front(icb, pop);
    }
    ckernel::cb_push_back(ocb, onetile);
}

ALWI void log_tile_to_cb(uint32_t icb, uint32_t ocb, uint32_t itile = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ckernel::cb_reserve_back(ocb, onetile);
    ckernel::cb_wait_front(icb, itile + 1);

    ckernel:: tile_regs_acquire();
    copy_tile_init_with_dt(icb);
    ckernel:: copy_tile(icb, itile, dst0);

    log_tile_init();
    log_tile(dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    if (pop) {
        ckernel::cb_pop_front(icb, pop);
    }
    ckernel::cb_push_back(ocb, onetile);
}

template <bool at_start, PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
ALWI void reduce_and_recip_tile_to_cb(
    uint32_t icb0, uint32_t icb1, uint32_t ocb, uint32_t size, uint32_t pop0 = 1, uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ckernel::cb_reserve_back(ocb, onetile);
    ckernel::cb_wait_front(icb1, onetile);

    ckernel:: tile_regs_acquire();
    reduce_init_delta_with_dt<at_start, reduce_type, reduce_dim>(ocb, icb0, icb1);
    for (uint32_t x = 0; x < size; ++x) {
        ckernel::cb_wait_front(icb0, x + 1);  // must be a cumulative wait for correctness

        constexpr uint32_t bcast_scaler0 = 0;  // 0th index from bcast_scaler CB
        ckernel::reduce_tile<reduce_type, reduce_dim>(icb0, icb1, x, bcast_scaler0, dst0);
    }
    if (pop0) {
        ckernel::cb_pop_front(icb0, pop0);
    }
    if (pop1) {
        ckernel::cb_pop_front(icb1, pop1);
    }

    ckernel::reduce_revert_delta(ocb);

    recip_tile_init();
    recip_tile(dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    ckernel::cb_push_back(ocb, onetile);
}

template <bool at_start, PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
ALWI void reduce_and_log_tile_to_cb(
    uint32_t icb0, uint32_t icb1, uint32_t ocb, uint32_t size, uint32_t pop0 = 1, uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ckernel::cb_reserve_back(ocb, onetile);
    ckernel::cb_wait_front(icb1, onetile);

    ckernel:: tile_regs_acquire();
    reduce_init_delta_with_dt<at_start, reduce_type, reduce_dim>(ocb, icb0, icb1);
    for (uint32_t x = 0; x < size; ++x) {
        ckernel::cb_wait_front(icb0, x + 1);  // must be a cumulative wait for correctness

        constexpr uint32_t bcast_scaler0 = 0;  // 0th index from bcast_scaler CB
        ckernel::reduce_tile<reduce_type, reduce_dim>(icb0, icb1, x, bcast_scaler0, dst0);
    }
    if (pop0) {
        ckernel::cb_pop_front(icb0, pop0);
    }
    if (pop1) {
        ckernel::cb_pop_front(icb1, pop1);
    }

    ckernel::reduce_revert_delta(ocb);

    log_tile_init();
    log_tile(dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    ckernel::tile_regs_release();

    ckernel::cb_push_back(ocb, onetile);
}

// TODO(seunghwan100): If p is 2 and decimal is 0, we can use sqrt_tile.
ALWI void power_tile_to_cb(
    std::uint8_t cb_x,
    std::uint8_t cb_xpow,
    std::uint8_t cb_logx,
    std::uint8_t cb_decimal,
    std::uint8_t cb_exp_lxmd,
    std::uint8_t cb_correct_xpow,
    uint32_t p,
    bool p_is_negative) {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    // x^p
    ckernel:: tile_regs_acquire();
    ckernel::cb_wait_front(cb_x, onetile);
    ckernel::cb_reserve_back(cb_xpow, onetile);

    copy_tile_init_with_dt(cb_x);
    ckernel:: copy_tile(cb_x, 0, dst0);

    power_tile_init();
    power_tile(dst0, p);

    if (p_is_negative) {
        recip_tile_init();
        recip_tile(dst0);
    }
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, cb_xpow);
    ckernel::tile_regs_release();

    ckernel::cb_push_back(cb_xpow, onetile);
    // We don't pop cb_x here.

    // log(x)
    ckernel:: tile_regs_acquire();
    ckernel::cb_reserve_back(cb_logx, onetile);

    copy_tile_init_with_dt(cb_x);
    ckernel:: copy_tile(cb_x, 0, dst0);

    log_tile_init();
    log_tile(dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, cb_logx);
    ckernel::tile_regs_release();

    ckernel::cb_pop_front(cb_x, onetile);
    ckernel::cb_push_back(cb_logx, onetile);

    // exp(log(x) * decimal)
    ckernel:: tile_regs_acquire();
    ckernel::cb_wait_front(cb_logx, onetile);
    ckernel::cb_reserve_back(cb_exp_lxmd, onetile);

    mul_tiles_init_with_dt(cb_logx, cb_decimal);
    mul_tiles(cb_logx, cb_decimal, 0, 0, dst0);

    exp_tile_init();
    exp_tile(dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, cb_exp_lxmd);
    ckernel::tile_regs_release();

    ckernel::cb_pop_front(cb_logx, onetile);
    ckernel::cb_push_back(cb_exp_lxmd, onetile);

    // x^p * exp(log(x) * decimal)(==(x + decimal)^p)
    ckernel:: tile_regs_acquire();
    ckernel::cb_wait_front(cb_xpow, onetile);
    ckernel::cb_wait_front(cb_exp_lxmd, onetile);
    ckernel::cb_reserve_back(cb_correct_xpow, onetile);

    mul_tiles_init_with_dt(cb_xpow, cb_exp_lxmd);
    mul_tiles(cb_xpow, cb_exp_lxmd, 0, 0, dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, cb_correct_xpow);
    ckernel::tile_regs_release();

    ckernel::cb_pop_front(cb_xpow, onetile);
    ckernel::cb_pop_front(cb_exp_lxmd, onetile);
    ckernel::cb_push_back(cb_correct_xpow, onetile);
}

ALWI void power_tile_with_abs_x_to_cb(
    std::uint8_t cb_x,
    std::uint8_t cb_xpow,
    std::uint8_t cb_logx,
    std::uint8_t cb_decimal,
    std::uint8_t cb_exp_lxmd,
    std::uint8_t cb_correct_xpow,
    uint32_t p,
    bool p_is_negative) {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    // x^p
    ckernel:: tile_regs_acquire();
    ckernel::cb_wait_front(cb_x, onetile);
    ckernel::cb_reserve_back(cb_xpow, onetile);

    copy_tile_init_with_dt(cb_x);
    ckernel:: copy_tile(cb_x, 0, dst0);

    abs_tile_init();
    abs_tile(dst0);

    power_tile_init();
    power_tile(dst0, p);

    if (p_is_negative) {
        recip_tile_init();
        recip_tile(dst0);
    }
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, cb_xpow);
    ckernel::tile_regs_release();

    ckernel::cb_push_back(cb_xpow, onetile);
    // We don't pop cb_x here.

    // log(x)
    ckernel:: tile_regs_acquire();
    ckernel::cb_reserve_back(cb_logx, onetile);

    copy_tile_init_with_dt(cb_x);
    ckernel:: copy_tile(cb_x, 0, dst0);

    abs_tile_init();
    abs_tile(dst0);

    log_tile_init();
    log_tile(dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, cb_logx);
    ckernel::tile_regs_release();

    ckernel::cb_pop_front(cb_x, onetile);
    ckernel::cb_push_back(cb_logx, onetile);

    // exp(log(x) * decimal)
    ckernel:: tile_regs_acquire();
    ckernel::cb_wait_front(cb_logx, onetile);
    ckernel::cb_reserve_back(cb_exp_lxmd, onetile);

    mul_tiles_init_with_dt(cb_logx, cb_decimal);
    mul_tiles(cb_logx, cb_decimal, 0, 0, dst0);

    exp_tile_init();
    exp_tile(dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, cb_exp_lxmd);
    ckernel::tile_regs_release();

    ckernel::cb_pop_front(cb_logx, onetile);
    ckernel::cb_push_back(cb_exp_lxmd, onetile);

    // x^p * exp(log(x) * decimal)(==(x + decimal)^p)
    ckernel:: tile_regs_acquire();
    ckernel::cb_wait_front(cb_xpow, onetile);
    ckernel::cb_wait_front(cb_exp_lxmd, onetile);
    ckernel::cb_reserve_back(cb_correct_xpow, onetile);

    mul_tiles_init_with_dt(cb_xpow, cb_exp_lxmd);
    mul_tiles(cb_xpow, cb_exp_lxmd, 0, 0, dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, cb_correct_xpow);
    ckernel::tile_regs_release();

    ckernel::cb_pop_front(cb_xpow, onetile);
    ckernel::cb_pop_front(cb_exp_lxmd, onetile);
    ckernel::cb_push_back(cb_correct_xpow, onetile);
}

ALWI void power_and_recip_tile_to_cb(
    std::uint8_t cb_x,
    std::uint8_t cb_xpow,
    std::uint8_t cb_logx,
    std::uint8_t cb_decimal,
    std::uint8_t cb_exp_lxmd,
    std::uint8_t cb_recip_xpow,
    uint32_t p,
    bool p_is_negative) {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    // x^p
    ckernel::cb_wait_front(cb_x, onetile);
    ckernel::cb_reserve_back(cb_xpow, onetile);

    ckernel:: tile_regs_acquire();
    copy_tile_init_with_dt(cb_x);
    ckernel:: copy_tile(cb_x, 0, dst0);

    power_tile_init();
    power_tile(dst0, p);

    if (p_is_negative) {
        recip_tile_init();
        recip_tile(dst0);
    }
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, cb_xpow);
    ckernel::tile_regs_release();

    ckernel::cb_push_back(cb_xpow, onetile);
    // We don't pop cb_x here.

    // log(x)
    ckernel::cb_reserve_back(cb_logx, onetile);

    ckernel:: tile_regs_acquire();
    copy_tile_init_with_dt(cb_x);
    ckernel:: copy_tile(cb_x, 0, dst0);

    log_tile_init();
    log_tile(dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, cb_logx);
    ckernel::tile_regs_release();

    ckernel::cb_pop_front(cb_x, onetile);
    ckernel::cb_push_back(cb_logx, onetile);

    // exp(log(x) * decimal)
    ckernel::cb_wait_front(cb_logx, onetile);
    ckernel::cb_reserve_back(cb_exp_lxmd, onetile);

    ckernel:: tile_regs_acquire();
    mul_tiles_init_with_dt(cb_logx, cb_decimal);
    mul_tiles(cb_logx, cb_decimal, 0, 0, dst0);

    exp_tile_init();
    exp_tile(dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, cb_exp_lxmd);
    ckernel::tile_regs_release();

    ckernel::cb_pop_front(cb_logx, onetile);
    ckernel::cb_push_back(cb_exp_lxmd, onetile);

    // 1 / (x^p * exp(log(x) * decimal))(==1 / (x + decimal)^p)
    ckernel::cb_wait_front(cb_xpow, onetile);
    ckernel::cb_wait_front(cb_exp_lxmd, onetile);
    ckernel::cb_reserve_back(cb_recip_xpow, onetile);

    ckernel:: tile_regs_acquire();
    mul_tiles_init_with_dt(cb_xpow, cb_exp_lxmd);
    mul_tiles(cb_xpow, cb_exp_lxmd, 0, 0, dst0);

    recip_tile_init();
    recip_tile(dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, cb_recip_xpow);
    ckernel::tile_regs_release();

    ckernel::cb_pop_front(cb_xpow, onetile);
    ckernel::cb_pop_front(cb_exp_lxmd, onetile);
    ckernel::cb_push_back(cb_recip_xpow, onetile);
}

ALWI void copy_tile_to_dst(uint32_t icb, uint32_t itile = 0, uint32_t dst = 0, bool cb_wait_and_pop = true) {
    constexpr uint32_t onetile = 1;
    if (cb_wait_and_pop) {
        ckernel::cb_wait_front(icb, onetile);
    }
    reconfig_data_format_srca(icb);
    copy_tile_to_dst_init_short(icb);
    ckernel:: copy_tile(icb, itile, dst);
    if (cb_wait_and_pop) {
        ckernel::cb_pop_front(icb, onetile);
    }
}

ALWI void pack_tile_from_dst(uint32_t ocb, uint32_t dst = 0) {
    constexpr uint32_t onetile = 1;
    ckernel::cb_reserve_back(ocb, onetile);
    pack_reconfig_data_format(ocb);
    ckernel:: pack_tile(dst, ocb);
    ckernel::cb_push_back(ocb, onetile);
}

}  // namespace ckernel
