// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr uint32_t cb_weight = tt::CBIndex::c_2;
    constexpr uint32_t cb_divisor = tt::CBIndex::c_3;

    constexpr uint32_t cb_tmp_weight = tt::CBIndex::c_24;
    constexpr uint32_t cb_tmp_input = tt::CBIndex::c_25;
    constexpr uint32_t cb_tmp1 = tt::CBIndex::c_26;
    constexpr uint32_t cb_divisor_recip = tt::CBIndex::c_27;  // 1/divisor
    constexpr uint32_t cb_tmp3 = tt::CBIndex::c_28;

    constexpr uint32_t cb_output = tt::CBIndex::c_16;

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t onetile = 1;

    binary_op_init_common(cb_tmp_weight, cb_tmp_input, cb_output);

#if defined(DIVISOR)
    ckernel::cb_wait_front(cb_divisor, onetile);

    ckernel:: tile_regs_acquire();
    copy_tile_init_with_dt(cb_divisor);
    ckernel:: copy_tile(cb_divisor, 0, dst0);
    recip_tile_init();
    recip_tile(dst0);
    ckernel:: tile_regs_commit();

    ckernel::cb_pop_front(cb_divisor, onetile);
    ckernel::cb_reserve_back(cb_divisor_recip, onetile);
    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, cb_divisor_recip);
    ckernel::tile_regs_release();
    ckernel::cb_push_back(cb_divisor_recip, onetile);
#endif

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        ckernel::cb_wait_front(cb_tmp_input, onetile);

        ckernel:: tile_regs_acquire();
        copy_tile_init_with_dt(cb_tmp_input);
        ckernel:: copy_tile(cb_tmp_input, 0, dst0);

        negative_tile_init();
        negative_tile(dst0);
        ckernel:: tile_regs_commit();

        ckernel::cb_pop_front(cb_tmp_input, onetile);

#if defined(WEIGHT)
        ckernel::cb_reserve_back(cb_tmp1, onetile);
        ckernel::tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp1);
        ckernel::tile_regs_release();
        ckernel::cb_push_back(cb_tmp1, onetile);

        // multiply weight
        ckernel::cb_wait_front(cb_tmp1, onetile);
        ckernel::cb_wait_front(cb_tmp_weight, onetile);

        ckernel:: tile_regs_acquire();
        mul_tiles_init_with_dt(cb_tmp1, cb_tmp_weight);
        mul_tiles(cb_tmp1, cb_tmp_weight, 0, 0, dst0);
        ckernel:: tile_regs_commit();

        ckernel::cb_pop_front(cb_tmp_weight, onetile);
        ckernel::cb_pop_front(cb_tmp1, onetile);

#if defined(DIVISOR)
        ckernel::cb_reserve_back(cb_tmp3, onetile);
        ckernel::tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp3);
        ckernel::tile_regs_release();
        ckernel::cb_push_back(cb_tmp3, onetile);

        ckernel::cb_wait_front(cb_tmp3, onetile);
        ckernel::cb_wait_front(cb_divisor_recip, onetile);
        ckernel:: tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
        reconfig_data_format(cb_tmp3, cb_divisor_recip);
#endif
        mul_tiles_bcast_scalar_init_short(cb_tmp3, cb_divisor_recip);
        mul_tiles_bcast_scalar(cb_tmp3, cb_divisor_recip, 0, 0, dst0);
        ckernel:: tile_regs_commit();
        ckernel::cb_pop_front(cb_tmp3, onetile);

        ckernel::cb_reserve_back(cb_output, onetile);
        ckernel::tile_regs_wait();
        pack_tile_with_dt(dst0, cb_output);
        ckernel::tile_regs_release();
        ckernel::cb_push_back(cb_output, onetile);
#else
        ckernel::cb_reserve_back(cb_output, onetile);
        ckernel::tile_regs_wait();
        pack_tile_with_dt(dst0, cb_output);
        ckernel::tile_regs_release();
        ckernel::cb_push_back(cb_output, onetile);
#endif
#else
#if defined(DIVISOR)
        ckernel::cb_reserve_back(cb_tmp1, onetile);
        ckernel::tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp1);
        ckernel::tile_regs_release();
        ckernel::cb_push_back(cb_tmp1, onetile);

        ckernel::cb_wait_front(cb_divisor_recip, onetile);
        ckernel::cb_wait_front(cb_tmp1, onetile);

        ckernel:: tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
        reconfig_data_format(cb_tmp1, cb_divisor_recip);
#endif
        mul_tiles_bcast_scalar_init_short(cb_tmp1, cb_divisor_recip);
        mul_tiles_bcast_scalar(cb_tmp1, cb_divisor_recip, 0, 0, dst0);
        ckernel:: tile_regs_commit();

        ckernel::cb_pop_front(cb_tmp1, onetile);

        ckernel::cb_reserve_back(cb_output, onetile);
        ckernel::tile_regs_wait();
        pack_tile_with_dt(dst0, cb_output);
        ckernel::tile_regs_release();
        ckernel::cb_push_back(cb_output, onetile);
#else
        ckernel::cb_reserve_back(cb_output, onetile);
        ckernel::tile_regs_wait();
        pack_tile_with_dt(dst0, cb_output);
        ckernel::tile_regs_release();
        ckernel::cb_push_back(cb_output, onetile);
#endif
#endif
    }

#if defined(DIVISOR)
    ckernel::cb_pop_front(cb_divisor_recip, onetile);
#endif
}
}  // namespace NAMESPACE
