// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    const uint32_t tile_offset = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_divisor = tt::CBIndex::c_3;
    constexpr uint32_t cb_output_grad = tt::CBIndex::c_0;
    constexpr uint32_t cb_tmp_weight = tt::CBIndex::c_24;
    constexpr uint32_t cb_tmp1 = tt::CBIndex::c_25;
    constexpr uint32_t cb_tmp2 = tt::CBIndex::c_26;
    constexpr uint32_t cb_input_grad = tt::CBIndex::c_16;

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t onetile = 1;

    ckernel:: init_sfpu(cb_output_grad, tt::CBIndex::c_16);

#if defined(DIVISOR)
    ckernel::cb_wait_front(cb_divisor, onetile);
    ckernel::cb_reserve_back(cb_tmp1, onetile);

    ckernel:: tile_regs_acquire();
    copy_tile_init_with_dt(cb_divisor);
    ckernel:: copy_tile(cb_divisor, 0, dst0);
    recip_tile_init();
    recip_tile(dst0);
    ckernel:: tile_regs_commit();

    ckernel::tile_regs_wait();
    pack_tile_with_dt(dst0, cb_tmp1);
    ckernel::tile_regs_release();

    ckernel::cb_push_back(cb_tmp1, onetile);
#endif

    ckernel::cb_wait_front(cb_output_grad, onetile);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
#if defined(DIVISOR)
        ckernel::cb_wait_front(cb_tmp_weight, onetile);
        ckernel::cb_reserve_back(cb_tmp2, onetile);

        ckernel:: tile_regs_acquire();
        mul_tiles_bcast_scalar_init_short_with_dt(cb_tmp_weight, cb_output_grad);
        mul_tiles_bcast_scalar(cb_tmp_weight, cb_output_grad, 0, 0, dst0);
        negative_tile_init();
        negative_tile(dst0);
        ckernel:: tile_regs_commit();

        ckernel::tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp2);
        ckernel::tile_regs_release();

        ckernel::cb_push_back(cb_tmp2, onetile);
        ckernel::cb_pop_front(cb_tmp_weight, onetile);

        ckernel::cb_reserve_back(cb_input_grad, onetile);
        ckernel::cb_wait_front(cb_tmp2, onetile);
        ckernel::cb_wait_front(cb_tmp1, onetile);

        ckernel:: tile_regs_acquire();
        mul_tiles_bcast_scalar_init_short_with_dt(cb_tmp2, cb_tmp1);
        mul_tiles_bcast_scalar(cb_tmp2, cb_tmp1, 0, 0, dst0);
        ckernel:: tile_regs_commit();

        ckernel::tile_regs_wait();
        pack_tile_with_dt(dst0, cb_input_grad);
        ckernel::tile_regs_release();

        ckernel::cb_push_back(cb_input_grad, onetile);
        ckernel::cb_pop_front(cb_tmp2, onetile);

#else
        ckernel::cb_wait_front(cb_tmp_weight, onetile);

        ckernel::cb_reserve_back(cb_input_grad, onetile);

        ckernel:: tile_regs_acquire();
        mul_tiles_bcast_scalar_init_short_with_dt(cb_tmp_weight, cb_output_grad);
        mul_tiles_bcast_scalar(cb_tmp_weight, cb_output_grad, 0, 0, dst0);
        negative_tile_init();
        negative_tile(dst0);

        ckernel:: tile_regs_commit();

        ckernel::tile_regs_wait();
        pack_tile_with_dt(dst0, cb_input_grad);
        ckernel::tile_regs_release();

        ckernel::cb_push_back(cb_input_grad, onetile);

        ckernel::cb_pop_front(cb_tmp_weight, onetile);
#endif
    }

#if defined(DIVISOR)
    ckernel::cb_pop_front(cb_divisor, onetile);
#endif
}
}  // namespace NAMESPACE
