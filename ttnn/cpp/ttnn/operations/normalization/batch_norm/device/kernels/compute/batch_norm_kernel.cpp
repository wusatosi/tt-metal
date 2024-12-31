// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/eltwise_binary.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

#include <cstdint>

namespace NAMESPACE {

ALWI void apply_rsqrt_to_sum_value(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb0, itile0 + 1);
    cb_wait_front(icb1, itile1 + 1);

    tile_regs_acquire();
    // add values and store them in dst
    add_tiles_init_with_dt(icb0, icb1);
    add_tiles(icb0, icb1, itile0, itile1, dst0);
    // apply rsqrt on dst
    rsqrt_tile_init();
    rsqrt_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    cb_pop_front(icb0, pop0);
    cb_pop_front(icb1, pop1);

    cb_push_back(ocb, onetile);
}

ALWI void process_tile(uint32_t cb_bcast, uint32_t cb_other, uint32_t cb_out, uint32_t freq, uint32_t tile_start) {
    constexpr uint32_t onetile = 1;

    cb_wait_front(cb_bcast, onetile);

    for (uint32_t j = tile_start; j < freq; ++j) {
        cb_wait_front(cb_other, onetile);
        cb_reserve_back(cb_out, onetile);

        tile_regs_acquire();
        sub_tiles(cb_bcast, cb_other, 0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, onetile);
        cb_pop_front(cb_other, onetile);
    }
    cb_pop_front(cb_bcast, onetile);
}

void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    constexpr uint32_t weight_has_value = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t bias_has_value = get_compile_time_arg_val(1) == 1;

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_in0 = tt::CBIndex::c_0;   // input
    constexpr auto cb_in1 = tt::CBIndex::c_1;   // batch_mean
    constexpr auto cb_out0 = tt::CBIndex::c_2;  // output -- > [(input - batch_mean)/(sqrt(batch_var + eps))] * weight
    constexpr auto cb_in2 = tt::CBIndex::c_3;   // batch_var
    constexpr auto cb_eps = tt::CBIndex::c_4;   // batch_var
    constexpr auto cb_den = tt::CBIndex::c_5;   // 1/(sqrt(batch_var + eps))
    constexpr auto cb_num = tt::CBIndex::c_6;   // input - batch_mean
    constexpr auto cb_weight = tt::CBIndex::c_16;  // weight tensor
    constexpr auto cb_tmp_1 = tt::CBIndex::c_17;   // (input - batch_mean)/(sqrt(batch_var + eps))
    constexpr auto cb_bias = tt::CBIndex::c_18;    // bias tensor

    auto cb_bcast = cb_in1;
    auto cb_other = cb_in0;

    binary_op_init_common(cb_bcast, cb_other, cb_out0);

    // input - batch_mean --> need to multiply with -1 her for right result
    sub_tiles_init();
    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;
    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(cb_bcast, cb_other, cb_num, tile_freq, tile_start);
    }
    if (remaining_iterations > 0) {
        process_tile(cb_bcast, cb_other, cb_num, remaining_iterations, tile_start);
    }

    constexpr auto cb_affine_or_out = (weight_has_value || bias_has_value) ? cb_tmp_1 : cb_out0;
    constexpr auto cb_scaled_output = (bias_has_value) ? cb_tmp_1 : cb_out0;
    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        apply_rsqrt_to_sum_value(cb_in2, cb_eps, cb_den, 0, 0, 1, 1);  // 1/(sqrt(batch_var + eps))
        mul_tiles_to_cb(
            cb_num, cb_den, cb_affine_or_out, 0, 0, 1, 1);  // (input - batch_mean)/(sqrt(batch_var + eps)) = result
        if (weight_has_value) {
            mul_tiles_to_cb(
                cb_affine_or_out, cb_weight, cb_scaled_output, tile_id, 0, 0, 1);  // result = result * weight
        }
        if (bias_has_value) {
            add_tiles_to_cb(cb_tmp_1, cb_bias, cb_out0, 0, 0, 1, 1);  // result = result + bias
        }
    }
}
}  // namespace NAMESPACE
