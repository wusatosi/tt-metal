// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t cb_in_exist_0 = tt::CB::c_in0;
    constexpr uint32_t cb_in_exist_1 = tt::CB::c_in2;
    constexpr uint32_t cb_in_exist_2 = tt::CB::c_in3;

    constexpr uint32_t cb_out_exist_0 = tt::CB::c_out0;
    constexpr uint32_t cb_out_exist_1 = tt::CB::c_out1;

    constexpr uint32_t cb_in_not_exist = tt::CB::c_in1;
    constexpr uint32_t cb_out_not_exist = tt::CB::c_out2;

    // Part 1: Original bug was found with this combination of codes but I couldn't reproduce it this way
    // The original codes below are some if-else to handle different optional output_tensor with optional_input_tensor
    // (for backward)

    // binary_op_init_common(cb_in_exist_0, cb_in_not_exist, cb_out_exist_0);
    // binary_op_init_common(cb_in_exist_0, cb_in_exist_1, cb_out_exist_0);
    // binary_op_init_common(cb_in_exist_0, cb_in_exist_2, cb_out_exist_0);
    // unary_op_init_common(cb_in_exist_0, cb_out_exist_0);
    // binary_op_init_common(cb_in_exist_0, cb_in_not_exist, cb_out_exist_1);
    // binary_op_init_common(cb_in_exist_0, cb_in_exist_1, cb_out_exist_1);
    // binary_op_init_common(cb_in_exist_0, cb_in_exist_2, cb_out_exist_1);
    // unary_op_init_common(cb_in_exist_0, cb_out_exist_1);

    // Part 2: Reproduce the same compilation bug but with different code:
    binary_op_init_common(cb_in_exist_0, cb_in_exist_1, cb_out_exist_0);
    binary_op_init_common(cb_in_exist_0, cb_in_exist_1, cb_out_not_exist);
    binary_op_init_common(cb_in_exist_0, cb_in_not_exist, cb_out_exist_0);
    binary_op_init_common(cb_in_not_exist, cb_in_exist_1, cb_out_exist_0);
    binary_op_init_common(cb_in_exist_0, cb_in_not_exist, cb_out_not_exist);
    binary_op_init_common(cb_in_not_exist, cb_in_not_exist, cb_out_exist_0);
    binary_op_init_common(cb_in_not_exist, cb_in_exist_1, cb_out_not_exist);
    unary_op_init_common(cb_in_exist_0, cb_out_exist_0);
    unary_op_init_common(cb_in_not_exist, cb_out_exist_0);
    unary_op_init_common(cb_in_exist_0, cb_out_not_exist);
    unary_op_init_common(cb_in_not_exist, cb_out_not_exist);
}
}  // namespace NAMESPACE
