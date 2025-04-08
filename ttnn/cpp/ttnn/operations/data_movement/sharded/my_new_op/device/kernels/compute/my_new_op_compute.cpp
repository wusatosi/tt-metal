// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/reg_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    constexpr auto cb_in0 = get_compile_time_arg_val(0);
    constexpr auto cb_in1 = get_compile_time_arg_val(1);
    constexpr auto cb_out0 = get_compile_time_arg_val(2);
    constexpr auto scalar = get_compile_time_arg_val(3);
    constexpr auto num_tiles = get_compile_time_arg_val(4);

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    add_tiles_init(cb_in0, cb_in1);
    ckernel::sqrt_tile_init();
    ckernel::binop_with_scalar_tile_init();

    // wait for a block of tiles in each of input CBs
    cb_wait_front(cb_in0, num_tiles);
    cb_wait_front(cb_in1, num_tiles);

    tile_regs_acquire();  // acquire 8 tile registers
    // add a block of tiles
    for (uint32_t i = 0; i < num_tiles; i++) {
        add_tiles(cb_in0, cb_in1, i, i, i);
        ckernel::sqrt_tile(i);
        ckernel::mul_unary_tile(i, scalar);
    }
    tile_regs_commit();  // signal the packer

    tile_regs_wait();  // packer waits here
    // pack a block of tiles
    pack_tile(0, cb_out0);
    tile_regs_release();  // packer releases

    // pop a block of tiles from each of input CBs
    cb_pop_front(cb_in0, num_tiles);
    cb_pop_front(cb_in1, num_tiles);

    // push a block of tiles to output CBIndex
    cb_push_back(cb_out0, num_tiles);
}
}  // namespace NAMESPACE
