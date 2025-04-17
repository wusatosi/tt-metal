// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "debug/dprint.h"

void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "/" << ENDL();
    for (uint16_t r = 0; r < 32; ++r) {
        DPRINT << (uint)r << " C: "
               << TileSlice(
                      cb_id,
                      tile_id,
                      SliceRange{
                          .h0 = (uint8_t)r,
                          .h1 = (uint8_t)(r + 1),
                          .hs = (uint8_t)1,
                          .w0 = (uint8_t)0,
                          .w1 = (uint8_t)32,
                          .ws = (uint8_t)1},
                      true,
                      untilize)
               << ENDL();
    }
    DPRINT << "*" << ENDL();
}

ALWI void add_op(uint32_t num_of_tiles) {
    // DPRINT << "Doing op for " << num_of_tiles << " tiles" << ENDL();

    const float scalar = get_compile_time_arg_val(2);
    DPRINT << "scalar = " << scalar << ENDL();
    constexpr auto cb_in0 = get_compile_time_arg_val(3);
    constexpr auto cb_in1 = get_compile_time_arg_val(4);

    constexpr auto cb_out0 = get_compile_time_arg_val(5);
    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    // constexpr auto cb_inp0 = get_compile_time_arg_val(6);

    // wait for unpack
    // DPRINT << " A and B to Dest" << ENDL();
    cb_wait_front(cb_in0, num_of_tiles);
    cb_wait_front(cb_in1, num_of_tiles);
    ckernel::tile_regs_acquire();
    // add vill result in Dest register
    ckernel::add_tiles_init(cb_in0, cb_in1);
    for (uint32_t i = 0; i < num_of_tiles; i++) {
        ckernel::add_tiles(cb_in0, cb_in1, i, i, i);
    }
    cb_pop_front(cb_in0, num_of_tiles);
    cb_pop_front(cb_in1, num_of_tiles);

    ckernel::sqrt_tile_init();
    for (uint32_t i = 0; i < num_of_tiles; i++) {
        ckernel::sqrt_tile(i);
    }

    ckernel::binop_with_scalar_tile_init();
    for (uint32_t i = 0; i < num_of_tiles; i++) {
        ckernel::mul_unary_tile(i, scalar);
    }

    ckernel::tile_regs_commit();  // mat jezgro zavrsilo, pusta dest registar

    // move to out
    // DPRINT << "Moving to output" << ENDL();
    cb_reserve_back(cb_out0, num_of_tiles);
    ckernel::tile_regs_wait();  // ceka packer jezgro i uzima dest
    for (uint32_t i = 0; i < num_of_tiles; i++) {
        ckernel::pack_tile(i, cb_out0, i);
    }
    ckernel::tile_regs_release();  // reg je pusten od strane packera
    cb_push_back(cb_out0, num_of_tiles);
    // print_full_tile(cb_out0);
}
namespace NAMESPACE {
void MAIN {
    DPRINT << "S" << ENDL();
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_size = get_compile_time_arg_val(1);

    uint32_t full_blocks = per_core_tile_cnt / per_core_block_size;
    uint32_t remainder = per_core_tile_cnt % per_core_block_size;

    // DPRINT << "Compute params:" << ENDL() << "per_core_tile_cnt: " << per_core_tile_cnt << ENDL()
    //        << "per_core_block_size: " << per_core_block_size << ENDL() << "full_blocks: " << full_blocks << ENDL()
    //        << "remainder: " << remainder << ENDL();

    // for (uint32_t i = 0; i < full_blocks; i++) {
    // qrt_add_mul(per_core_block_size);
    add_op(remainder);

    //}
    // cb_wait_front(cb_in0, num_of_tiles);
    // sqrt_add_mul(remainder);

    DPRINT << "E" << ENDL();
}
}  // namespace NAMESPACE
