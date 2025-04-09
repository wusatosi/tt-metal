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
ALWI void sqrt_add_mul(uint32_t num_of_tiles) {
    constexpr auto cb_in0 = get_compile_time_arg_val(2);
    constexpr auto cb_in1 = get_compile_time_arg_val(3);
    constexpr auto cb_scalar0 = get_compile_time_arg_val(4);
    constexpr auto cb_sqrt = get_compile_time_arg_val(5);
    constexpr auto cb_out0 = get_compile_time_arg_val(6);

    // add A and B
    DPRINT << "sqrt(A+B)" << ENDL();
    ckernel::binary_op_init_common(cb_in0, cb_in1, cb_sqrt);

    cb_wait_front(cb_in0, num_of_tiles);
    cb_wait_front(cb_in1, num_of_tiles);
    ckernel::tile_regs_acquire();

    ckernel::add_tiles_init(cb_in0, cb_in1);
    for (uint32_t i = 0; i < num_of_tiles; i++) {
        ckernel::add_tiles(cb_in0, cb_in1, i, i, i);
    }

    ckernel::sqrt_tile_init();

    ckernel::tile_regs_commit();
    cb_pop_front(cb_in0, num_of_tiles);
    cb_pop_front(cb_in1, num_of_tiles);

    // push intermediate result
    DPRINT << "Push sqrt(A+B)" << ENDL();
    cb_reserve_back(cb_sqrt, num_of_tiles);
    ckernel::tile_regs_wait();
    for (uint32_t i = 0; i < num_of_tiles; i++) {
        ckernel::pack_tile(i, cb_sqrt, i);
    }
    ckernel::tile_regs_release();
    cb_push_back(cb_sqrt, num_of_tiles);

    // multiply with scalar
    DPRINT << "sqrt(A+B)*C" << ENDL();
    ckernel::binary_op_init_common(cb_scalar0, cb_sqrt, cb_out0);

    cb_wait_front(cb_scalar0, num_of_tiles);
    cb_wait_front(cb_sqrt, num_of_tiles);
    ckernel::tile_regs_acquire();

    ckernel::mul_tiles_init(cb_scalar0, cb_sqrt);
    for (uint32_t i = 0; i < num_of_tiles; i++) {
        ckernel::mul_tiles(cb_scalar0, cb_sqrt, i, i, i);
    }

    ckernel::tile_regs_commit();
    cb_pop_front(cb_scalar0, num_of_tiles);
    cb_pop_front(cb_sqrt, num_of_tiles);

    // move to out
    DPRINT << "Push output" << ENDL();
    cb_reserve_back(cb_out0, num_of_tiles);
    ckernel::tile_regs_wait();
    for (uint32_t i = 0; i < num_of_tiles; i++) {
        ckernel::pack_tile(i, cb_out0, i);
    }
    ckernel::tile_regs_release();
    cb_push_back(cb_out0, num_of_tiles);
}

void MAIN {
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_size = get_compile_time_arg_val(1);

    uint32_t full_blocks = per_core_tile_cnt / per_core_block_size;
    uint32_t remainder = per_core_tile_cnt % per_core_block_size;

    DPRINT << "Compute params:" << ENDL() << "per_core_tile_cnt: " << per_core_tile_cnt << ENDL()
           << "per_core_block_size: " << per_core_block_size << ENDL() << "full_blocks: " << full_blocks << ENDL()
           << "remainder: " << remainder << ENDL();

    for (uint32_t i = 0; i < full_blocks; i++) {
        sqrt_add_mul(per_core_block_size);
    }
    sqrt_add_mul(remainder);

    DPRINT << "Compute done" << ENDL();
}
}  // namespace NAMESPACE
