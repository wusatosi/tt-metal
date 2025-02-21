#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/recip.h"

namespace NAMESPACE {
void MAIN {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_in2 = tt::CBIndex::c_2;
    constexpr auto cb_in3 = tt::CBIndex::c_3;
    constexpr auto cb_out0 = tt::CBIndex::c_13;
    constexpr auto cb_out1 = tt::CBIndex::c_14;
    constexpr auto cb_out2 = tt::CBIndex::c_15;
    constexpr auto cb_out3 = tt::CBIndex::c_16;

    // (A+B)*C/D

    // +

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    add_tiles_init(cb_in0, cb_in1);

    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);

    tile_regs_acquire();  // math acquire

    add_tiles(cb_in0, cb_in1, 0, 0, 0);

    tile_regs_commit();  // math release

    tile_regs_wait();  // packer acquire

    cb_reserve_back(cb_out0, 1);
    pack_tile(0, cb_out0);
    cb_push_back(cb_out0, 1);

    tile_regs_release();  // packer releases

    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);

    // *

    binary_op_init_common(cb_out0, cb_in2, cb_out1);
    mul_tiles_init(cb_out0, cb_in2);

    cb_wait_front(cb_out0, 1);
    cb_wait_front(cb_in2, 1);

    tile_regs_acquire();

    mul_tiles(cb_out0, cb_in2, 0, 0, 0);

    tile_regs_commit();  // math release

    tile_regs_wait();  // packer acquire

    cb_reserve_back(cb_out1, 1);
    pack_tile(0, cb_out1);
    cb_push_back(cb_out1, 1);

    tile_regs_release();  // packer releases

    cb_pop_front(cb_out0, 1);
    cb_pop_front(cb_in2, 1);

    // /

    // copy tile from cb to dst
    cb_wait_front(cb_in3, 1);
    tile_regs_acquire();
    copy_tile_init(cb_in3);
    copy_tile(cb_in3, 0, 0);
    cb_pop_front(cb_in3, 1);

    recip_tile_init();
    recip_tile(0);
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_out2, 1);
    pack_tile(0, cb_out2);
    cb_push_back(cb_out2, 1);
    tile_regs_release();  // packer releases

    binary_op_init_common(cb_out1, cb_out2, cb_out3);
    mul_tiles_init(cb_out1, cb_out2);

    cb_wait_front(cb_out1, 1);
    cb_wait_front(cb_out2, 1);

    tile_regs_acquire();

    mul_tiles(cb_out1, cb_out2, 0, 0, 0);

    tile_regs_commit();  // math release

    tile_regs_wait();  // packer acquire

    cb_reserve_back(cb_out3, 1);
    pack_tile(0, cb_out3);
    cb_push_back(cb_out3, 1);

    tile_regs_release();  // packer releases

    cb_pop_front(cb_out1, 1);
    cb_pop_front(cb_out2, 1);
}
}  // namespace NAMESPACE
