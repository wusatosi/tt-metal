// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "debug/dprint.h"

inline void print_pages(uint32_t l1_addr, uint32_t pagelen, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * pagelen;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < pagelen; ++j, ++ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
    DPRINT << ENDL();
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t x_block_size = get_compile_time_arg_val(0);
    constexpr uint32_t w_block_size = get_compile_time_arg_val(1);

    uint32_t num_blocks = get_arg_val<uint32_t>(0);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_tilize = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    unary_op_init_common(cb_in, cb_out);

    for (uint32_t n = 0; n < num_blocks; n++) {
        // have to global init here, otherwise pcc is bad
        // if n > 0, then some register isn't cleared and the output of tilize_block is garbage
        // unary_op_init_common(cb_in, cb_out);
        // tilize input via unpack and then pack
        tilize_init_short(cb_in, 1);

        cb_wait_front(cb_in, x_block_size);
        // results are correct according to unpacker here
        DPRINT << "block: " << n << ENDL();
        UNPACK(print_pages((uint32_t)(get_local_cb_interface(cb_in).fifo_rd_ptr << 4), x_block_size, 32));
        cb_reserve_back(cb_tilize, 1);

        // removing this line causes the output of tilize_block to be garbage in the second iteration
        tilize_block(cb_in, 1, cb_tilize);  // tilize and pack into cb_tilize

        for (uint8_t i = 0; i < 32; ++i) {
            uint8_t j = i + 1u;
            DPRINT_PACK(
                { DPRINT << TSLICE(cb_tilize, 0, SliceRange{.h0 = i, .h1 = j, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1}); });
        }
        DPRINT_PACK({ DPRINT << ENDL() << ENDL(); });

        // tile slice according to unpacker is garbage after tilize_block in the second iteration, missing an uninit?
        cb_push_back(cb_tilize, 1);
        cb_pop_front(cb_in, x_block_size);

        tilize_uninit(cb_in);

        // transpose input
        cb_wait_front(cb_tilize, 1);
        transpose_wh_init_short(cb_tilize);
        pack_untilize_dst_init_short<1>(cb_out);

        tile_regs_acquire();
        transpose_wh_tile(cb_tilize, 0, 0);  // transpose call
        tile_regs_commit();

        // pack and untilize
        cb_reserve_back(cb_out, w_block_size);

        tile_regs_wait();
        pack_untilize_dst<1>(cb_out);  // pack call
        tile_regs_release();

        cb_push_back(cb_out, w_block_size);

        cb_wait_front(cb_out, w_block_size);
        pack_untilize_uninit();

        cb_pop_front(cb_tilize, 1);
    }
}
}  // namespace NAMESPACE
