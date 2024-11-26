// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/reshuffle.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"

/*
for (uint8_t i = 0; i < 32; ++i) {
                uint8_t j = i + 1u;
                DPRINT_PACK({ DPRINT  << TSLICE(cb_tilize, 0, SliceRange{ .h0 = i, .h1 = j, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }) << ENDL(); });
            }
*/

void print_tile(uint32_t cb){
        //DPRINT_UNPACK({DPRINT << "*****************************" << ENDL();});
        uint8_t Wt = 1;
        auto ptr = (volatile tt_l1_ptr uint16_t *) (cb_interface[cb].fifo_rd_ptr << 4);
        for (uint8_t i = 0; i < 32; ++i) {
            for (uint8_t j = 0; j < 32; ++j) {
                DPRINT_UNPACK({ DPRINT << BF16(ptr[i * Wt*32 + j]) << " "; });
            }
                DPRINT_UNPACK(DPRINT << " # "<< ENDL() << ENDL() << ENDL() << ENDL());
        }
        //DPRINT_UNPACK({DPRINT << "*****************************" << ENDL();});
}

namespace NAMESPACE {
void MAIN {
    const uint32_t tiles_per_core = get_arg_val<uint32_t>(0); // 1

    //DPRINT << "tiles_per_core: " << tiles_per_core << ENDL(); // 1

    constexpr uint32_t max_tiles_per_core = get_compile_time_arg_val(0); // 1
    constexpr uint32_t input_height = get_compile_time_arg_val(1); // 4

    constexpr uint32_t cb_grad = tt::CBIndex::c_0;
    constexpr uint32_t cb_index = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_intermed = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask = tt::CBIndex::c_24;
    constexpr uint32_t cb_chunk_count_scratch = tt::CBIndex::c_25;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    //DPRINT << cb_out << ENDL();

    unary_op_init_common(cb_grad);
    //print_tile(cb_grad);

    for (uint32_t i = 0; i < input_height; ++i) { // 4 iterations
        cb_wait_front(cb_grad, max_tiles_per_core); // wait till 1 tile arrives in cb_grad

        // Get chunk_count from reader
        volatile uint32_t *chunk_addr_ptr;
        cb_get_tile(cb_chunk_count_scratch, 0, &chunk_addr_ptr); // load from cb_mask to chunk_addr_ptr
        uint32_t chunk_count = chunk_addr_ptr[4];  // Need to shift because read ptr is off by 1 << 4 in BBE
        cb_release_tile(cb_chunk_count_scratch);

        for (uint32_t chunk = 0; chunk < chunk_count; ++chunk) {  // chunk_count = 3
            cb_wait_front(cb_mask, 1);
            // get cb_index pointer from unpack to math thread
            volatile uint *idx_addr_ptr;
            uint32_t tile_to_get = 0;
            cb_get_tile(cb_mask, tile_to_get, &idx_addr_ptr); // load from cb_mask to idx_add_rptr
            uint32_t idx_addr = reinterpret_cast<uint32_t>(idx_addr_ptr);

            cb_wait_front(cb_out_intermed, max_tiles_per_core); // unpack wait tiles

            cb_reserve_back(cb_out, max_tiles_per_core); // pack wait for free tiles

            for (uint32_t hidden_dim = 0; hidden_dim < tiles_per_core; hidden_dim++) { // 1 iteration
                tile_regs_acquire(); // wait for dest available
                tile_regs_wait(); // wait for math done

                copy_tile(cb_grad, hidden_dim, 0); // unpack A; math copy
                copy_tile(cb_out_intermed, hidden_dim, 1);  // unpack A; math copy

                // copied these to dest indexes 0 and 1

                reshuffle_rows_tile_init();
                reshuffle_rows_tile(0, idx_addr); // 0 -> tile index in dest, idx_addr-> address of resulut

                pack_tile(1, cb_out, hidden_dim);  // reshuffle puts output into Tile 1 in DEST hidden_dim is index in cb_out


                tile_regs_commit(); // math dest section done
                tile_regs_release(); // pack section done
            }

            //MATH((DPRINT << ENDL() << ENDL() << ENDL() << "*******************************" << ENDL() << ENDL()));

            cb_push_back(cb_out, max_tiles_per_core); // push tiles to brisc
            cb_pop_front(cb_out_intermed, max_tiles_per_core); //Pop N tiles from the incoming stream

            cb_release_tile(cb_mask); // all threads release
            cb_pop_front(cb_mask, 1);
        }

        cb_pop_front(cb_grad, max_tiles_per_core);
    }
}
}  // namespace NAMESPACE
