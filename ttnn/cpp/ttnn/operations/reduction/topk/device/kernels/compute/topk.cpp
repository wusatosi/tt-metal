// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "debug/dprint.h"

inline void print_loop(uint32_t count) {
    // UNPACK(DPRINT << "U-LOOP:" << (uint32_t)count << ENDL());
    MATH(DPRINT << "M-LOOP:" << (uint32_t)count << ENDL());
    // PACK(DPRINT << "P-LOOP:" << (uint32_t)count << ENDL());
}

inline void print_full_tile_column0(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    UNPACK(DPRINT << "U=====!" << ENDL());
    MATH(DPRINT << "M=====!" << ENDL());
    PACK(DPRINT << "P=====!" << ENDL());
    for (uint8_t r = 0; r < 32; ++r) {
        SliceRange sr_left = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 1, .ws = 1};
        UNPACK(DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " ");
        MATH(DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " ");
        PACK(DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " ");
    }
    UNPACK(DPRINT << ENDL() << "U+++++!" << ENDL());
    MATH(DPRINT << ENDL() << "M+++++!" << ENDL());
    PACK(DPRINT << ENDL() << "P+++++!" << ENDL());
}

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    UNPACK(DPRINT << "U=====!" << ENDL());
    MATH(DPRINT << "M=====!" << ENDL());
    PACK(DPRINT << "P=====!" << ENDL());
    for (uint8_t r = 0; r < 32; ++r) {
        SliceRange sr_left = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 16, .ws = 1};
        SliceRange sr_right = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 17, .w1 = 32, .ws = 1};
        UNPACK(
            DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " "
                   << TileSlice(cb_id, tile_id, sr_right, true, untilize) << ENDL());
        MATH(
            DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " "
                   << TileSlice(cb_id, tile_id, sr_right, true, untilize) << ENDL());
        PACK(
            DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " "
                   << TileSlice(cb_id, tile_id, sr_right, true, untilize) << ENDL());
    }
    UNPACK(DPRINT << "U+++++!" << ENDL());
    MATH(DPRINT << "M+++++!" << ENDL());
    PACK(DPRINT << "P+++++!" << ENDL());
}

#include "topk_common_funcs.hpp"

// topk llk needs a global variable atm
// this can only be removed once that's fixed
int32_t topk_replay_init = 0;

namespace NAMESPACE {

void transpose_and_pack(uint32_t input_cb_index, uint32_t dest_cb_index, uint32_t total_tiles) {
    reconfig_data_format_srca(input_cb_index);
    transpose_wh_init_short(input_cb_index);
    pack_reconfig_data_format(input_cb_index);

    cb_wait_front(input_cb_index, total_tiles);
    for (uint32_t i = 0; i < total_tiles; ++i) {
        acquire_dst();
        cb_reserve_back(dest_cb_index, 1);
        transpose_wh_tile(input_cb_index, i, 0);
        pack_tile(0, dest_cb_index);
        cb_push_back(dest_cb_index, 1);
        release_dst();
    }
    cb_pop_front(input_cb_index, total_tiles);
}

void process_and_sort_tiles(
    uint32_t input_cb_index,
    uint32_t index_cb_index,
    uint32_t input_transposed_cb_index,
    uint32_t index_transposed_cb_index,
    uint32_t max_size,
    bool& ascending,
    int logk) {
    // copy two new tiles at a time since topk_local_sort can process 2 tiles at a time
    for (uint32_t index = 0; index < max_size; index += 2) {
        cb_wait_front(input_cb_index, 2);
        cb_wait_front(index_cb_index, 2);

        acquire_dst();
        // local sort into k groups

        reconfig_data_format_srca(input_cb_index);
        transpose_wh_init_short(input_cb_index);
        transpose_wh_tile(input_cb_index, 0, 0);
        transpose_wh_tile(input_cb_index, 1, 1);

        reconfig_data_format_srca(index_cb_index);
        transpose_wh_init_short(index_cb_index);
        transpose_wh_tile(index_cb_index, 0, 2);
        transpose_wh_tile(index_cb_index, 1, 3);

        cb_pop_front(input_cb_index, 2);
        cb_pop_front(index_cb_index, 2);

        // llk_topk_sort -> inplace
        // ckernel::topk_local_sort(0, (int)ascending, logk);

        cb_reserve_back(input_transposed_cb_index, 2);
        cb_reserve_back(index_transposed_cb_index, 2);
        // pack value tiles into cb_intermed0
        pack_reconfig_data_format(input_transposed_cb_index);
        pack_tile(0, input_transposed_cb_index);
        pack_tile(1, input_transposed_cb_index);

        // pack index tiles into cb_intermed1
        pack_reconfig_data_format(index_transposed_cb_index);
        pack_tile(2, index_transposed_cb_index);
        pack_tile(3, index_transposed_cb_index);

        release_dst();
        // ascending = switch_dir ? !ascending : ascending;

        cb_push_back(input_transposed_cb_index, 2);
        cb_push_back(index_transposed_cb_index, 2);
    }
}

void MAIN {
    constexpr uint32_t input_val_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t input_ind_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t transposed_val_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t transposed_ind_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t result_prep_val_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t result_prep_ind_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t output_val_cb_index = get_compile_time_arg_val(6);
    constexpr uint32_t output_ind_cb_index = get_compile_time_arg_val(7);
    constexpr uint32_t Ht = get_compile_time_arg_val(8);
    constexpr uint32_t Wt = get_compile_time_arg_val(9);
    constexpr uint32_t K = get_compile_time_arg_val(10);
    constexpr uint32_t logk = get_compile_time_arg_val(11);
    constexpr uint32_t logNk = get_compile_time_arg_val(12);
    constexpr uint32_t largest = get_compile_time_arg_val(13);
    constexpr uint32_t sorted = get_compile_time_arg_val(14);

    // dest indices for where to unpack the tiles for the llk
    // the input goes in index 0,1 and the index goes in index 2,3
    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;
    constexpr uint32_t output_tiles = (K + 31) / 32;
    bool ascending = !largest;
    // init pack, compute and unpack

    ckernel::topk_tile_init();
    transpose_wh_init(input_val_cb_index, output_val_cb_index);
    transpose_wh_init(input_ind_cb_index, output_ind_cb_index);
    /*
        process_and_sort_tiles(
                input_val_cb_index,
                input_ind_cb_index,
                transposed_val_cb_index,
                transposed_ind_cb_index,
                (Ht*Wt),
                ascending,
                logk);
                */
    // skip first tile (acquired above), then iterate over remaining tiles

    cb_reserve_back(result_prep_val_cb_index, output_tiles);
    cb_reserve_back(result_prep_ind_cb_index, output_tiles);

    print_loop(output_tiles);
    uint32_t ktiles_saved = 0;
    for (uint32_t count = 0; count < (Ht * Wt); count++) {
        acquire_dst();

        if (count == 0) {
            cb_wait_front(input_val_cb_index, 2);
            cb_wait_front(input_ind_cb_index, 2);

            reconfig_data_format_srca(input_val_cb_index);
            transpose_wh_init_short(input_val_cb_index);
            transpose_wh_tile(input_val_cb_index, 0, 0);
            transpose_wh_tile(input_val_cb_index, 1, 1);

            reconfig_data_format_srca(input_ind_cb_index);
            transpose_wh_init_short(input_ind_cb_index);
            transpose_wh_tile(input_ind_cb_index, 0, 2);
            transpose_wh_tile(input_ind_cb_index, 1, 3);

            cb_pop_front(input_val_cb_index, 2);
            cb_pop_front(input_ind_cb_index, 2);
            count++;
        } else {
            print_loop(100000 + count);
            cb_wait_front(input_val_cb_index, 1);
            cb_wait_front(input_ind_cb_index, 1);

            reconfig_data_format_srca(input_val_cb_index);
            transpose_wh_init_short(input_val_cb_index);
            transpose_wh_tile(input_val_cb_index, 0, 1);

            reconfig_data_format_srca(input_ind_cb_index);
            transpose_wh_init_short(input_ind_cb_index);
            transpose_wh_tile(input_ind_cb_index, 0, 3);

            cb_pop_front(input_val_cb_index, 1);
            cb_pop_front(input_ind_cb_index, 1);
        }

        for (uint32_t index = 0; index < output_tiles; index++) {
            if (ktiles_saved > index) {
                print_loop(1000 + index);
                reconfig_data_format_srca(result_prep_val_cb_index);
                copy_tile_init(result_prep_val_cb_index);
                copy_tile(result_prep_val_cb_index, index, 0);
                reconfig_data_format_srca(result_prep_ind_cb_index);
                copy_tile_init(result_prep_ind_cb_index);
                copy_tile(result_prep_ind_cb_index, index, 2);
            }

            // merge values - move larger 32 values into 0th dest and lower 32 values into 1st dest
            ckernel::topk_local_sort(0, (int)ascending, 5);

            print_loop(1100 + index);
            pack_reconfig_data_format(result_prep_val_cb_index);
            pack_tile<true>(0, result_prep_val_cb_index, index);
            pack_reconfig_data_format(result_prep_ind_cb_index);
            pack_tile<true>(2, result_prep_ind_cb_index, index);

            if (output_tiles > ktiles_saved) {
                ktiles_saved++;  // for tile above
            }

            // save remainder tile in resul_prep CB if we don't have enough tiles saved yes, otherwise discard
            if (output_tiles > ktiles_saved) {
                index++;  // increment index once here to avoid overwriting the tile we just added above
                print_loop(1200 + index);
                pack_reconfig_data_format(result_prep_val_cb_index);
                pack_tile<true>(1, result_prep_val_cb_index, index);
                pack_reconfig_data_format(result_prep_ind_cb_index);
                pack_tile<true>(3, result_prep_ind_cb_index, index);
                index = output_tiles;  // terminating condition for this loop
                ktiles_saved++;        // for tile added within this if statement
            }
        }
        release_dst();
    }
    cb_push_back(result_prep_val_cb_index, output_tiles);
    cb_push_back(result_prep_ind_cb_index, output_tiles);

    // transpose value tiles and pack into output buffer
    transpose_and_pack(result_prep_val_cb_index, output_val_cb_index, output_tiles);

    // transpose index tiles and pack into output buffer
    transpose_and_pack(result_prep_ind_cb_index, output_ind_cb_index, output_tiles);
}

}  // namespace NAMESPACE
