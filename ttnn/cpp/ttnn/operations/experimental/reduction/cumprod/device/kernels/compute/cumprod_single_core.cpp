// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

#include "compute_kernel_api/common.h"

#include "../cumprod_common.hpp"

namespace NAMESPACE {
void MAIN {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(1);
    const uint32_t PHi = get_arg_val<uint32_t>(2);
    const uint32_t PLo = get_arg_val<uint32_t>(3);
    const uint32_t HtWt = get_arg_val<uint32_t>(4);
    const uint32_t core_utilization_count = get_arg_val<uint32_t>(5);
    const uint32_t compute_with_storage_grid_size_x = get_arg_val<uint32_t>(6);

    const uint32_t x{get_absolute_logical_x()};
    const uint32_t y{get_absolute_logical_y()};
    const uint32_t core_id{y * compute_with_storage_grid_size_x + x};
    const uint32_t all_work_units{PHi * PLo * HtWt};
    const uint32_t first_work_unit{get_first_work_unit(core_id, core_utilization_count, all_work_units)};
    const uint32_t last_work_unit{get_last_work_unit(core_id, core_utilization_count, all_work_units)};

    // unary_op_init_common(cb_in, cb_intermed);
    binary_op_init_common(cb_in, cb_one, cb_out);

    cb_wait_front(cb_one, ONE_TILE);

    // TODO(jbbieniekTT): the following algorithm is to be explained.
    for (uint32_t i{first_work_unit}; i < last_work_unit; ++i) {
        bool enable_reload = false;
        // DPRINT << "compute: " << i << ENDL();
        // acquire_dst();
        // // tile_regs_acquire();
        // // reconfig_data_format(cb_one);
        // copy_tile_to_dst_init_short(cb_one);
        // copy_tile(cb_one, FIRST_TILE, TILE_DEST);
        // // tile_regs_commit();

        // pack_reconfig_data_format(cb_intermed);
        // // tile_regs_wait();
        // cb_reserve_back(cb_intermed, ONE_TILE);
        // pack_tile(TILE_DEST, cb_intermed);
        // cb_push_back(cb_intermed, ONE_TILE);
        // // tile_regs_release();
        // release_dst();

        for (uint32_t j = 0; j < tiles_per_row; ++j) {
            acquire_dst();
            uint32_t cb_mul = enable_reload ? cb_intermed : cb_one;
            // reconfig_data_format_srca(cb_in);
            // reconfig_data_format(cb_in, cb_intermed);
            cb_wait_front(cb_in, ONE_TILE);
            // copy_tile_to_dst_init_short(cb_in);
            // copy_tile(cb_in, first_tile, TILE_DEST);

            // reconfig_data_format_srca(cb_intermed);
            // cb_wait_front(cb_intermed, ONE_TILE);
            // copy_tile_to_dst_init_short(cb_intermed);
            // copy_tile(cb_intermed, first_tile, TILE_ACC);

            // DPRINT << "compute 111 j: " << j << ENDL();

            // if (j < 3)
            mul_tiles_init(cb_in, cb_mul);

            // DPRINT << "compute 222 j: " << j << ENDL();

            // tile_regs_acquire();

            mul_tiles(cb_in, cb_mul, FIRST_TILE, FIRST_TILE, TILE_ACC);
            // DPRINT << "compute 333 j: " << j << ENDL();

            // tile_regs_commit();

            cb_pop_front(cb_in, ONE_TILE);
            if (enable_reload) {
                cb_pop_front(cb_intermed, ONE_TILE);
            }
            // DPRINT << "compute 444 j: " << j << ENDL();

            cb_reserve_back(cb_intermed, ONE_TILE);
            // pack_reconfig_data_format(cb_intermed);
            pack_tile(TILE_ACC, cb_intermed);
            cb_push_back(cb_intermed, ONE_TILE);
            // DPRINT << "compute 555 j: " << j << ENDL();

            release_dst();

            DPRINT << "DST RELEASED!!!!!!" << ENDL();

            acquire_dst();
            // tile_regs_wait();

            // DPRINT << "compute 666 j: " << j << ENDL();

            // DPRINT << "compute 666111 j: " << j << ENDL();
            cb_wait_front(cb_intermed, ONE_TILE);
            // if (j < 3)
            copy_tile_to_dst_init_short(cb_intermed);
            copy_tile(cb_intermed, FIRST_TILE, TILE_DEST);
            // pack_reconfig_data_format(cb_out);
            cb_reserve_back(cb_out, ONE_TILE);
            // pack_reconfig_data_format(cb_out);
            // DPRINT << "compute 777333 j: " << j << ENDL();
            pack_tile(TILE_DEST, cb_out);
            // DPRINT << "compute 777444 j: " << j << ENDL();
            cb_push_back(cb_out, ONE_TILE);

            // DPRINT << "compute 777 j: " << j << ENDL();

            // cb_reserve_back(cb_intermed, ONE_TILE);
            // // pack_reconfig_data_format(cb_intermed);
            // pack_tile(TILE_DEST, cb_intermed);
            // cb_push_back(cb_intermed, ONE_TILE);

            // tile_regs_release();
            release_dst();
            enable_reload = true;

            DPRINT << "END j: " << j << ENDL();
        }

        // DPRINT << "compute 999: " << i << ENDL();
        // cb_wait_front(cb_intermed, ONE_TILE);
        cb_pop_front(cb_intermed, ONE_TILE);
    }

    cb_pop_front(cb_one, ONE_TILE);
}

}  // namespace NAMESPACE
