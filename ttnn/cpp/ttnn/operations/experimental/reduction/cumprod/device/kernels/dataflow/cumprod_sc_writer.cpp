// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../cumprod_common.hpp"

void kernel_main() {
    const uint32_t output_dram_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(2);
    const uint32_t PHi = get_arg_val<uint32_t>(3);
    const uint32_t PLo = get_arg_val<uint32_t>(4);
    const uint32_t HtWt = get_arg_val<uint32_t>(5);
    const uint32_t core_utilization_count = get_arg_val<uint32_t>(6);
    const uint32_t compute_with_storage_grid_size_x = get_arg_val<uint32_t>(7);

    const uint32_t x{get_absolute_logical_x()};
    const uint32_t y{get_absolute_logical_y()};
    const uint32_t core_id{y * compute_with_storage_grid_size_x + x};
    const uint32_t all_work_units{PHi * PLo * HtWt};
    const uint32_t first_work_unit{get_first_work_unit(core_id, core_utilization_count, all_work_units)};
    const uint32_t last_work_unit{get_last_work_unit(core_id, core_utilization_count, all_work_units)};

    const uint32_t ublock_size_bytes = get_tile_size(cb_out);
    const uint32_t input_sram_addr = get_read_ptr(cb_out);

    const auto input_dataformat = get_dataformat(cb_out);
    const auto output_data_format = get_dataformat(cb_out);

    uint32_t bytes_per_element{};

    switch (input_dataformat) {
        case DataFormat::Float32: bytes_per_element = 4; break;
        case DataFormat::Float16_b:
        case DataFormat::Float16: bytes_per_element = 2; break;
        case DataFormat::UInt8: bytes_per_element = 1; break;
        case DataFormat::UInt16: bytes_per_element = 2; break;
        case DataFormat::Int32:
        case DataFormat::UInt32: bytes_per_element = 4; break;
        default: bytes_per_element = 4; break;
    }

    const uint32_t& output_tile_bytes = ublock_size_bytes;

    InterleavedAddrGenFast<true> dram_output_addrg = {
        .bank_base_address = output_dram_base_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    // TODO(jbbieniekTT): the following algorithm is to be explained.
    for (uint32_t i{first_work_unit}; i < last_work_unit; ++i) {
        const uint32_t i0{i / (PHi * HtWt)};
        const uint32_t i1{i % (PHi * HtWt)};
        for (uint32_t j{0}; j < tiles_per_row; ++j) {
            const uint32_t tileid = get_tile_id(i0, i1, j, tiles_per_row, PLo, PHi, HtWt);

            cb_reserve_back(cb_out, ONE_TILE);

            const uint32_t data_sram_addr = get_write_ptr(cb_out);
            noc_async_read_tile(tileid, dram_output_addrg, data_sram_addr);
            noc_async_read_barrier();

            cb_push_back(cb_out, ONE_TILE);
        }
    }

    // for (unsigned i0 = 0; i0 < PLo; i0++) {
    //     for (unsigned i1 = 0; i1 < PHi * HtWt; i1++) {
    //         for (unsigned j = 0; j < tiles_per_row; j++) {
    //             uint32_t tileid = get_tile_id(i0, i1, j, tiles_per_row, PLo, PHi, HtWt);

    //             cb_wait_front(cb_in, ONE_TILE);

    //             noc_async_write_tile(tileid, dram_output_addrg, input_sram_addr);
    //             noc_async_write_barrier();

    //             cb_pop_front(cb_in, ONE_TILE);
    //         }
    //     }
    // }
}
