// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../cumprod_common.hpp"

namespace {

constexpr union {
    float f;
    int32_t u;
} caster{.f = 1.0f};

}  // namespace

void kernel_main() {
    const uint32_t input_dram_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(2);
    const uint32_t PHi = get_arg_val<uint32_t>(3);
    const uint32_t PLo = get_arg_val<uint32_t>(4);
    const uint32_t HtWt = get_arg_val<uint32_t>(5);
    const uint32_t core_loop_count = get_arg_val<uint32_t>(6);
    const uint32_t total_number_of_cores = get_arg_val<uint32_t>(7);
    const uint32_t compute_with_storage_grid_size_x = get_arg_val<uint32_t>(8);
    const uint32_t compute_with_storage_grid_size_y = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_out = tt::CBIndex::c_0;
    constexpr uint32_t cb_one = tt::CBIndex::c_2;
    const uint32_t x{get_absolute_logical_x()};
    const uint32_t y{get_absolute_logical_y()};
    const uint32_t core_id{y * compute_with_storage_grid_size_x + x};
    const uint32_t all_work_units{PHi * PLo * HtWt};
    const uint32_t start_tile{get_start_tile(core_id, total_number_of_cores, all_work_units)};
    const uint32_t end_tile{get_end_tile(core_id, total_number_of_cores, all_work_units)};

    cb_reserve_back(cb_one, ONE_TILE);
    uint32_t data_one_addr = get_write_ptr(cb_one);

    const int32_t ACC_START_VALUE_F32{caster.u};
    constexpr int32_t ACC_START_VALUE_F16{0x3F80};
    // TODO(jbbieniekTT): the below ones will work only if applied LLK is appropriately preconfigured for those.
    constexpr int32_t ACC_START_VALUE_I32{0x1};
    constexpr int32_t ACC_START_VALUE_I16{0x1};
    constexpr int32_t ACC_START_VALUE_I8{0x1};

    const auto& input_data_format = get_dataformat(cb_in);

    uint32_t ublock_size_bytes = get_tile_size(cb_in);
    uint32_t l1_addr_out = get_write_ptr(cb_in);

    const uint32_t input_tile_bytes = ublock_size_bytes;
    const uint32_t output_tile_bytes = ublock_size_bytes;
    InterleavedAddrGenFast<true> dram_input_addrg = {
        .bank_base_address = input_dram_base_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    uint32_t scaler{0};

    uint32_t bytes_per_element = 4;
    switch (input_data_format) {
        case DataFormat::Float32:
            scaler = ACC_START_VALUE_F32;
            bytes_per_element = 4;
            break;
        case DataFormat::Float16_b:
        case DataFormat::Float16:
            scaler = (ACC_START_VALUE_F16 << 16) | ACC_START_VALUE_F16;
            bytes_per_element = 2;
            break;
        case DataFormat::UInt8:
            scaler = (ACC_START_VALUE_I8 << 24) | (ACC_START_VALUE_I8 << 16) | (ACC_START_VALUE_I8 << 8) |
                     (ACC_START_VALUE_I8);
            bytes_per_element = 1;
            break;
        case DataFormat::UInt16:
            scaler = (ACC_START_VALUE_I16 << 16) | ACC_START_VALUE_I16;
            bytes_per_element = 2;
            break;
        case DataFormat::Int32:
        case DataFormat::UInt32:
            scaler = ACC_START_VALUE_I32;
            bytes_per_element = 4;
            break;
        default:
            scaler = 1;
            bytes_per_element = 4;
            break;
    }

    // TODO(jbbieniekTT): the following algorithm is to be explained.
    uint32_t* data_one{(uint32_t*)data_one_addr};
    for (uint32_t i = 0; i < ublock_size_bytes / sizeof(decltype(data_one)); i++) {
        data_one[i] = scaler;
    }

    cb_push_back(cb_one, ONE_TILE);

    for (uint32_t i{start_tile}; i < end_tile; ++i) {
        const uint32_t i0{i / (PHi * HtWt)};
        const uint32_t i1{i % (PHi * HtWt)};
        for (uint32_t j{0}; j < tiles_per_row; ++j) {
            const uint32_t tileid = get_tile_id(i0, i1, j, tiles_per_row, PLo, PHi, HtWt);

            cb_reserve_back(cb_in, ONE_TILE);

            const uint32_t data_sram_addr = get_write_ptr(cb_in);
            noc_async_read_tile(tileid, dram_input_addrg, data_sram_addr);
            noc_async_read_barrier();

            cb_push_back(cb_in, ONE_TILE);
        }
    }

    // for (unsigned i0 = 0; i0 < PLo; i0++) {
    //     for (unsigned i1 = 0; i1 < PHi * HtWt; i1++) {
    //         for (unsigned j = 0; j < tiles_per_row; j++) {
    //             uint32_t tileid = get_tile_id(i0, i1, j, tiles_per_row, PLo, PHi, HtWt);

    //             cb_reserve_back(cb_out, 1);

    //             uint32_t data_sram_addr = get_write_ptr(cb_out);
    //             noc_async_read_tile(tileid, dram_input_addrg, data_sram_addr);
    //             noc_async_read_barrier();

    //             cb_push_back(cb_out, 1);
    //         }
    //     }
    // }
}
