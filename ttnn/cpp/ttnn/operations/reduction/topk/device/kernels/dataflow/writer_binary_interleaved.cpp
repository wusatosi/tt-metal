// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

#include "debug/dprint.h"

inline void print_loop(uint32_t count) { DPRINT << "=writer:" << (uint32_t)count << ENDL(); }

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "=writer=!" << ENDL();
    for (uint8_t r = 0; r < 1; ++r) {
        SliceRange sr_left = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 16, .ws = 1};
        SliceRange sr_right = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 17, .w1 = 32, .ws = 1};
        DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " "
               << TileSlice(cb_id, tile_id, sr_right, true, untilize) << ENDL();
    }
}

void kernel_main() {
    uint32_t dst_addr0 = get_arg_val<uint32_t>(0);
    uint32_t dst_addr1 = get_arg_val<uint32_t>(1);

    constexpr uint32_t output_val_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t output_ind_cb_index = get_compile_time_arg_val(1);
    constexpr bool values_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool output_ind_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t Ht = get_compile_time_arg_val(4);
    constexpr uint32_t K = get_compile_time_arg_val(5);
    constexpr uint32_t Kt = (K + 31) / 32;

    // can amortize the noc reads by doing them side by side for the two tensors
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes_values = get_tile_size(output_val_cb_index);
    const DataFormat data_format_values = get_dataformat(output_val_cb_index);

    const InterleavedAddrGenFast<values_is_dram> interleaved_accessor0 = {
        .bank_base_address = dst_addr0, .page_size = tile_bytes_values, .data_format = data_format_values};

    const uint32_t tile_bytes_ind = get_tile_size(output_ind_cb_index);
    const DataFormat data_format_ind = get_dataformat(output_ind_cb_index);

    const InterleavedAddrGenFast<output_ind_is_dram> interleaved_accessor1 = {
        .bank_base_address = dst_addr1, .page_size = tile_bytes_ind, .data_format = data_format_ind};

    // Get Kt rows of values and then Kt rows of indices from compute kernel
    for (uint32_t i = 0; i < Kt; ++i) {
        // topk values
        cb_wait_front(output_val_cb_index, onetile);
        uint32_t l1_read_addr_val = get_read_ptr(output_val_cb_index);
        noc_async_write_tile(i, interleaved_accessor0, l1_read_addr_val);
        noc_async_write_barrier();
        cb_pop_front(output_val_cb_index, onetile);

        cb_wait_front(output_ind_cb_index, onetile);
        uint32_t l1_read_addr_ind = get_read_ptr(output_ind_cb_index);
        noc_async_write_tile(i, interleaved_accessor1, l1_read_addr_ind);
        noc_async_write_barrier();
        cb_pop_front(output_ind_cb_index, onetile);
    }
}
