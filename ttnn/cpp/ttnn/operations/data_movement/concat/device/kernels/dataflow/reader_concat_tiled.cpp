// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_id_0 = get_compile_time_arg_val(0);
    constexpr bool tensor_in_dram = get_compile_time_arg_val(1) == 1;
    const uint32_t element_size_bytes = get_compile_time_arg_val(2);
    uint32_t logical_height = get_compile_time_arg_val(3);
    uint32_t logical_width = get_compile_time_arg_val(4);
    uint32_t padded_height = get_compile_time_arg_val(5);
    uint32_t padded_width = get_compile_time_arg_val(6);
    uint32_t tiles_per_2d_tensor = get_compile_time_arg_val(7);
    uint32_t tiles_per_tile_row = get_compile_time_arg_val(8);
    // hardware constraints
    constexpr uint32_t tile_size = get_compile_time_arg_val(9);
    constexpr uint32_t tile_hw = tile_size * tile_size;
    constexpr uint32_t face_size = get_compile_time_arg_val(10);
    constexpr uint32_t face_hw = face_size * face_size;
    constexpr uint32_t alignment_adjustor = 16;

    uint32_t rt_arg_ind = 0;
    uint32_t dst_addr = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t cb_page_size = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t starting_tile_offset = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t num_elems = get_arg_val<uint32_t>(rt_arg_ind++);
    // uint32_t starting_dim = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t starting_row = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t starting_col = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t starting_tile = get_arg_val<uint32_t>(rt_arg_ind++);

    const DataFormat data_format = get_dataformat(cb_id_0);
    const InterleavedAddrGenFast<tensor_in_dram> s0 = {
        .bank_base_address = dst_addr,
        .page_size = tile_hw * element_size_bytes,
        .data_format = data_format  // page_size needs to be tile_size_bytes
    };

    // Reserve and push the fill value into the circular buffer
    cb_reserve_back(cb_id_0, 1);
    uint32_t l1_read_addr = get_read_ptr(cb_id_0);
    volatile tt_l1_ptr uint32_t* l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);

    uint32_t row = starting_row;
    uint32_t col = starting_col;
    uint32_t till_col_end = logical_width - starting_col;
    // uint32_t curr_tile = starting_tile;
    uint32_t curr_tile = (row / tile_size) * tiles_per_tile_row + (start_col / tile_size) + tile_offset;
    uint32_t r_f_offset = ((row % tile_size) / face_size) * 2 * face_hw + (row % face_size) * face_size;
    uint32_t c_f_offset = ((start_col % tile_size) / face_size) * face_hw + (start_col % face_size);
    uint32_t face_offset = r_f_offset + c_f_offset;
    for (uint32_t i = num_elems; i > 0; i--) {
        // We want to begin reading tensor values into L1 for the writer to write to the new concatenated tensor
        uint64_t start_tile_noc_addr = get_noc_addr(curr_tile, s0);
        uint64_t src_noc_addr = start_tile_noc_addr + face_offset * element_size_bytes;
        // uint32_t alignment_offset = dst_noc_addr % alignment_adjustor;
        uint32_t elems_to_read = col % face_size == 0 ? face_size : face_size - (col % face_size);
        if (elems_to_read > till_col_end) {
            elems_to_read = till_col_end;
        }
        uint32_t bytes_to_read = elems_to_read * element_size_bytes;
    }

    auto fill_pad_2d_tensor = [&](const uint32_t& tile_offset) {
        uint32_t start_col;
        for (uint32_t row = 0; row < padded_height; row++) {
            if (row < logical_height) {
                start_col = logical_width;
            } else {
                start_col = 0;
            }
            uint32_t curr_tile = (row / tile_size) * tiles_per_tile_row + (start_col / tile_size) + tile_offset;
            uint32_t r_f_offset = ((row % tile_size) / face_size) * 2 * face_hw + (row % face_size) * face_size;
            uint32_t c_f_offset = ((start_col % tile_size) / face_size) * face_hw + (start_col % face_size);
            uint32_t face_offset = r_f_offset + c_f_offset;

            for (uint32_t col = start_col; col < padded_width;) {
                // so for each iteration of col, we will be writing at most 2 faces
                uint64_t start_tile_noc_addr = get_noc_addr(curr_tile, s0);
                uint32_t face = face_offset / (face_hw);

                uint64_t dst_noc_addr = start_tile_noc_addr + face_offset * element_size_bytes;
                uint32_t alignment_offset = dst_noc_addr % alignment_adjustor;
                uint32_t elems_to_write = col % face_size == 0 ? face_size : face_size - (col % face_size);
                uint32_t bytes_to_write = elems_to_write * element_size_bytes;
                noc_async_write(l1_write_addr + alignment_offset, dst_noc_addr, bytes_to_write);
                col += elems_to_write;
                face_offset += elems_to_write;

                if (face % 2 == 0) {
                    face_offset += face_size * (face_size - 1);
                } else {
                    curr_tile++;
                    face_offset -= face_size * (face_size + 1);
                }
            }
        }
    };

    for (uint32_t t = 0; t < num_2d_tensors; t++) {
        fill_pad_2d_tensor(t * tiles_per_2d_tensor + starting_tile_offset);
    }
    noc_async_write_barrier();
}
