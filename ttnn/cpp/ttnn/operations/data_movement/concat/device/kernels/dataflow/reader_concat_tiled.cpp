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
    uint32_t num_input_tensors = get_compile_time_arg_val(9);
    // hardware constraints
    constexpr uint32_t tile_size = get_compile_time_arg_val(10);
    constexpr uint32_t tile_hw = tile_size * tile_size;
    constexpr uint32_t face_size = get_compile_time_arg_val(11);
    constexpr uint32_t face_hw = face_size * face_size;
    constexpr uint32_t alignment_adjustor = 16;

    constexpr uint32_t dst_addr = get_compile_time_arg_val(12);

    uint32_t rt_arg_ind = 0;
    uint32_t cb_page_size = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t num_elems = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t starting_tile_offset = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t current_tensor = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t starting_row = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t starting_col = get_arg_val<uint32_t>(rt_arg_ind++);

    // Organize input tensor information
    uint32_t input_tensor_addrs[num_input_tensors];
    uint32_t input_tensor_volumes[num_input_tensors];
    uint32_t input_tensor_heights[num_input_tensors];
    uint32_t input_tensor_widths[num_input_tensors];
    uint32_t input_tensor_tiles_per_2d_tensors[num_input_tensors];
    uint32_t input_tensor_tiles_per_tile_rows[num_input_tensors];

    for (uint32_t i = 0; i < num_input_tensors; i++) {
        input_tensor_addrs[i] = get_arg_val<uint32_t>(rt_arg_ind++);
        input_tensor_volumes[i] = get_arg_val<uint32_t>(rt_arg_ind++);
        input_tensor_heights[i] = get_arg_val<uint32_t>(rt_arg_ind++);
        input_tensor_widths[i] = get_arg_val<uint32_t>(rt_arg_ind++);
        // calculate tiles_per_2d_tensor and tiles_per_tile_rows
        uint32_t input_tensor_padded_height = (input_tensor_heights[i] / tile_size + 1) * tile_size;
        uint32_t input_tensor_padded_width = (input_tensor_widths[i] / tile_size + 1) * tile_size;
        input_tensor_tiles_per_2d_tensors[i] =
            (input_tensor_padded_height / tile_size) * (input_tensor_padded_width / tile_size);
        input_tensor_tiles_per_tile_rows[i] = input_tensor_padded_width / tile_size;
    }

    const DataFormat data_format = get_dataformat(cb_id_0);
    InterleavedAddrGenFast<tensor_in_dram> s0 = {
        .bank_base_address = dst_addr,
        .page_size = tile_hw * element_size_bytes,
        .data_format = data_format  // page_size needs to be tile_size_bytes
    };

    // Reserve and push the fill value into the circular buffer
    cb_reserve_back(cb_id_0, 1);
    uint32_t l1_read_addr = get_read_ptr(cb_id_0);

    uint32_t row = starting_row;
    uint32_t col = starting_col;
    uint32_t curr_tile = (row / tile_size) * tiles_per_tile_row + (col / tile_size) + starting_tile_offset;
    uint32_t r_f_offset = ((row % tile_size) / face_size) * 2 * face_hw + (row % face_size) * face_size;
    uint32_t c_f_offset = ((col % tile_size) / face_size) * face_hw + (col % face_size);
    uint32_t face_offset = r_f_offset + c_f_offset;
    uint32_t remaining_elems = num_elems;

    while (remaining_elems > 0) {
        uint64_t start_tile_noc_addr = get_noc_addr(curr_tile, s0);
        uint32_t face = face_offset / (face_hw);
        uint32_t till_row_end = logical_width - col;

        uint64_t src_noc_addr = start_tile_noc_addr + face_offset * element_size_bytes;
        uint32_t elems_in_face = col % face_size == 0 ? face_size : face_size - (col % face_size);
        uint32_t elems_to_read = std::min({elems_in_face, till_row_end, remaining_elems});
        uint32_t bytes_to_read = elems_to_read * element_size_bytes;

        noc_async_read(src_noc_addr, l1_read_addr, bytes_to_read);
        noc_async_read_barrier();
        cb_push_back(cb_id_0, 1);

        remaining_elems -= elems_to_read;
        col += elems_to_read;
        face_offset += elems_to_read;

        // Check if we need to move to next row
        if (col >= logical_width) {
            row++;
            col = 0;

            // Check if we've completed the current 2D tensor
            if (row >= input_tensor_heights[current_tensor]) {
                // Round up to the next tensor's starting tile
                // uint32_t current_tensor = curr_tile / tiles_per_2d_tensor;
                current_tensor++;
                s0.bank_base_address = input_tensor_addrs[current_tensor];
                curr_tile = (current_tensor + 1) * tiles_per_2d_tensor + starting_tile_offset;
                row = 0;
            } else {
                curr_tile = (row / tile_size) * tiles_per_tile_row + (col / tile_size) + starting_tile_offset;
            }

            r_f_offset = ((row % tile_size) / face_size) * 2 * face_hw + (row % face_size) * face_size;
            c_f_offset = ((col % tile_size) / face_size) * face_hw + (col % face_size);
            face_offset = r_f_offset + c_f_offset;
            continue;
        }

        // Handle face transitions within the same row
        if (face % 2 == 0) {
            face_offset += face_size * (face_size - 1);
        } else {
            curr_tile++;
            face_offset -= face_size * (face_size + 1);
        }
    }

    noc_async_write_barrier();
}
