// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dprint.h"

void kernel_main() {
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t input_unit_size_in_bytes = get_compile_time_arg_val(2);  // stick size
    constexpr uint32_t num_input_channels = get_compile_time_arg_val(3);
    constexpr uint32_t input_width = get_compile_time_arg_val(4);

    // Runtime-args
    const uint32_t num_sticks = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = num_sticks / input_width;

    // temp, fixme
    constexpr uint32_t padded_stick_num_elements = input_unit_size_in_bytes / 2;

    constexpr uint32_t output_width = input_width * num_input_channels / 2;

    volatile tt_l1_ptr uint16_t* src_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(src_cb_id));
    volatile tt_l1_ptr uint16_t* dst_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(dst_cb_id));
    uint32_t num_outputs_written = 0;
    for (uint32_t i = 0; i < num_sticks; i++) {
        // Copy the first half of the sticks to the destination
        for (uint32_t j = 0; j < num_input_channels / 2; j++) {
            dst_cb_ptr[j * padded_stick_num_elements] = src_cb_ptr[j];
        }

        // Copy the second half of the sticks to the destination in the row below
        volatile tt_l1_ptr uint16_t* dst_cb_ptr_row_below = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
            reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst_cb_ptr) + output_width * input_unit_size_in_bytes);
        for (uint32_t j = num_input_channels / 2; j < num_input_channels; j++) {
            dst_cb_ptr_row_below[j * padded_stick_num_elements] = src_cb_ptr[j];
        }

        num_outputs_written += num_input_channels;
        src_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
            reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src_cb_ptr) + input_unit_size_in_bytes);
        dst_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
            reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst_cb_ptr) +
            (num_input_channels / 2) * input_unit_size_in_bytes);
        if (num_outputs_written == output_width) {
            num_outputs_written = 0;
            // skip row we have just written to
            dst_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst_cb_ptr) + output_width * input_unit_size_in_bytes);
        }
    }
}
