// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dprint.h"

void kernel_main() {
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t input_unit_size_in_bytes = get_compile_time_arg_val(2);  // stick size
    constexpr uint32_t num_input_channels = get_compile_time_arg_val(3);
    constexpr uint32_t input_width = get_compile_time_arg_val(4);
    constexpr uint32_t num_output_channels = get_compile_time_arg_val(5);

    // Runtime-args
    const uint32_t num_sticks = get_arg_val<uint32_t>(0);

    // temp, fixme
    constexpr uint32_t stick_num_elements = input_unit_size_in_bytes / 2;  // assuming hardcoded bfloat16
    constexpr uint32_t output_width = input_width * num_input_channels / (2 * num_output_channels);

    volatile tt_l1_ptr uint16_t* src_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(src_cb_id));
    volatile tt_l1_ptr uint16_t* dst_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(dst_cb_id));

    constexpr uint32_t num_elements_to_write_in_dst_stick = num_output_channels;
    constexpr uint32_t half_of_input_channels = num_input_channels / 2;

    uint32_t num_input_sticks_read = 0;
    uint32_t num_sticks_to_traverse = num_sticks;
    for (uint32_t i = 0; i < num_sticks_to_traverse; i++) {
        uint32_t written_in_dst_stick = 0;
        uint32_t stick_index = 0;
        for (uint32_t j = 0; j < half_of_input_channels; j++) {
            dst_cb_ptr[stick_index * stick_num_elements + written_in_dst_stick] = src_cb_ptr[j];
            written_in_dst_stick++;
            if (written_in_dst_stick == num_elements_to_write_in_dst_stick) {
                stick_index++;
                written_in_dst_stick = 0;
            }
        }

        // Copy the second half of the sticks to the destination in the row below
        volatile tt_l1_ptr uint16_t* dst_cb_ptr_row_below = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
            reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst_cb_ptr) + output_width * input_unit_size_in_bytes);

        written_in_dst_stick = 0;
        stick_index = 0;
        for (uint32_t j = 0; j < half_of_input_channels; j++) {
            dst_cb_ptr_row_below[stick_index * stick_num_elements + written_in_dst_stick] =
                src_cb_ptr[j + half_of_input_channels];
            written_in_dst_stick++;
            if (written_in_dst_stick == num_elements_to_write_in_dst_stick) {
                stick_index++;
                written_in_dst_stick = 0;
            }
        }

        src_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
            reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src_cb_ptr) + input_unit_size_in_bytes);
        dst_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
            reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst_cb_ptr) +
            (2 * input_unit_size_in_bytes));  // we wrote 2 sticks in stick index, move it by 2
        num_input_sticks_read++;
        if (num_input_sticks_read == input_width) {
            num_input_sticks_read = 0;
            // skip row we have just written to
            dst_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst_cb_ptr) + output_width * input_unit_size_in_bytes);
        }
    }

    // dst_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(dst_cb_id));
    // for (uint32_t i = 0; i < 18; i++) {
    //         DPRINT << "ROW:" << ENDL();
    //     for (uint32_t j = 0; j < 130; j++) {
    //         for (int k = 0; k < 1; k++) {
    //             DPRINT << dst_cb_ptr[i * input_unit_size_in_bytes * 130 + j * input_unit_size_in_bytes + k] << ",";
    //         }
    //     }
    //         DPRINT << "ROW END" << ENDL();
    // }
    //     dst_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(dst_cb_id));
    // for (uint32_t i = 0; i < 18; i++) {
    //         DPRINT << "ROW:" << ENDL();
    //     for (uint32_t j = 0; j < 130; j++) {
    //             DPRINT << dst_cb_ptr[i * stick_num_elements * 130 + j * stick_num_elements + 0] << ",
    //             addr: " <<  (uint32_t) &dst_cb_ptr[i * stick_num_elements * 130 + j *
    //             stick_num_elements + 0] << ",";
    //             //DPRINT << dst_cb_ptr[i * stick_num_elements * 130 + j * stick_num_elements + 1] <<
    //             ",";
    //     }
    //         DPRINT << "ROW END" << ENDL();
    // }
}
