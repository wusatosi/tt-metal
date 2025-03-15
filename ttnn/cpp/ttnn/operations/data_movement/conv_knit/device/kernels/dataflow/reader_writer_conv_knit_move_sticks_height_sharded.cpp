// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "tt_metal/hw/inc/utils/utils.h"

inline __attribute__((always_inline)) constexpr uint32_t log2_constexpr(uint32_t n) {
    uint32_t p = 0;
    while (n > 1) {
        n >>= 1;
        ++p;
    }
    return p;
}

void kernel_main() {
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t input_unit_size_in_bytes = get_compile_time_arg_val(2);  // stick size
    constexpr uint32_t num_input_channels = get_compile_time_arg_val(3);
    constexpr uint32_t input_width = get_compile_time_arg_val(4);
    constexpr uint32_t num_output_channels = get_compile_time_arg_val(5);
    constexpr uint32_t num_sticks_for_all_riscvs = get_compile_time_arg_val(6);
    constexpr uint32_t num_sticks_for_this_riscv = get_compile_time_arg_val(7);
    constexpr uint32_t current_riscv_stick_starting_index = get_compile_time_arg_val(8);

    // temp, fixme
    constexpr uint32_t stick_num_elements = input_unit_size_in_bytes / 2;  // assuming hardcoded bfloat16
    constexpr uint32_t output_width = input_width * num_input_channels / (2 * num_output_channels);
    constexpr uint32_t num_elements_to_write_in_dst_stick = num_output_channels;
    constexpr uint32_t half_of_input_channels = num_input_channels / 2;
    constexpr uint32_t num_widths_done_by_other_riscvs = current_riscv_stick_starting_index / input_width;
    constexpr bool num_elements_to_write_in_dst_stick_is_power_of_2 = is_power_of_2(num_elements_to_write_in_dst_stick);

    volatile tt_l1_ptr uint16_t* src_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(src_cb_id));
    volatile tt_l1_ptr uint16_t* dst_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(dst_cb_id));
    src_cb_ptr += current_riscv_stick_starting_index * stick_num_elements;
    dst_cb_ptr += num_widths_done_by_other_riscvs * output_width * stick_num_elements +
                  current_riscv_stick_starting_index * 2 *
                      stick_num_elements;  // move to adequate row in dst in addition to moving to the next sticks

    uint32_t num_input_sticks_read = current_riscv_stick_starting_index - num_widths_done_by_other_riscvs * input_width;
    for (uint32_t i = 0; i < num_sticks_for_this_riscv; i++) {
        uint32_t written_in_dst_stick = 0;
        uint32_t stick_index = 0;

        // Copy the second half of the sticks to the destination in the row below
        volatile tt_l1_ptr uint16_t* dst_cb_ptr_row_below = dst_cb_ptr + output_width * stick_num_elements;

#pragma GCC unroll 16
        for (uint32_t j = 0; j < half_of_input_channels; j++) {
            uint32_t dst_index = stick_index * stick_num_elements + written_in_dst_stick;
            dst_cb_ptr[dst_index] = src_cb_ptr[j];
            dst_cb_ptr_row_below[dst_index] = src_cb_ptr[j + half_of_input_channels];
            written_in_dst_stick++;

            if constexpr (num_elements_to_write_in_dst_stick_is_power_of_2) {
                uint16_t temp = written_in_dst_stick;
                stick_index += (temp >> log2_constexpr(num_elements_to_write_in_dst_stick));
                written_in_dst_stick = temp & (num_elements_to_write_in_dst_stick - 1);
            } else {
                stick_index += written_in_dst_stick / num_elements_to_write_in_dst_stick;
                written_in_dst_stick %= num_elements_to_write_in_dst_stick;
            }
        }

        src_cb_ptr += stick_num_elements;
        dst_cb_ptr += 2 * stick_num_elements;  // we wrote 2 sticks in stick index, move it by 2
        num_input_sticks_read++;

        if (__builtin_expect(num_input_sticks_read == input_width, 0)) {
            num_input_sticks_read = 0;
            // skip row we have just written to
            dst_cb_ptr += output_width * stick_num_elements;
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
