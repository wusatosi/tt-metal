// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// #include "debug/dprint.h"  // required in all kernels using DPRINT

#include <cstdint>
#include "risc_common.h"

template <typename T>
void helper_print_cb(const uint32_t cb_id, const uint32_t height, const uint32_t width, const uint32_t stick_size) {
    DPRINT << "dst " << "height=" << height << ";width=" << width << ";stick_size=" << stick_size << ENDL();
    T* cb_ptr = reinterpret_cast<T*>(get_read_ptr(cb_id));
    for (uint32_t h = 0; h < height; h++) {
        for (uint32_t w = 0; w < width; w++) {
            DPRINT << " <";
            for (uint32_t c = 0; c < stick_size; c++) {
                DPRINT << HEX() << *(cb_ptr + h * width * stick_size + w * stick_size + c) << DEC() << ",";
            }
            DPRINT << "> ";
        }
        DPRINT << ENDL();
    }
}

template <typename T>
void helper_clear_cb(
    const uint32_t cb_id, const uint32_t height, const uint32_t width, const uint32_t stick_size, const T value) {
    T* dst = reinterpret_cast<T*>(get_write_ptr(cb_id));
    for (uint32_t h = 0; h < height; h++) {
        for (uint32_t w = 0; w < width; w++) {
            for (uint32_t c = 0; c < stick_size; c++) {
                *dst = value;
                dst++;
            }
        }
    }
}

void kernel_main() {
    // if (noc_index == 0)
    // {
    //     DPRINT << "EARLY EXIT!" << ENDL();
    //     return;
    // }

    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t width = get_compile_time_arg_val(2);
    constexpr uint32_t height = get_compile_time_arg_val(3);
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t stride_h = get_compile_time_arg_val(5);
    constexpr uint32_t stride_w = get_compile_time_arg_val(6);
    constexpr uint32_t first_half = get_compile_time_arg_val(7);

    const uint32_t start_x = get_arg_val<uint32_t>(0) + VIRTUAL_TENSIX_START_X;  //
    const uint32_t end_x = get_arg_val<uint32_t>(1) + VIRTUAL_TENSIX_START_X;    //
    const uint32_t start_y = get_arg_val<uint32_t>(2) + VIRTUAL_TENSIX_START_Y;  //
    const uint32_t end_y = get_arg_val<uint32_t>(3) + VIRTUAL_TENSIX_START_Y;    //
    const uint32_t src_width_stride = get_arg_val<uint32_t>(4);
    const uint32_t src_height_offset_to_next = get_arg_val<uint32_t>(5);
    const uint32_t src_offset = get_arg_val<uint32_t>(6);
    const uint32_t dst_stride = get_arg_val<uint32_t>(7);
    const uint32_t dst_offset = get_arg_val<uint32_t>(8);

    uint32_t stick_size = stick_size_bytes / 2;

    // handy for debug
    // helper_print_cb<uint16_t>(src_cb_id, height, width, stick_size);
    // helper_clear_cb<uint16_t>(dst_cb_id, height, width, stick_size, 0);
    // helper_print_cb<uint16_t>(dst_cb_id, height, width, stick_size);

    // Go through nodes (start_x, start_y) to (end_x, end_y)
    // Copy your stick (dst_batch) to the dst buffer
    // both DM0/DM1 read from all nodes, assuming that's creating uniform load to the NOC
    // they split reading even/odd lines
    // DPRINT << "NOC" << (uint32_t)noc_index << ENDL();
    auto dst_address = get_write_ptr(dst_cb_id) + dst_offset;
    for (uint32_t src_noc_y = start_y; src_noc_y < end_y; src_noc_y++) {
        // DPRINT << "src_noc_y = " << src_noc_y << ENDL();
        for (uint32_t src_noc_x = start_x; src_noc_x < end_x; src_noc_x++) {
            // DPRINT << "src_noc_x = " << src_noc_x << ENDL();
            // src_noc_address points to the start of the first line (AB) or the second line (CD)
            auto src_noc_address = get_noc_addr(src_noc_x, src_noc_y, get_read_ptr(src_cb_id)) + src_offset;
            // Copy half of data data from src to dst
            for (uint32_t h = 0; h < height / 2; h += stride_h) {
                for (uint32_t w = 0; w < width; w += stride_w) {
                    noc_async_read_one_packet(src_noc_address, dst_address, stick_size_bytes);
                    src_noc_address += src_width_stride;
                    dst_address += stick_size_bytes;
                }
                // skip lines to the next src line
                src_noc_address += src_height_offset_to_next;
            }
            dst_address += dst_stride / 2;  // / 2; // dst_stride - one src image in dst size
        }
    }
    noc_async_read_barrier();

    // handy for debug
    // helper_print_cb<uint16_t>(dst_cb_id, height, width, stick_size);

    // DPRINT << "Deinterleave done" << ENDL();
}
