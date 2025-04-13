// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// #include "debug/dprint.h"  // required in all kernels using DPRINT

constexpr uint32_t round_up_to_multiple_of_64(uint32_t value) { return (value + 63) & ~63; }

void kernel_main() {
    // Deinterleaves input image in src cb to dest cb.
    //
    // The input data is expected to be interleaved in the following way:
    //     A B A B A B
    //     C D C D C D
    //     A B A B A B
    //     C D C D C D
    // The output data is expected to be deinterleaved in the following way:
    //     A A A
    //     A A A
    //     B B B
    //     B B B
    //     C C C
    //     C C C
    //     D D D
    //     D D D
    //
    // Image width, height and number of channels are given as compile time arguments.
    // Kernel processes AB or CD lines depending on the value of AB_notCD argument.

    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t width = get_compile_time_arg_val(2);
    constexpr uint32_t height = get_compile_time_arg_val(3);
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t stride_h = get_compile_time_arg_val(5);
    constexpr uint32_t stride_w = get_compile_time_arg_val(6);
    constexpr uint32_t AB_notCD = get_compile_time_arg_val(7);

    // DPRINT << "Deinterelave" << src_cb_id <<  dst_cb_id << " " << width << " " << height << " " << stick_size <<
    // AB_notCD << ENDL();

    constexpr uint32_t line_size_bytes = width * stick_size_bytes;
    constexpr uint32_t batch_size_bytes = height * line_size_bytes / stride_h / stride_w;

    // src_noc_address points to the start of the first line (AB) or the second line (CD)
    // src_noc_address_even points to A or C
    // src_noc_address_odd points to B or D
    auto src_noc_address_odd =
        get_noc_addr(get_read_ptr(src_cb_id)) + (AB_notCD ? 0 : line_size_bytes) + stick_size_bytes;
    auto src_noc_address_even = get_noc_addr(get_read_ptr(src_cb_id)) + (AB_notCD ? 0 : line_size_bytes);

    // dst address points to the start of batch output in dst buffer (ABCD)
    // even processes As or Cs, odd processes Bs or Ds
    // dst_address_even points to A or C
    // dst_address_odd points to B or D
    constexpr uint32_t dst_base_offset_even = (AB_notCD) ? 0 : 2 * batch_size_bytes;
    constexpr uint32_t dst_base_offset_odd = (AB_notCD) ? batch_size_bytes : 3 * batch_size_bytes;
    auto dst_address_even = get_write_ptr(dst_cb_id) + dst_base_offset_even;
    auto dst_address_odd = get_write_ptr(dst_cb_id) + dst_base_offset_odd;

    constexpr uint32_t h_start = (AB_notCD) ? 0 : 1;

    // Copy data from src to dst
    for (uint32_t h = h_start; h < height; h += stride_h) {
        for (uint32_t w = 0; w < width; w += stride_w) {
            // this order gave best performance. 70% utilization when single core
            noc_async_read_one_packet(src_noc_address_odd, dst_address_odd, stick_size_bytes);
            src_noc_address_odd += 2 * stick_size_bytes;
            dst_address_odd += stick_size_bytes;
            noc_async_read_one_packet(src_noc_address_even, dst_address_even, stick_size_bytes);
            src_noc_address_even += 2 * stick_size_bytes;
            dst_address_even += stick_size_bytes;
        }
        // skip one line
        src_noc_address_odd += line_size_bytes;
        src_noc_address_even += line_size_bytes;
        // DPRINT << "Deinterleave " << h << ENDL();
    }
    noc_async_read_barrier();

    // DPRINT << "Deinterleave done" << ENDL();
}
