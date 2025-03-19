// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "firmware_common.h"

inline void print_bf16_pages(uint32_t l1_addr, uint32_t elts_per_page, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * elts_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < elts_per_page; ++j, ++ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}

#define dump(a)                                       \
    do {                                              \
        DPRINT << "Act: " << #a " = " << a << ENDL(); \
    } while (false);

#define DILATION_W get_compile_time_arg_val(4)
void kernel_main() {
    constexpr bool act_in_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t stride_w = get_compile_time_arg_val(2);
    constexpr uint32_t dilation_h = get_compile_time_arg_val(3);
    constexpr uint32_t dilation_w = get_compile_time_arg_val(4);
    constexpr uint32_t conv_act_size_w_ = get_compile_time_arg_val(5);
    constexpr uint32_t conv_output_w_last_index = get_compile_time_arg_val(6) - 1;
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(7);
    // need to have these as compile-time, they are inner loop bouds / unroll loops / constexpr conditionals based on
    // them
    constexpr uint32_t window_outer = get_compile_time_arg_val(8);
    constexpr uint32_t window_inner = get_compile_time_arg_val(9);
    constexpr uint32_t act_block_h_datums = get_compile_time_arg_val(10);
    constexpr uint32_t act_block_num_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t weight_size_w = get_compile_time_arg_val(12);
    uint32_t conv_act_size_w_padded = get_compile_time_arg_val(13);
    constexpr uint32_t act_block_w_extra_align_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t weight_size_h = get_compile_time_arg_val(15);
    constexpr uint32_t act_num_blocks_h = get_compile_time_arg_val(16);
    constexpr uint32_t act_block_h_datums_last_block = get_compile_time_arg_val(25);

    constexpr uint32_t act_block_h_datums_read_last_block =
        act_block_h_datums_last_block > act_block_h_datums ? act_block_h_datums / 2 : act_block_h_datums_last_block / 2;
    constexpr uint32_t act_block_h_datums_second_reader = get_compile_time_arg_val(26);
    constexpr uint32_t act_block_h_datums_second_reader_read = act_block_h_datums_second_reader / 2;

    constexpr uint32_t cb_id_act = get_compile_time_arg_val(27);
    constexpr uint32_t cb_id_sharded_act = get_compile_time_arg_val(28);
    constexpr uint32_t cb_reader_indices = get_compile_time_arg_val(29);

    uint32_t i = 0;
    uint32_t noop = get_arg_val<uint32_t>(i);
    i += 1;

    if (noop) {
        return;
    }

    // LOOP TO FILL READER OFFSETS
    /* We can add another loop to read chunks of a stick as well.
     * - Duplicate reader_offset for same stick X times (window_inner must be 1)
     * - New loop between outer and inner that loops X times reading from same stick
     * - Read conv_act_c_read_bytes / X each time
     * - Update l1_write_addr_act by conv_act_c_read_bytes
     */
    uint32_t reader_offsets[weight_size_w * weight_size_h];
    uint32_t reader_offset = 0;  // Constant offset for each pixel within filter window
    uint32_t reader_offset_idx = 0;
    for (uint32_t channel_stick_h = 0; channel_stick_h < weight_size_h; channel_stick_h++) {
        uint32_t reader_offset_row = reader_offset;
        for (uint32_t channel_stick_w = 0; channel_stick_w < weight_size_w; channel_stick_w++) {
            reader_offsets[reader_offset_idx++] = reader_offset_row;
            reader_offset_row += dilation_w;
        }
        // -1 to go back to previous reader_offset
        reader_offset += (dilation_h * conv_act_size_w_padded);
    }

    dump(conv_act_size_w_padded);
    dump(DILATION_W);
    dump(dilation_h);
    constexpr uint32_t act_block_h_datums_read = act_block_h_datums / 2;  // Extra /2 because of packed uint16 reads
    constexpr uint32_t act_block_num_tiles_read = act_block_num_tiles;

    // LOOP TO FILL READER INDICES
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices));

    uint32_t reader_idx = 0;

    // TODO: need to make the read coalescing optimization cleaner
    // pass coalesce_window_inner_reads as a compile time arg and num_coalesced_reads so we can constexpr the if
    // currently works for the case of num_coalesced_reads == weight_size_w since these reads are contiguous on both
    // src/dst side we check if window_inner == weight_size_w to make sure coalescing is legal along full window_inner
    // so the loop can be removed
    constexpr bool coalesce_window_inner_reads = true;
    constexpr uint32_t num_coalesced_reads = weight_size_w;
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 100) ? num_coalesced_reads * conv_act_c_read_bytes : conv_act_c_read_bytes);
    // the conditional selecting between coalescing and no-colescing must be constexpr to that compiler can optimized
    // the other path away this has shown to be a big perf win

    // coalesce reads along weight_size_w
    reader_offset_idx = 0;
    uint32_t act_l1_offset = 0;
    uint32_t act_l1_read_addr = get_read_ptr(cb_id_sharded_act);

    static_assert(coalesced_read_bytes <= NOC_MAX_BURST_SIZE);
    // set_state uses just x/y from the get_noc_addr, addr is ignored
    noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), coalesced_read_bytes);

    constexpr uint32_t stride_w_bytes = dilation_w * conv_act_c_read_bytes;

    uint32_t start_reader_idx = 1;
    dump(coalesced_read_bytes / 2);

    uint32_t offsets[2 * weight_size_w];
    uint32_t packed_reader_indices_sz = packed_reader_indices_ptr[0] & 0xFFFF;
    print_bf16_pages(act_l1_read_addr, conv_act_c_read_bytes / 2, (packed_reader_indices_sz + 2) * window_outer);
    dump(packed_reader_indices_sz);
    dump(conv_act_c_read_bytes);
    uint32_t is_cont = packed_reader_indices_ptr[0] >> 16;
    dump(is_cont);
    dump(act_block_w_extra_align_bytes);
    for (uint32_t bh = 0; bh < act_num_blocks_h; bh++) {
        uint32_t cont_offset = 0;
        for (uint32_t outer = 0; outer < window_outer; outer++) {
            // Reset reader_idx to finish act_block_h_datums
            // reader_idx = start_reader_idx;
            reader_idx = start_reader_idx + outer * (packed_reader_indices_sz / 2);

            cb_reserve_back(cb_id_act, act_block_num_tiles_read);
            uint32_t l1_write_addr_act = get_write_ptr(cb_id_act);
            // uint32_t reader_offset = act_l1_read_addr + (reader_offsets[reader_offset_idx] * conv_act_c_read_bytes);
            // uint32_t elems_to_jump = min(conv_act_size_w_padded * dilation_h, (cont_offset));
            // cont_offset = 0;
            // uint32_t reader_offset = act_l1_read_addr + outer * (12 * conv_act_c_read_bytes);
            uint32_t reader_offset = act_l1_read_addr;
            // dump(reader_offset);
            // #pragma GCC unroll 4 // unroll didn't help, but act_block_h_datums (loop bound) being const does help
            uint32_t act_block_h_datums_read_curr =
                bh == act_num_blocks_h - 1 ? act_block_h_datums_read_last_block : act_block_h_datums_read;

            // uint32_t prev_idx = 0xFFFFFFFF;
            dump(act_block_h_datums_read_curr);
            dump(coalesced_read_bytes);
            uint32_t two_reader_indices;
            uint32_t reader_idx_1 = 0;
            uint32_t reader_idx_2 = 0;
            for (uint32_t bhd = 0; bhd < act_block_h_datums_read_curr; bhd++) {
                // local read from reader_index + reader_offset;
                if (is_cont) {
                    two_reader_indices = packed_reader_indices_ptr[reader_idx];
                    reader_idx_1 = two_reader_indices & 0xffff;
                    reader_idx_2 = two_reader_indices >> 16;
                    act_l1_offset = reader_offset + (reader_idx_1 * conv_act_c_read_bytes);
                    for (uint32_t i = 0; i < window_inner; i++) {
                        noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                        l1_write_addr_act += (coalesced_read_bytes);
                        act_l1_offset += stride_w_bytes;
                    }
                    l1_write_addr_act += act_block_w_extra_align_bytes;
                    act_l1_offset = reader_offset + (reader_idx_2 * conv_act_c_read_bytes);
                    for (uint32_t i = 0; i < window_inner; i++) {
                        noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                        l1_write_addr_act += (coalesced_read_bytes);
                        act_l1_offset += stride_w_bytes;
                    }
                    l1_write_addr_act += act_block_w_extra_align_bytes;
                    DPRINT << "readeer_idx: " << reader_idx << "reader_idx_1: " << reader_idx_1
                           << " reader_idx_2: " << reader_idx_2 << ENDL();
                    reader_idx++;
                } else {
                    for (uint32_t i = 0; i < (window_inner * 2); i += 2) {
                        two_reader_indices = packed_reader_indices_ptr[reader_idx];
                        reader_idx_1 = two_reader_indices & 0xffff;
                        reader_idx_2 = two_reader_indices >> 16;
                        act_l1_offset = reader_offset + (reader_idx_1 * conv_act_c_read_bytes);
                        noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                        l1_write_addr_act += (coalesced_read_bytes + act_block_w_extra_align_bytes);
                        act_l1_offset = reader_offset + (reader_idx_2 * conv_act_c_read_bytes);
                        noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                        l1_write_addr_act += (coalesced_read_bytes + act_block_w_extra_align_bytes);
                        // DPRINT << "reader_idx_1: " << reader_idx_1 << " reader_idx_2: " << reader_idx_2 << ENDL();
                        reader_idx++;
                    }
                }

                // dump(reader_idx_1);
                // dump(reader_idx_2);
                // if (prev_idx != 0xFFFFFFFF && prev_idx != (reader_idx_1 - 1) && reader_idx_2 != 0) {
                //     cont_offset += dilation_w * (window_inner - 1);
                //     DPRINT << "cont_offset: " << cont_offset << ENDL();
                // }
                // prev_idx = reader_idx_1;
                // if (prev_idx != (reader_idx_2 - 1) && reader_idx_2 != 0) {
                //     cont_offset += dilation_w * (window_inner - 1);
                //     DPRINT << "cont_offset: " << cont_offset << ENDL();
                // }
                // prev_idx = reader_idx_2;
#if DILATION_W == 100
                act_l1_offset = reader_offset + (reader_idx_1 * conv_act_c_read_bytes);
                noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                // print_bf16_pages(l1_write_addr_act, coalesced_read_bytes / 2, 1);
                l1_write_addr_act += (coalesced_read_bytes + act_block_w_extra_align_bytes);

                act_l1_offset = reader_offset + (reader_idx_2 * conv_act_c_read_bytes);
                noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                // print_bf16_pages(l1_write_addr_act, coalesced_read_bytes / 2, 1);
                l1_write_addr_act += (coalesced_read_bytes + act_block_w_extra_align_bytes);
#else
                // act_l1_offset = reader_offset + (reader_idx_1 * conv_act_c_read_bytes);
                // for (uint32_t inner = 0; inner < weight_size_w; inner++) {
                //     noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                //     l1_write_addr_act += conv_act_c_read_bytes;
                //     act_l1_offset += reader_offset + reader_idx1 * conv_act_c_read_bytes;
                //     reader_idx1++;
                // }
                noc_async_read_barrier();
                print_bf16_pages(
                    l1_write_addr_act - (2 * conv_act_c_read_bytes * window_inner),
                    (conv_act_c_read_bytes / 2) * window_inner,
                    2);

                // act_l1_offset = reader_offset + (reader_idx_1 * conv_act_c_read_bytes);
                // for (uint32_t inner = 0; inner < weight_size_w; inner++) {
                //     noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                //     l1_write_addr_act += conv_act_c_read_bytes;
                //     act_l1_offset += reader_offset + reader_idx1 * conv_act_c_read_bytes;
                //     reader_idx1++;
                // }
                // noc_async_read_barrier();
                // print_bf16_pages(
                //     l1_write_addr_act - (conv_act_c_read_bytes * window_inner),
                //     (conv_act_c_read_bytes * window_inner) / 2,
                //     1);
#endif
                // reader_idx++;
                // if (reader_idx_2 != 0) {
                //     cont_offset += 2;
                // }
            }
            // cont_offset += dilation_w * (window_inner - 1);
            // dump(cont_offset);
            // dump(cont_offset);
            noc_async_read_barrier();

            cb_push_back(cb_id_act, act_block_num_tiles_read);

            reader_offset_idx += window_inner;
        }
        reader_offset_idx = 0;

        start_reader_idx += 16;
        dump(start_reader_idx);
#ifdef SPLIT_READER
        start_reader_idx += act_block_h_datums_second_reader_read;
#endif
    }
    noc_async_write_barrier();
}
