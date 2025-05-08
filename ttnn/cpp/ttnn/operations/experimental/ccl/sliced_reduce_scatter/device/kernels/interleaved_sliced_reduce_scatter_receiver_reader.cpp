// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_ring_id = get_compile_time_arg_val(0);
constexpr uint32_t ring_size = get_compile_time_arg_val(1);
constexpr uint32_t num_pages_per_packet = get_compile_time_arg_val(2);
constexpr uint32_t chunk_granularity = get_compile_time_arg_val(3);
constexpr uint32_t chunk_num_tiles = get_compile_time_arg_val(4);
constexpr uint32_t num_chunks_per_shard = get_compile_time_arg_val(5);
constexpr uint32_t page_size = get_compile_time_arg_val(6);
constexpr uint32_t input_cb_id = get_compile_time_arg_val(7);
constexpr uint32_t accumulator_cb_id = get_compile_time_arg_val(8);
constexpr uint32_t output_cb_id = get_compile_time_arg_val(9);
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(10);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(11);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(12);

// TODO: CT args
constexpr uint32_t N_DRAM_BANKS = 12;
constexpr uint32_t NUM_SENDERS = ring_size - 1;

void kernel_main() {
    size_t arg_idx = 0;

    address_t intermediate_buffer_addr = get_arg_val<address_t>(arg_idx++);
    address_t input_buffer_addr = get_arg_val<address_t>(arg_idx++);
    address_t output_buffer_addr = get_arg_val<address_t>(arg_idx++);
    uint32_t in_row_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t in_col_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_row_device_stride = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_col_device_stride = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_shard_row_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_shard_col_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_row_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_col_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_pages_per_packet = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* global_semaphore_addr = (tt_l1_ptr uint32_t*)get_arg_addr(arg_idx);
    arg_idx += ring_size;

    constexpr uint32_t out_row_start = 0;
    constexpr uint32_t out_col_start = 0;
    uint32_t out_row_end = out_row_start + input_shard_row_tiles;
    uint32_t out_col_end = out_col_start + input_shard_col_tiles;

    constexpr bool is_dram = true;  // TODO: CT arg
    auto input_tensor_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = input_buffer_addr, .page_size = page_size, .data_format = get_dataformat(input_cb_id)};
    auto intermediate_tensor_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = intermediate_buffer_addr,
        .page_size = page_size,
        .data_format = get_dataformat(input_cb_id)};
    auto output_tensor_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = output_buffer_addr, .page_size = page_size, .data_format = get_dataformat(input_cb_id)};

    bool my_cur_is_forward = num_targets_forward_direction > num_targets_backward_direction;
    uint32_t hop_count = 0;
    uint32_t dst_ring_id;
    for (uint32_t i = 0; i < ring_size; ++i) {
        // This is the inverse of the sender reader logic
        const bool do_reduce = i != 0;
        hop_count++;

        if (my_cur_is_forward) {
            dst_ring_id = (my_ring_id + hop_count) % ring_size;
        } else {
            dst_ring_id = (my_ring_id - hop_count + ring_size) % ring_size;
        }
        if (hop_count == ring_size / 2) {
            hop_count = 0;
        } else {
            my_cur_is_forward = !my_cur_is_forward;
        }

        if (i == ring_size - 1) {
            // TODO: synchronize with matmul and ensure all other receivers have finished
            // Follows same logic as sender reader for local copy.
            uint32_t shard_row_start_id = my_ring_id * input_row_device_stride;
            uint32_t shard_col_start_id = my_ring_id * input_col_device_stride;
            uint32_t shard_row_end_id = shard_row_start_id + input_shard_row_tiles;
            uint32_t shard_col_end_id = shard_col_start_id + input_shard_col_tiles;

            DPRINT << "starting local copy" << ENDL();
            for (uint32_t row_tile_id = 0; row_tile_id < input_shard_row_tiles; row_tile_id++) {
                for (uint32_t col_tile_id = 0; col_tile_id < input_shard_col_tiles;
                     col_tile_id += num_pages_per_packet) {
                    uint32_t intermed_row_id = row_tile_id + shard_row_start_id;
                    uint32_t intermed_col_id = col_tile_id + shard_col_start_id;
                    uint32_t intermed_tile_id = intermed_row_id * in_col_tiles + intermed_col_id;
                    // DPRINT << "tile_id: " << tile_id << "\n";

                    // Local read
                    cb_reserve_back(input_cb_id, num_pages_per_packet);
                    // DPRINT << "reserve input cb col_tile_id " << col_tile_id << ENDL();
                    const uint32_t l1_write_addr_base = get_write_ptr(input_cb_id);
                    uint32_t l1_write_addr = l1_write_addr_base;

                    uint32_t num_pages_to_read = std::min(shard_col_end_id - intermed_col_id, num_pages_per_packet);
                    for (uint32_t j = 0; j < num_pages_to_read; j++) {
                        noc_async_read_tile(intermed_tile_id, input_tensor_addrgen, l1_write_addr);
                        l1_write_addr += page_size;
                        intermed_tile_id++;
                    }

                    noc_async_read_barrier();
                    cb_push_back(input_cb_id, num_pages_per_packet);

                    // Accumulator read
                    cb_reserve_back(accumulator_cb_id, num_pages_per_packet);
                    // DPRINT << "reserve accumulator cb col_tile_id " << col_tile_id << ENDL();
                    uint32_t accumulator_l1_write_addr = get_write_ptr(accumulator_cb_id);

                    uint32_t tile_id = row_tile_id * in_col_tiles + col_tile_id;
                    for (uint32_t j = 0; j < num_pages_to_read; j++) {
                        noc_async_read_tile(tile_id, output_tensor_addrgen, accumulator_l1_write_addr);
                        accumulator_l1_write_addr += page_size;
                        tile_id++;
                    }

                    noc_async_read_barrier();
                    cb_push_back(accumulator_cb_id, num_pages_per_packet);
                }
            }
            DPRINT << "local copy done" << ENDL();
        } else {
            // Copy from intermediate buffer to output buffer
            // Compute where remote sender dumped data into intermediate buffer.
            // Should follow same logic as sender writer.
            DPRINT << "starting remote copy" << ENDL();
            // TODO: If first receiver, write directly to output cb. otherwise, intermediate cb
            uint32_t cb_in0 = do_reduce ? input_cb_id : output_cb_id;

            const uint32_t sender_relative_ring_id = (dst_ring_id < my_ring_id) ? dst_ring_id : dst_ring_id - 1;

            volatile tt_l1_ptr uint32_t* global_semaphore_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr[dst_ring_id]);
            uint32_t packet_id = 0;

            DPRINT << "receiver reader global_semaphore_addr[dst_ring_id]: "
                   << (uint32_t)global_semaphore_addr[dst_ring_id] << ENDL();

            for (uint32_t out_row_id = out_row_start; out_row_id < out_row_end; out_row_id++) {
                for (uint32_t out_col_id = out_col_start; out_col_id < out_col_end;
                     out_col_id += num_pages_per_packet) {
                    cb_reserve_back(cb_in0, num_pages_per_packet);
                    size_t l1_write_addr = get_write_ptr(cb_in0);
                    uint32_t num_pages_to_read = std::min(out_col_end - out_col_id, num_pages_per_packet);

                    constexpr uint32_t payload_size_bytes = contig_pages_advanced * page_size;

                    // Calculate which chunk we need and wait for it
                    uint32_t current_chunk_id = packet_id / chunk_granularity;
                    uint32_t wait_chunk_id = current_chunk_id + 1;  // Chunks are 1-based
                    // Ensure that current chunk has been sent
                    while (*global_semaphore_ptr < wait_chunk_id);
                    // DPRINT << "Got chunk " << wait_chunk_id << ENDL();

                    for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                        uint32_t global_id = sender_relative_ring_id + packet_id * NUM_SENDERS;
                        uint32_t first_id = (global_id % N_DRAM_BANKS) + 2 * N_DRAM_BANKS * (global_id / N_DRAM_BANKS);
                        uint64_t packet_addr =
                            get_noc_addr(first_id, intermediate_tensor_addrgen, 0 /*offset*/, 0 /*noc_id*/);

                        noc_async_read(packet_addr, l1_write_addr, payload_size_bytes);
                        l1_write_addr += payload_size_bytes;
                        packet_id++;
                    }
                    noc_async_read_barrier();

                    cb_push_back(cb_in0, num_pages_per_packet);

                    if (do_reduce) {
                        // read from output tensor into accumulator_cb
                        cb_reserve_back(accumulator_cb_id, num_pages_per_packet);
                        uint32_t accumulator_l1_write_addr = get_write_ptr(accumulator_cb_id);
                        DPRINT << "reading accumulator row, col: " << out_row_id << ", " << out_col_id << ENDL();
                        uint32_t tile_id = out_row_id * in_col_tiles + out_col_id;
                        for (uint32_t j = 0; j < num_pages_to_read; j++) {
                            noc_async_read_tile(tile_id, output_tensor_addrgen, accumulator_l1_write_addr);
                            accumulator_l1_write_addr += page_size;
                            tile_id++;
                        }

                        noc_async_read_barrier();
                        cb_push_back(accumulator_cb_id, num_pages_per_packet);
                    }
                }
            }
            DPRINT << "remote copy done" << ENDL();
        }

        // Reset global semaphore
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr[dst_ring_id]) = 0;
        // DPRINT << "reset done\n";
    }
}
