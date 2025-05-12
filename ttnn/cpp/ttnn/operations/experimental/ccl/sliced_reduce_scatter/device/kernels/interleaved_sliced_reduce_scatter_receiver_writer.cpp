// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

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
constexpr uint32_t compute_output_cb_id = get_compile_time_arg_val(7);
constexpr uint32_t reader_output_cb_id = get_compile_time_arg_val(8);
constexpr uint32_t sync_cb_id = get_compile_time_arg_val(9);
void kernel_main() {
    size_t arg_idx = 0;

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

    constexpr uint32_t out_row_start = 0;
    constexpr uint32_t out_col_start = 0;
    uint32_t out_row_end = out_row_start + input_shard_row_tiles;
    uint32_t out_col_end = out_col_start + input_shard_col_tiles;

    constexpr bool is_dram = true;  // TODO: CT arg
    auto output_tensor_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = output_buffer_addr,
        .page_size = page_size,
        .data_format = get_dataformat(reader_output_cb_id)};

    // Copy from intermediate buffer to output buffer
    // Compute where remote sender dumped data into intermediate buffer.
    // Should follow same logic as sender writer.

    // DPRINT << "out_row_start " << out_row_start << ENDL();
    // DPRINT << "out_row_end " << out_row_end << ENDL();
    // DPRINT << "out_col_start " << out_col_start << ENDL();
    // DPRINT << "out_col_end " << out_col_end << ENDL();
    // DPRINT << "out_col_tiles " << out_col_tiles << ENDL();
    // DPRINT << "input_shard_row_tiles " << input_shard_row_tiles << ENDL();
    // DPRINT << "input_shard_col_tiles " << input_shard_col_tiles << ENDL();

    for (uint32_t device_id = 0; device_id < ring_size; device_id++) {
        // if (device_id > 1) {
        //     continue;
        // }
        uint32_t output_cb = device_id > 0 ? compute_output_cb_id : reader_output_cb_id;
        // if (device_id >= 2) {
        //     continue;
        // }
        uint32_t tiles_written = 0;
        for (uint32_t out_row_id = out_row_start; out_row_id < out_row_end; out_row_id++) {
            for (uint32_t out_col_id = out_col_start; out_col_id < out_col_end; out_col_id += num_pages_per_packet) {
                cb_wait_front(output_cb, num_pages_per_packet);
                uint32_t pages_acked = get_cb_tiles_acked_ptr(output_cb)[0];
                uint32_t pages_received_ptr = (uint32_t)get_cb_tiles_received_ptr(output_cb);
                uint32_t received_data = reg_read(pages_received_ptr);
                uint16_t pages_received = ((uint16_t)received_data) - pages_acked;
                DPRINT << "WRITER: pages_acked: " << pages_acked << " received_data: " << received_data
                       << " pages_received: " << pages_received << ENDL();
                // DPRINT << "wait front output cb col_tile_id " << out_col_id << ENDL();
                size_t l1_read_addr = get_read_ptr(output_cb);
                uint32_t num_pages_to_read = std::min(out_col_end - out_col_id, num_pages_per_packet);

                constexpr uint32_t contig_pages_advanced = 1;  // always write 1 tile at a time to output
                for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                    uint32_t col_tile = out_col_id + j;
                    uint32_t tile_id = out_row_id * out_col_tiles + col_tile;
                    noc_async_write_tile(tile_id, output_tensor_addrgen, l1_read_addr);
                    tiles_written++;
                    DPRINT << "d" << device_id << " WRITER WRITING TILE " << tile_id << " to output_cb " << output_cb
                           << ENDL();
                    l1_read_addr += page_size;
                }
                noc_async_write_barrier();

                cb_pop_front(output_cb, num_pages_per_packet);
                if (device_id > 0) {
                    cb_wait_front(sync_cb_id, 1);
                    cb_pop_front(sync_cb_id, 1);
                }
            }
        }
        DPRINT << "Finished writing shard " << device_id << ENDL();
        // DPRINT << "tiles_written " << tiles_written << ENDL();
        // DPRINT << "Done writing shard " << device_id << ENDL();
    }
    noc_async_write_barrier();
}
