// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
namespace NAMESPACE {
void MAIN {
    // Define all compile-time arguments at the beginning
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t accumulator_cb = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb = get_compile_time_arg_val(2);
    constexpr uint32_t num_packets = get_compile_time_arg_val(3);
    constexpr uint32_t tiles_per_packet = get_compile_time_arg_val(4);
    constexpr uint32_t ring_size = get_compile_time_arg_val(5);

    for (uint32_t i = 0; i < ring_size - 1; ++i) {
        // Initialize binary operations - use the same constants consistently
        binary_op_init_common(input_cb_id, accumulator_cb, output_cb);
        add_tiles_init(input_cb_id, accumulator_cb, false);

        // Wait for input data once before beginning processing
        for (uint32_t packet_id = 0; packet_id < num_packets; packet_id++) {
            cb_wait_front(input_cb_id, tiles_per_packet);
            // Reserve output space once before processing
            cb_wait_front(accumulator_cb, tiles_per_packet);
            cb_reserve_back(output_cb, tiles_per_packet);
            acquire_dst();
            for (uint32_t tile_id = 0; tile_id < tiles_per_packet; tile_id++) {
                add_tiles(input_cb_id, accumulator_cb, tile_id, tile_id, tile_id);
                pack_tile(tile_id, output_cb);
            }
            release_dst();
            cb_pop_front(input_cb_id, tiles_per_packet);

            cb_pop_front(accumulator_cb, tiles_per_packet);
            cb_push_back(output_cb, tiles_per_packet);
        }
    }
}
}  // namespace NAMESPACE
