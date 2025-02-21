// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/dprint_tile.h"

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t total_num_reduction_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(3);

    if (num_blocks == 8) {
        cb_id = 4;
    }

    // DPRINT << "total_num_reduction_tiles: " << total_num_reduction_tiles << ENDL();

    // runtime args
    size_t arg_idx = 0;
    const uint32_t signal_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t worker_core_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_core_y = get_arg_val<uint32_t>(arg_idx++);

    volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);

    // 1. Wait for signal from All-Gather worker
    noc_semaphore_wait(signal_semaphore_addr_ptr, VALID);
    noc_semaphore_set(signal_semaphore_addr_ptr, 0);

    // if (num_blocks == 4) {
    //     for (uint32_t i = 0; i < total_num_reduction_tiles; i++) {
    //         DPRINT << TSLICE(cb_id, i, SliceRange::h2_w0_32(), false, true) << ENDL();
    //     }
    // }

    // 2. Signal compute kernel to start processing
    cb_push_back(cb_id, total_num_reduction_tiles);

    cb_wait_front(cb_out_id, total_num_reduction_tiles / num_blocks);
    noc_semaphore_inc(get_noc_addr(worker_core_x, worker_core_y, signal_semaphore_addr), 1);
}
