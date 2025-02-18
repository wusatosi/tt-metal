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
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(1);
    constexpr uint32_t total_num_reduction_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t ring_size = get_compile_time_arg_val(3);

    // runtime args
    size_t arg_idx = 0;
    const uint32_t signal_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t link_worker_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t link_worker_y = get_arg_val<uint32_t>(arg_idx++);

    volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);

    // 1. Wait for signal from All-Gather worker
    noc_semaphore_wait(signal_semaphore_addr_ptr, VALID);
    noc_semaphore_set(signal_semaphore_addr_ptr, 0);

    // 2. Signal compute kernel to start processing
    cb_push_back(cb_id, total_num_reduction_tiles);
    DPRINT << TSLICE(cb_id, 1, SliceRange::hw041(), false, false) << ENDL();

    // cb_wait_front(cb_id_out, total_num_reduction_tiles / ring_size);
    // noc_semaphore_inc(get_noc_addr(link_worker_x, link_worker_y, signal_semaphore_addr), 1);
}
