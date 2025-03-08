// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    // // 16 us
    // for (uint32_t i = 0; i < 5; i++) {
    //     for (uint32_t j = 0; j < 1000; j++) {
    //         asm volatile("nop");
    //     }
    // }

    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t total_num_reduction_tiles = get_compile_time_arg_val(1);

    if (total_num_reduction_tiles == 4) {
        // 5 us
        for (uint32_t i = 0; i < 1650; i++) {
            asm volatile("nop");
        }
    } else {
        // 13 us
        for (uint32_t i = 0; i < 4000; i++) {
            asm volatile("nop");
        }
    }

    return;

    // runtime args
    size_t arg_idx = 0;
    const uint32_t signal_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);

    // 1. Wait for signal from All-Gather worker
    noc_semaphore_wait(signal_semaphore_addr_ptr, VALID);
    noc_semaphore_set(signal_semaphore_addr_ptr, 0);

    // 2. Signal compute kernel to start processing
    cb_push_back(cb_id, total_num_reduction_tiles);
}
