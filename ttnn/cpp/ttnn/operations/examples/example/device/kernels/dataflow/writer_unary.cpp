// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t num_tiles = 256;
    uint32_t start_id = 0;

    constexpr uint32_t cb_out1 = 1;
    constexpr uint32_t block_size = 256;

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; i += block_size) {
        cb_wait_front(cb_out1, block_size);
        cb_pop_front(cb_out1, block_size);
    }
}
