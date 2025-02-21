// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = 0;
    cb_push_back(cb_in0, 128);

    constexpr uint32_t cb_in1 = 1;
    cb_push_back(cb_in1, 1);

    constexpr uint32_t cb_in2 = 2;
    cb_push_back(cb_in2, 1);
}
