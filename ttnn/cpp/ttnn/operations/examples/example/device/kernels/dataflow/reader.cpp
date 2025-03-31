// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    const auto onetile = 1;

    constexpr auto cb_input_fp32 = tt::CBIndex::c_25;

    cb_reserve_back(cb_input_fp32, onetile);
    auto write_ptr = get_write_ptr(cb_input_fp32);
    auto ptr = reinterpret_cast<volatile tt_l1_ptr float*>(write_ptr);
    for (int i = 0; i < 16; i++) {
        ptr[i] = 1;
    }
    cb_push_back(cb_input_fp32, onetile);
}
