// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    constexpr auto cb_scalar = tt::CBIndex::c_2;

    cb_reserve_back(cb_scalar, 1);
    float* ptr = reinterpret_cast<float*>(get_write_ptr(cb_scalar));
    ptr[0] = 2.013671875f;
    cb_push_back(cb_scalar, 1);

    cb_reserve_back(cb_scalar, 1);
    ptr = reinterpret_cast<float*>(get_write_ptr(cb_scalar));
    ptr[0] = 2.013671875f;
    cb_push_back(cb_scalar, 1);

}
