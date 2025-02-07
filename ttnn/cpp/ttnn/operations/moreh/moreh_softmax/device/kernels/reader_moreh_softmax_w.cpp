// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    {
        cb_reserve_back(cb_in0, 1);
        float* ptr = reinterpret_cast<float*>(get_write_ptr(cb_in0));
        memset(ptr, 0, 1024 * sizeof(float));
        // this case result is 2.0
        ptr[0] = 1.0019531;
        ptr[16] = 1.0f;

        // this case result is 2.0019531250
        // ptr[0] = 1.0019531;
        // ptr[1] = 1.0f;

        cb_push_back(cb_in0, 1);
    }

    {
        constexpr auto cb_in2 = tt::CBIndex::c_2;
        cb_reserve_back(cb_in2, 1);
        float* ptr = reinterpret_cast<float*>(get_write_ptr(cb_in2));
        memset(ptr, 0, 1024 * sizeof(float));
        for (int i = 0; i < 1024; i++) {
            ptr[i] = 1;
        }
        cb_push_back(cb_in2, 1);
    }
}
