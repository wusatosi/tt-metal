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
        ptr[0] = 1.0039063;
        cb_push_back(cb_in0, 1);
    }
}
