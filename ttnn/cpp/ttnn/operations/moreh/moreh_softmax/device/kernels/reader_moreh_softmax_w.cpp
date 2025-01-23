// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_scaler = tt::CBIndex::c_1;
    constexpr auto cb_fp32_scaler = tt::CBIndex::c_2;

    union Scalar {
        float f;
        uint32_t u;
    } s;

    {
        cb_reserve_back(cb_in0, 1);
        float* ptr = reinterpret_cast<float*>(get_write_ptr(cb_in0));
        memset(ptr, 0, 1024 * sizeof(float));
        ptr[0] = 1.265625f;
        ptr[1] = 1.7890625;
        ptr[2] = 0.271484375;
        ptr[3] = 1.671875;

        cb_push_back(cb_in0, 1);
    }

    {
        cb_reserve_back(cb_scaler, 1);
        uint16_t* ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_scaler));
        memset(ptr, 0, 1024 * sizeof(uint16_t));
        s.f = 1.0f;
        ptr[0] = static_cast<uint16_t>(s.u >> 16);
        ptr[1] = static_cast<uint16_t>(s.u >> 16);
        ptr[2] = static_cast<uint16_t>(s.u >> 16);
        ptr[3] = static_cast<uint16_t>(s.u >> 16);
        cb_push_back(cb_scaler, 1);
    }

    {
        cb_reserve_back(cb_fp32_scaler, 1);
        float* ptr = reinterpret_cast<float*>(get_write_ptr(cb_fp32_scaler));
        memset(ptr, 0, 1024 * sizeof(float));
        ptr[0] = 1.0;
        ptr[1] = 1.0;
        ptr[2] = 1.0;
        ptr[3] = 1.0;
        cb_push_back(cb_fp32_scaler, 1);
    }
}
