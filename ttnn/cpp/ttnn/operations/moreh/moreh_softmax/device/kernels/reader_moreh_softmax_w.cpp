// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t scaler = get_arg_val<uint32_t>(4);
    uint32_t mask_w = get_arg_val<uint32_t>(5);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_mask = tt::CBIndex::c_1;
    constexpr auto cb_scaler = tt::CBIndex::c_2;

    uint32_t l1_write_addr_in;

    uint32_t src_in_tile_bytes = get_tile_size(cb_in);
    const DataFormat src_in_data_format = get_dataformat(cb_in);

    constexpr bool in_is_dram = get_compile_time_arg_val(0) == 1;

    const InterleavedAddrGenFast<in_is_dram> src_in = {
        .bank_base_address = src_addr, .page_size = src_in_tile_bytes, .data_format = src_in_data_format};

    generate_bcast_scaler(cb_scaler, scaler);
    generate_mask_w(cb_mask, mask_w);


    cb_reserve_back(cb_in, 1);

    l1_write_addr_in = get_write_ptr(cb_in);
    noc_async_read_tile(0, src_in, l1_write_addr_in);
    noc_async_read_barrier();

    cb_push_back(cb_in, 1);

}
