// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id_out = tt::CBIndex::c_16;
    uint32_t tile_bytes = get_tile_size(cb_id_out);

    const DataFormat data_format = get_dataformat(cb_id_out);
    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    cb_wait_front(cb_id_out, 1);

    auto l1_read_addr = get_read_ptr(cb_id_out);
    noc_async_write_tile(0, s, l1_read_addr);
    noc_async_write_barrier();

    cb_pop_front(cb_id_out, 1);
}
