/**
void fabric_async_write(uint32_t routing,   // the network plane to use for this transaction
    uint32_t src_addr,  // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t size  // number of bytes to write to remote destination
**/

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t rt_index = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(rt_index++);
    const uint32_t dst_addr = get_arg_val<uint32_t>(rt_index++);
    const uint32_t dst_mesh_id = get_arg_val<uint32_t>(rt_index++);
    const uint32_t dst_dev_id = get_arg_val<uint32_t>(rt_index++);
    const uint32_t page_size_bytes = get_arg_val<uint32_t>(rt_index++);
    const uint32_t number_of_pages = get_arg_val<uint32_t>(rt_index++);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGen<src_is_dram> s = {.bank_base_address = src_addr, .page_size = page_size_bytes};
    const InterleavedAddrGen<dst_is_dram> d = {.bank_base_address = dst_addr, .page_size = page_size_bytes};

    for (uint32_t i = 0; i < number_of_pages; ++i) {
        const uint64_t source_address = get_noc_addr(i, s);
        const uint64_t dest_address = get_noc_addr(i, d);
        fabric_async_write(source_address, dst_mesh_id, dst_dev_id, dest_address);
    }
}
