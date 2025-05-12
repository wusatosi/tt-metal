// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <cstddef>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;

constexpr uint32_t cb_index = get_compile_time_arg_val(0);

// constexpr uint32_t dst_cb_index = get_compile_time_arg_val(1);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    DPRINT << "HELLO FROM READER\n";
    size_t arg_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t receiver_semaphore_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_num_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t device_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t device_order = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tensor_page_size = get_arg_val<uint32_t>(arg_idx++);

    auto tensor0_addrgen = InterleavedAddrGenFast<true>{
        .bank_base_address = input_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_index)};

    auto output_tensor_addrgen = InterleavedAddrGenFast<true>{
        .bank_base_address = output_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_index)};

    uint32_t tile_id = 0;
    for (; tile_id < input_num_tiles; tile_id++) {
        cb_reserve_back(cb_index, 1);
        tensor0_addrgen.noc_async_read_tile(tile_id, get_write_ptr(cb_index));
        noc_async_read_barrier();
        cb_push_back(cb_index, 1);
    }

    tile_id = 0;
    for (; tile_id < input_num_tiles; tile_id++) {
        cb_reserve_back(cb_index, 1);
        tensor0_addrgen.noc_async_read_tile(tile_id, get_write_ptr(cb_index));
        noc_async_read_barrier();
        cb_push_back(cb_index, 1);
    }

    volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore_address);

    noc_semaphore_wait(signal_semaphore_addr_ptr, 1);

    DPRINT << "DONE READER\n";
}
