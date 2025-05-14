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

constexpr uint32_t in_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t is_forward = get_compile_time_arg_val(1);
// constexpr uint32_t dst_cb_index = get_compile_time_arg_val(1);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    DPRINT << "HELLO FROM READER\n";
    size_t arg_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_num_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t device_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t device_order = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tensor_page_size = get_arg_val<uint32_t>(arg_idx++);
    uint32_t local_semaphore = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_semaphore);

    constexpr uint32_t start_tile = 0;
    constexpr uint32_t end_tile = 2;

    auto tensor0_addrgen = InterleavedAddrGenFast<true>{
        .bank_base_address = input_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(in_cb_index)};

    auto output_tensor_addrgen = InterleavedAddrGenFast<true>{
        .bank_base_address = output_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(in_cb_index)};

    uint32_t tile_id = 0;
    for (; tile_id < input_num_tiles; tile_id++) {
        cb_reserve_back(in_cb_index, 1);
        auto tensor_tile_addr = tensor0_addrgen.get_noc_addr(tile_id);
        noc_async_read(tensor_tile_addr, get_write_ptr(in_cb_index), input_tensor_page_size);
        noc_async_read_barrier();
        cb_push_back(in_cb_index, 1);
    }

    int device_direction = is_forward ? -1 : 1;

    // For each device, read the output tensor from previous device into cb for forwarding
    for (int device_iter = 0; device_iter < 7; device_iter++) {
        uint32_t device_to_process = (((device_order + device_iter * device_direction) % 8) + 8) % 8;
        for (tile_id = start_tile; tile_id < end_tile; tile_id++) {
            cb_reserve_back(in_cb_index, 1);
            uint64_t tile_addr = output_tensor_addrgen.get_noc_addr(device_to_process * input_num_tiles + tile_id);
            noc_async_read(tile_addr, get_write_ptr(in_cb_index), input_tensor_page_size);
            noc_async_read_barrier();
            cb_push_back(in_cb_index, 1);
        }
        DPRINT << "READER: Waiting for semaphore value: " << device_iter + 1 << " and got "
               << *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr_ptr) << "\n";
        noc_semaphore_wait(signal_semaphore_addr_ptr, device_iter + 1);
    }

    DPRINT << "DONE READER\n";
}
