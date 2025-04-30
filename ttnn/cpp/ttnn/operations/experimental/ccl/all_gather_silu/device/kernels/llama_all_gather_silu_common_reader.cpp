// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t cb0_id = get_compile_time_arg_val(1);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(2);
constexpr uint32_t cb_out_id = get_compile_time_arg_val(3);

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    address_t buffer_address0 = get_arg_val<address_t>(arg_idx++);
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t first_core_tile_start_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;

    uint32_t start_tile = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_transactions = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t signal_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    tt_l1_ptr uint32_t* core_noc_x_reshard = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_transactions;
    tt_l1_ptr uint32_t* core_noc_y_reshard = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_transactions;
    tt_l1_ptr uint32_t* size_transaction = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_transactions;
    uint32_t shift_value = 0;
    volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);

    // interleaved addrgen

    uint32_t tiles_read = 0;
    uint32_t shard_tile_id = first_core_tile_start_offset;
    uint32_t core_id = 0;

    while (tiles_read < num_tiles_to_read) {
        uint32_t num_tiles_to_read_this_core =
            std::min(num_tiles_per_core - shard_tile_id, num_tiles_to_read - tiles_read);
        cb_reserve_back(cb0_id, num_tiles_to_read_this_core);
        const uint32_t l1_write_addr = get_write_ptr(cb0_id);
        uint64_t read_addr = get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], tensor_address0);
        read_addr += shard_tile_id * tensor0_page_size;

        noc_async_read(read_addr, l1_write_addr, num_tiles_to_read_this_core * tensor0_page_size);
        noc_async_read_barrier();

        cb_push_back(cb0_id, num_tiles_to_read_this_core);
        tiles_read += num_tiles_to_read_this_core;
        shard_tile_id = 0;
        core_id++;
    }

    if (size_transaction[0] != 4) {
        shift_value += (4 - size_transaction[0]) * tensor0_page_size;
    }
    uint32_t l1_write_addr_reshard = get_write_ptr(cb_out_id);
    noc_semaphore_wait(signal_semaphore_addr_ptr, 1);
    noc_semaphore_set(signal_semaphore_addr_ptr, 0);

    for (uint32_t i = 0; i < num_transactions; i++) {
        uint64_t read_addr_reshard = get_noc_addr(core_noc_x_reshard[i], core_noc_y_reshard[i], buffer_address0);
        read_addr_reshard += shift_value;

        noc_async_read(read_addr_reshard, l1_write_addr_reshard, size_transaction[i] * tensor0_page_size);

        l1_write_addr_reshard += (size_transaction[i] * tensor0_page_size);
        shift_value = 0;
    }
    noc_async_read_barrier();
}
