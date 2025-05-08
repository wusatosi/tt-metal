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

constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t cb1_id = get_compile_time_arg_val(1);
constexpr uint32_t data_size = get_compile_time_arg_val(2);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    uint32_t tensor_address0 = get_arg_val<uint32_t>(arg_idx++);

    uint32_t receiver_semaphore_address = get_arg_val<uint32_t>(arg_idx++);
    DPRINT << "HELLO FROM READER\n";

    uint32_t tensor0_page_size = 1088;

    auto dst_addrgen = InterleavedAddrGenFast<false>{
        .bank_base_address = tensor_address0,
        .page_size = tensor0_page_size,
        .data_format = get_dataformat(cb1_id),
    };

    DPRINT << "WAITING ON SEMAPHORE " << receiver_semaphore_address << "\n";
    noc_semaphore_wait((uint32_t*)receiver_semaphore_address, 1);
    DPRINT << "!!!!! READING FROM ETH !!!!\n";
    // // noc_async_write_tile(0, dst_addrgen, get_write_ptr(cb1_id));
    // // DPRINT << "DONE READING FROM ETH\n";
    // // noc_async_write_barrier();
    // noc_semaphore_set((uint32_t*)receiver_semaphore_address, 0);
    DPRINT << "DONE reader.\n";
}
