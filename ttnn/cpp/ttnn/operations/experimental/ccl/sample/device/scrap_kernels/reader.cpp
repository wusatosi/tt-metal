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

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    DPRINT << "HELLO FROM READER\n";
    size_t arg_idx = 0;
    uint32_t tensor_address0 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t receiver_semaphore_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore_address);

    DPRINT << "WAITING ON SEMAPHORE \n";
    // noc_semaphore_wait(signal_semaphore_addr_ptr, 1);
    while (1) {
        DPRINT << "SEM addr: " << (uint32_t)signal_semaphore_addr_ptr
               << " VALUE: " << (uint32_t)*signal_semaphore_addr_ptr << "\n";
        for (int i = 0; i < 1000000000; i++) {
        }
    }
    DPRINT << "!!!!! READING FROM ETH !!!!\n";
    DPRINT << "DONE reader.\n";
}
