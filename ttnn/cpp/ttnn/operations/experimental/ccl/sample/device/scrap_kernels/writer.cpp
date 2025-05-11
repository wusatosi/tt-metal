// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <cstddef>
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;

constexpr uint32_t cb_to_allgather_writer = get_compile_time_arg_val(0);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    uint32_t reserved_packet_header_cb_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tensor_address0 = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_bank_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t device_id = get_arg_val<uint32_t>(arg_idx++);
    DPRINT << "Device ID: " << device_id << "\n";
    if (device_id == 2) {
        int maxx = 6;
        int maxy = 6;
        for (int x = 0; x <= maxx; x++) {
            for (int y = 0; y <= maxy; y++) {
                uint64_t out_ready_sem_noc_addr_in_pkt = get_noc_addr(x, y, out_ready_sem_bank_addr, 0);
                DPRINT << "NOC Address parts: " << x << ", " << y << ", " << out_ready_sem_bank_addr << "\n";
                noc_semaphore_inc(out_ready_sem_noc_addr_in_pkt, 1, 0);
                noc_async_write_barrier();
            }
        }
    }

    DPRINT << "Done SCRAP writer.\n";
}
