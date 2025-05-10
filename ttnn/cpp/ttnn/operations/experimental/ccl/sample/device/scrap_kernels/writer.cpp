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
    uint32_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);

    int x = 1;
    int y = 0;
    int maxx = 4;
    int maxy = 4;
    uint64_t out_ready_sem_noc_addr_in_pkt =
        get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr, 0);
    DPRINT << "NOC Address parts: " << x << ", " << y << ", " << out_ready_sem_bank_addr << "\n";
    noc_semaphore_inc(out_ready_sem_noc_addr_in_pkt, 2, 0);
    noc_async_write_barrier();

    DPRINT << "Done SCRAP writer.\n";
}
