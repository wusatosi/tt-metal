// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_fabric_test_kernels_utils.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_common.hpp"
#include <array>

// clang-format on

void kernel_main() {
    size_t rt_args_idx = 0;
    auto worker_config = tt::tt_fabric::ControllerWorkerConfig::build_from_args(rt_args_idx);

    std::array<volatile tt_l1_ptr PACKET_HEADER_TYPE*, NUM_DIRECTIONS> packet_headers;
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, NUM_DIRECTIONS> fabric_connection_handles;

    uint32_t header_address = worker_config.packet_header_buffer_address;
    uint64_t noc_dest_addr = get_noc_addr(0);
    noc_dest_addr += worker_config.remote_controller_sync_address;
    for (auto i = 0; i < NUM_DIRECTIONS; i++) {
        if (worker_config.hops_count[i] == 0) {
            continue;
        }

        packet_headers[i] = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header_address);
        packet_headers[i]->to_chip_multicast(
            MulticastRoutingCommandHeader{1, static_cast<uint8_t>(worker_config.hops_count[i])});
        packet_headers[i]->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{noc_dest_addr, 1, std::numeric_limits<uint16_t>::max()});
        header_address += sizeof(PACKET_HEADER_TYPE);

        fabric_connection_handles[i] =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    }

    // wait for senders semaphore
    auto senders_to_controller_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_config.senders_to_controller_sem_address);
    noc_semaphore_wait(senders_to_controller_sem_ptr, worker_config.num_sender_workers);
    // reset sem for further notifications
    senders_to_controller_sem_ptr[0] = 0;

    // wait for host's signal
    if (worker_config.wait_for_host_signal) {
        auto host_to_controller_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_config.host_to_controller_sem_address);
        noc_semaphore_wait(host_to_controller_sem_ptr, 1);
    }

    uint32_t num_remote_controllers = 0;
    for (auto i = 0; i < NUM_DIRECTIONS; i++) {
        if (worker_config.hops_count[i] == 0) {
            continue;
        }
        fabric_connection_handles[i].open();
        num_remote_controllers += worker_config.hops_count[i];
    }

    // sync with other controllers
    auto remote_controller_sync_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_config.remote_controller_sync_address);
    for (auto sync_step = 1; sync_step <= 2; sync_step++) {
        for (auto i = 0; i < NUM_DIRECTIONS; i++) {
            if (worker_config.hops_count[i] == 0) {
                continue;
            }

            fabric_connection_handles[i].wait_for_empty_write_slot();
            fabric_connection_handles[i].send_payload_flush_non_blocking_from_address(
                (uint32_t)packet_headers[i], sizeof(PACKET_HEADER_TYPE));
        }
        noc_async_write_barrier();
        while (remote_controller_sync_ptr[0] < (num_remote_controllers * sync_step));
    }

    // teardown connections
    for (auto i = 0; i < NUM_DIRECTIONS; i++) {
        if (worker_config.hops_count[i] == 0) {
            continue;
        }
        fabric_connection_handles[i].close();
    }
    noc_async_write_barrier();

    // notify workers to begin sending traffic
    auto controller_to_workers_sem_ptr =
        reinterpret_cast<tt_l1_ptr uint32_t*>(worker_config.controller_to_workers_sem_address);
    controller_to_workers_sem_ptr[0] = 1;
    uint64_t mcast_dest_addr =
        get_noc_addr_helper(worker_config.mcast_encoding, (uint32_t)controller_to_workers_sem_ptr);
    noc_async_write_multicast_loopback_src(
        (uint32_t)controller_to_workers_sem_ptr, mcast_dest_addr, sizeof(uint32_t), worker_config.num_mcast_dests);
    noc_async_writes_flushed();
}
