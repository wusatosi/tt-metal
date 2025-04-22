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
#include <array>

// clang-format on

inline void setup_connection_and_header(
    tt::tt_fabric::WorkerToFabricEdmSender& connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint32_t packet_payload_size_bytes,
    uint64_t noc_dest_addr,
    bool mcast_mode,
    uint32_t hops) {
    connection.open();
    if (mcast_mode) {
        packet_header->to_chip_multicast(MulticastRoutingCommandHeader{1, static_cast<uint8_t>(hops)});
    } else {
        packet_header->to_chip_unicast(static_cast<uint8_t>(hops));
    }

    packet_header->to_noc_unicast_write(NocUnicastCommandHeader{noc_dest_addr}, packet_payload_size_bytes);
}

inline void send_packet(
    tt::tt_fabric::WorkerToFabricEdmSender& connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint32_t payload_buffer_address,
    uint32_t packet_payload_size_bytes,
    uint64_t noc_dest_addr) {
#ifndef BENCHMARK_MODE
    packet_header->to_noc_unicast_write(NocUnicastCommandHeader{noc_dest_addr}, packet_payload_size_bytes);
#endif

    connection.wait_for_empty_write_slot();
    connection.send_payload_without_header_non_blocking_from_address(payload_buffer_address, packet_payload_size_bytes);
    connection.send_payload_flush_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

inline void teardown_connection(tt::tt_fabric::WorkerToFabricEdmSender& connection) { connection.close(); }

void kernel_main() {
    size_t rt_args_idx = 0;
    auto worker_config = tt::tt_fabric::SenderWorkerConfig::build_from_args(rt_args_idx);

    std::array<volatile tt_l1_ptr PACKET_HEADER_TYPE*, NUM_DIRECTIONS> packet_headers;
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, NUM_DIRECTIONS> fabric_connection_handles;

    uint32_t header_address = worker_config.packet_header_buffer_address;
    for (auto i = 0; i < NUM_DIRECTIONS; i++) {
        if (worker_config.hops_count[i] == 0) {
            continue;
        }

        packet_headers[i] = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header_address);
        header_address += sizeof(PACKET_HEADER_TYPE);

        fabric_connection_handles[i] =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    }

    tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(worker_config.test_results_address);
    zero_l1_buf(test_results, TEST_RESULTS_SIZE_BYTES);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    // notify controller worker that this sender is ready
    uint64_t controller_sem_noc_address =
        get_noc_addr_helper(worker_config.controller_noc_encoding, worker_config.senders_to_controller_sem_address);
    noc_semaphore_inc(controller_sem_noc_address, 1);

    // wait for notification from controller
    auto controller_to_workers_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_config.controller_to_workers_sem_address);
    noc_semaphore_wait(controller_to_workers_sem_ptr, 1);

    uint64_t noc_dest_addr_l = get_noc_addr_helper(worker_config.receiver_noc_encoding, worker_config.target_address);
    uint64_t noc_dest_addr_h = noc_dest_addr_l + L1_BUFFER_SIZE_PER_SENDER_BYTES;
    uint64_t noc_dest_addr = noc_dest_addr_l;
    for (auto i = 0; i < NUM_DIRECTIONS; i++) {
        if (worker_config.hops_count[i] == 0) {
            continue;
        }

        setup_connection_and_header(
            fabric_connection_handles[i],
            packet_headers[i],
            worker_config.packet_payload_size_bytes,
            noc_dest_addr,
            worker_config.is_mcast_enabled[i],
            worker_config.hops_count[i]);
    }

    tt_l1_ptr uint32_t* start_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(worker_config.payload_buffer_address);
    fill_packet_data(start_addr, worker_config.packet_payload_size_bytes / 16, 0);
    // send warm-up packets to saturate buffers
    for (auto warmup_iter = 0; warmup_iter < 1000; warmup_iter++) {
        for (auto i = 0; i < NUM_DIRECTIONS; i++) {
            if (worker_config.hops_count[i] == 0) {
                continue;
            }

            send_packet(
                fabric_connection_handles[i],
                packet_headers[i],
                worker_config.payload_buffer_address,
                worker_config.packet_payload_size_bytes,
                noc_dest_addr);
        }
    }

    uint64_t start_timestamp = get_timestamp();
    uint32_t seed = worker_config.time_seed ^ worker_config.sender_id;
    for (uint32_t packet_id = 0; packet_id < worker_config.num_packets; packet_id++) {
#ifndef BENCHMARK_MODE
            seed = prng_next(seed);
            // fill packet data for sanity testing
            tt_l1_ptr uint32_t* start_addr =
                reinterpret_cast<tt_l1_ptr uint32_t*>(worker_config.payload_buffer_address);
            fill_packet_data(start_addr, worker_config.packet_payload_size_bytes / 16, seed);
#endif
            for (auto i = 0; i < NUM_DIRECTIONS; i++) {
                if (worker_config.hops_count[i] == 0) {
                    continue;
                }

                send_packet(
                    fabric_connection_handles[i],
                    packet_headers[i],
                    worker_config.payload_buffer_address,
                    worker_config.packet_payload_size_bytes,
                    noc_dest_addr);
            }
#ifndef BENCHMARK_MODE
            noc_dest_addr += worker_config.packet_payload_size_bytes;
            // reset noc dest addr if the next packet is running out of bounds
            if (noc_dest_addr + worker_config.packet_payload_size_bytes > noc_dest_addr_h) {
                noc_dest_addr = noc_dest_addr_l;
            }
#endif
            noc_async_write_barrier();
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

    for (auto i = 0; i < NUM_DIRECTIONS; i++) {
        if (worker_config.hops_count[i] == 0) {
            continue;
        }
        teardown_connection(fabric_connection_handles[i]);
    }

    noc_async_write_barrier();

    uint64_t bytes_sent = worker_config.packet_payload_size_bytes * worker_config.num_packets;
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_CYCLES_INDEX] = (uint32_t)cycles_elapsed;
    test_results[TT_FABRIC_CYCLES_INDEX + 1] = cycles_elapsed >> 32;
    test_results[TT_FABRIC_WORD_CNT_INDEX] = (uint32_t)bytes_sent;
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = bytes_sent >> 32;
}
