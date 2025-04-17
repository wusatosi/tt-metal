// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_common.hpp"

namespace tt::tt_fabric {

struct ControllerWorkerConfig {
    static constexpr uint8_t MASTER_EDM_STATUS_SIZE_BYTES = 16;

    static ControllerWorkerConfig build_from_args(std::size_t& arg_idx) {
        uint32_t base_address = get_arg_val<uint32_t>(arg_idx++);
        uint32_t num_sender_workers = get_arg_val<uint32_t>(arg_idx++);
        uint32_t num_mcast_dests = get_arg_val<uint32_t>(arg_idx++);
        uint32_t mcast_encoding = get_arg_val<uint32_t>(arg_idx++);

        tt_l1_ptr uint32_t* hops_count = reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(arg_idx));
        arg_idx += NUM_DIRECTIONS;

        return ControllerWorkerConfig(base_address, num_sender_workers, num_mcast_dests, mcast_encoding, hops_count);
    }

    ControllerWorkerConfig(
        uint32_t base_address,
        uint32_t num_sender_workers,
        uint32_t num_mcast_dests,
        uint32_t mcast_encoding,
        tt_l1_ptr uint32_t* hops_count) {
        this->host_to_controller_sem_address = base_address + HOST_TO_CONTROLLER_SEM_OFFSET;
        this->remote_controller_sync_address = base_address + CONTROLLER_TO_CONTROLLER_SEM_OFFSET;
        this->controller_to_workers_sem_address = base_address + CONTROLLER_TO_WORKERS_SEM_OFFSET;
        this->senders_to_controller_sem_address = base_address + SENDERS_TO_CONTROLLER_SEM_OFFSET;
        this->receivers_to_controller_sem_address = base_address + RECEIVERS_TO_CONTROLLER_SEM_OFFSET;
        this->worker_usable_base_address = base_address + WORKER_USABLE_BASE_ADDRESS_OFFSET;
        this->packet_header_buffer_address = this->worker_usable_base_address + MASTER_EDM_STATUS_SIZE_BYTES;

        this->num_sender_workers = num_sender_workers;
        this->num_mcast_dests = num_mcast_dests;
        this->mcast_encoding = mcast_encoding;
        for (auto i = 0; i < NUM_DIRECTIONS; i++) {
            this->hops_count[i] = hops_count[i];
        }
    }

    // memory map
    uint32_t host_to_controller_sem_address = 0;
    uint32_t remote_controller_sync_address = 0;
    uint32_t controller_to_workers_sem_address = 0;
    uint32_t senders_to_controller_sem_address = 0;
    uint32_t receivers_to_controller_sem_address = 0;
    uint32_t worker_usable_base_address = 0;
    uint32_t packet_header_buffer_address = 0;

    // test params
    uint32_t num_sender_workers = 0;
    uint32_t num_mcast_dests = 0;
    uint32_t mcast_encoding = 0;
    std::array<uint32_t, NUM_DIRECTIONS> hops_count = {0};
};

struct SenderWorkerConfig {
    static constexpr uint16_t PACKET_HEADER_BUFFER_SIZE_BYTES = 1024;

    static SenderWorkerConfig build_from_args(std::size_t& arg_idx) {
        uint32_t base_address = get_arg_val<uint32_t>(arg_idx++);
        uint32_t routing_plane_id = get_arg_val<uint32_t>(arg_idx++);
        uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
        uint32_t num_packets = get_arg_val<uint32_t>(arg_idx++);
        uint32_t time_seed = get_arg_val<uint32_t>(arg_idx++);
        uint32_t sender_id = get_arg_val<uint32_t>(arg_idx++);
        uint32_t controller_noc_encoding = get_arg_val<uint32_t>(arg_idx++);
        uint32_t receiver_noc_encoding = get_arg_val<uint32_t>(arg_idx++);

        tt_l1_ptr uint32_t* is_mcast_enabled = reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(arg_idx));
        arg_idx += NUM_DIRECTIONS;

        tt_l1_ptr uint32_t* hops_count = reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(arg_idx));
        arg_idx += NUM_DIRECTIONS;

        return SenderWorkerConfig(
            base_address,
            routing_plane_id,
            packet_payload_size_bytes,
            num_packets,
            time_seed,
            sender_id,
            controller_noc_encoding,
            receiver_noc_encoding,
            is_mcast_enabled,
            hops_count);
    }

    SenderWorkerConfig(
        uint32_t base_address,
        uint32_t routing_plane_id,
        uint32_t packet_payload_size_bytes,
        uint32_t num_packets,
        uint32_t time_seed,
        uint32_t sender_id,
        uint32_t controller_noc_encoding,
        uint32_t receiver_noc_encoding,
        tt_l1_ptr uint32_t* is_mcast_enabled,
        tt_l1_ptr uint32_t* hops_count) {
        this->test_results_address = base_address + TEST_RESULTS_ADDRESS_OFFSET;
        this->controller_to_workers_sem_address = base_address + CONTROLLER_TO_WORKERS_SEM_OFFSET;
        this->senders_to_controller_sem_address = base_address + SENDERS_TO_CONTROLLER_SEM_OFFSET;
        this->worker_usable_base_address = base_address + WORKER_USABLE_BASE_ADDRESS_OFFSET;
        this->packet_header_buffer_address = this->worker_usable_base_address;
        this->payload_buffer_address = this->packet_header_buffer_address + PACKET_HEADER_BUFFER_SIZE_BYTES;
        this->base_target_address = base_address + BASE_TARGET_ADDRESS_OFFSET;
        this->target_address = this->base_target_address + (L1_BUFFER_SIZE_PER_SENDER_BYTES * routing_plane_id);

        this->packet_payload_size_bytes = packet_payload_size_bytes;
        this->num_packets = num_packets;
        this->time_seed = time_seed;
        this->controller_noc_encoding = controller_noc_encoding;
        this->receiver_noc_encoding = receiver_noc_encoding;
        this->sender_id = sender_id;
        for (auto i = 0; i < NUM_DIRECTIONS; i++) {
            this->is_mcast_enabled[i] = is_mcast_enabled[i];
            this->hops_count[i] = hops_count[i];
        }
    }

    // memory map
    uint32_t test_results_address = 0;
    uint32_t controller_to_workers_sem_address = 0;
    uint32_t senders_to_controller_sem_address = 0;
    uint32_t worker_usable_base_address = 0;
    uint32_t packet_header_buffer_address = 0;
    uint32_t payload_buffer_address = 0;
    uint32_t base_target_address = 0;
    uint32_t target_address = 0;

    // test parameters
    uint32_t packet_payload_size_bytes = 0;
    uint32_t num_packets = 0;
    uint32_t time_seed = 0;
    uint32_t controller_noc_encoding = 0;
    uint32_t receiver_noc_encoding = 0;
    uint32_t sender_id = 0;
    std::array<bool, NUM_DIRECTIONS> is_mcast_enabled = {false};
    std::array<uint32_t, NUM_DIRECTIONS> hops_count = {0};
};

struct ReceiverWorkerConfig {
    static ReceiverWorkerConfig build_from_args(std::size_t& arg_idx) {
        uint32_t base_address = get_arg_val<uint32_t>(arg_idx++);
        uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
        uint32_t num_packets = get_arg_val<uint32_t>(arg_idx++);
        uint32_t time_seed = get_arg_val<uint32_t>(arg_idx++);

        tt_l1_ptr uint32_t* sender_ids = reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(arg_idx));
        arg_idx += MAX_NUM_SENDERS_PER_RECEIVER;

        return ReceiverWorkerConfig(base_address, packet_payload_size_bytes, num_packets, time_seed, sender_ids);
    }

    ReceiverWorkerConfig(
        uint32_t base_address,
        uint32_t packet_payload_size_bytes,
        uint32_t num_packets,
        uint32_t time_seed,
        tt_l1_ptr uint32_t* sender_ids) {
        this->test_results_address = base_address + TEST_RESULTS_ADDRESS_OFFSET;
        this->controller_to_workers_sem_address = base_address + CONTROLLER_TO_WORKERS_SEM_OFFSET;
        this->receivers_to_controller_sem_address = base_address + RECEIVERS_TO_CONTROLLER_SEM_OFFSET;
        this->base_target_address = base_address + BASE_TARGET_ADDRESS_OFFSET;

        this->packet_payload_size_bytes = packet_payload_size_bytes;
        this->num_packets = num_packets;
        this->time_seed = time_seed;
        for (uint32_t i = 0; i < MAX_NUM_SENDERS_PER_RECEIVER; i++) {
            this->sender_ids[i] = sender_ids[i];
            this->target_addresses[i] = this->base_target_address + (L1_BUFFER_SIZE_PER_SENDER_BYTES * i);
        }
    }

    // memory map
    uint32_t test_results_address = 0;
    uint32_t controller_to_workers_sem_address = 0;
    uint32_t receivers_to_controller_sem_address = 0;
    uint32_t base_target_address = 0;

    // test parameters
    uint32_t packet_payload_size_bytes = 0;
    uint32_t num_packets = 0;
    uint32_t time_seed = 0;
    uint32_t num_senders = 0;
    std::array<uint32_t, MAX_NUM_SENDERS_PER_RECEIVER> sender_ids = {0};
    std::array<uint32_t, MAX_NUM_SENDERS_PER_RECEIVER> target_addresses = {0};
};

}  // namespace tt::tt_fabric
