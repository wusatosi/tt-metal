// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
// #include <unistd.h>

template <uint8_t noc_ind = noc_index>
FORCE_INLINE std::uint64_t static_noc_multicast_addr(
    std::uint32_t noc_x_start,
    std::uint32_t noc_y_start,
    std::uint32_t noc_x_end,
    std::uint32_t noc_y_end,
    std::uint32_t addr) {
    if constexpr (noc_ind == 0) {
        return get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, addr);
    } else {
        return get_noc_multicast_addr(noc_x_end, noc_y_end, noc_x_start, noc_y_start, addr);
    }
}

void kernel_main() {
    // Constants for indexing
    constexpr uint8_t x_index = 0;
    constexpr uint8_t y_index = 1;

    size_t rt_arg_idx = 0;

    // Define all compile-time arguments at the beginning
    constexpr uint32_t input_tensor_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t fabric_sender_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t fabric_receiver_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t accumulator_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t output_tensor_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t chip_id = get_compile_time_arg_val(6);
    constexpr uint32_t per_core_width_in_pages = get_compile_time_arg_val(7);
    constexpr uint32_t per_core_width_in_pages_output = get_compile_time_arg_val(8);
    constexpr uint32_t num_pages_per_packet = get_compile_time_arg_val(9);
    constexpr uint32_t input_shard_cores_per_device = get_compile_time_arg_val(10);
    constexpr uint32_t num_devices = get_compile_time_arg_val(11);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t output_cores_per_device = get_compile_time_arg_val(13);
    constexpr uint32_t packet_worker_start_x = get_compile_time_arg_val(14);
    constexpr uint32_t packet_worker_start_y = get_compile_time_arg_val(15);
    constexpr uint32_t packet_worker_end_x = get_compile_time_arg_val(16);
    constexpr uint32_t packet_worker_end_y = get_compile_time_arg_val(17);
    constexpr uint32_t num_sender_cores = get_compile_time_arg_val(18);
    constexpr uint32_t total_num_read_txns = get_compile_time_arg_val(19);
    constexpr bool slice_on_height = get_compile_time_arg_val(20) == 1;

    // Derived compile-time constants
    constexpr uint32_t input_tensor_cores =
        slice_on_height ? input_shard_cores_per_device : input_shard_cores_per_device * num_devices;

    // constexpr uint32_t num_packets_total_per_device =
    //     (input_shard_cores_per_device * per_core_width_in_pages + num_pages_per_packet - 1) / num_pages_per_packet;

    // DPRINT << "per_core_width_in_pages " << per_core_width_in_pages <<ENDL();
    // DPRINT << "num_pages_per_packet " << num_pages_per_packet <<ENDL();
    // DPRINT << "num_packets_total_per_device " << num_packets_total_per_device <<ENDL();

    // Precompute constants for optimization
    // constexpr uint32_t per_core_num_pages_in_width_bytes = per_core_width_in_pages * page_size_bytes;
    constexpr uint32_t num_dests =
        (packet_worker_end_x - packet_worker_start_x + 1) * (packet_worker_end_y - packet_worker_start_y + 1);
    constexpr uint32_t chip_id_offset = slice_on_height ? 0 : chip_id * num_pages_per_packet * page_size_bytes;

    constexpr uint32_t other_devices = num_devices - 1;
    constexpr uint8_t device_order[other_devices] =
        DEVICE_ORDER;  // this is code gen'd in the program factory using the defines
    constexpr uint8_t input_core_xy[input_tensor_cores][2] = INPUT_CORE_XY;
    constexpr uint8_t output_core_xy[output_cores_per_device][2] = OUTPUT_CORE_XY;
    constexpr uint8_t schedule[total_num_read_txns][3] = SCHEDULE;
    constexpr uint32_t total_senders = num_sender_cores * other_devices;

    // Runtime arguments
    uint32_t receiver_semaphore_address = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t local_semaphore_address = get_semaphore(get_arg_val<uint32_t>(rt_arg_idx++));
    bool sender_core = (bool)get_arg_val<uint32_t>(rt_arg_idx++);
    bool worker_core = (bool)get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t linear_input_packet_start_idx = get_arg_val<uint32_t>(rt_arg_idx++);
    bool receiver_core = (bool)get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t sender_shard_start = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t sender_shard_end = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t sender_total_num_pages = get_arg_val<uint32_t>(rt_arg_idx++);

    // Bank base addresses (compute once)
    const uint32_t bank_base_address = get_read_ptr(input_tensor_cb_id);

    DPRINT << "bank_base_address " << (uint)bank_base_address << ENDL();
    // DPRINT << "sender_shard_end " << (uint)sender_shard_end << ENDL();
    // DPRINT << "sender_core " << (uint)sender_core << ENDL();
    // DPRINT << "per_core_num_pages_in_width_bytes " << (uint)per_core_num_pages_in_width_bytes << ENDL();
    // uint32_t total_push_tiles = 0;

    uint32_t sender_read_addr = get_write_ptr(fabric_sender_cb_id);

    if (sender_core) {
        for (uint32_t target_device_id : device_order) {
            const uint32_t base_core = slice_on_height ? 0 : target_device_id * input_shard_cores_per_device;
            const uint32_t read_offset_per_device = slice_on_height ? target_device_id * per_core_width_in_pages : 0;

            uint32_t num_pages_read = 0;
            uint32_t num_pages_reserve_push = 0;
            uint32_t shard_idx = sender_shard_start;
            // for (uint32_t shard_idx = sender_shard_start; shard_idx < sender_shard_end; shard_idx++) {
            while (num_pages_read < sender_total_num_pages) {
                const uint8_t curr_core = base_core + schedule[shard_idx][0];
                const uint32_t read_offset = read_offset_per_device + schedule[shard_idx][1];
                const uint32_t read_size = schedule[shard_idx][2];
                num_pages_reserve_push += read_size;

                auto num_pages_left = sender_total_num_pages - num_pages_read;
                const uint32_t curr_packet_num_pages = std::min(num_pages_per_packet, num_pages_left);

                // DPRINT << "curr_core " << (uint)curr_core << ENDL();
                // DPRINT << "read_offset " << read_offset << ENDL();
                // DPRINT << "read_size " << read_size << ENDL();
                // DPRINT << "curr_packet_num_pages " << curr_packet_num_pages << ENDL();
                // DPRINT << "num_pages_reserve_push " << num_pages_reserve_push << ENDL();

                const uint32_t x = input_core_xy[curr_core][x_index];
                const uint32_t y = input_core_xy[curr_core][y_index];
                const uint32_t offset_address = bank_base_address + (read_offset * page_size_bytes);
                const uint64_t shard_noc_addr = get_noc_addr(x, y, offset_address);
                const uint32_t transfer_size = read_size * page_size_bytes;

                DPRINT << "read_offset " << (uint)read_offset << ENDL();
                DPRINT << "page_size_bytes " << (uint)page_size_bytes << ENDL();
                DPRINT << "offset_address " << (uint)offset_address << ENDL();
                DPRINT << "sender_read_addr " << (uint)sender_read_addr << ENDL();
                DPRINT << "transfer_size " << (uint)transfer_size << ENDL();

                cb_reserve_back(fabric_sender_cb_id, num_pages_reserve_push);
                noc_async_read(shard_noc_addr, sender_read_addr, transfer_size);

                if (num_pages_reserve_push >= curr_packet_num_pages) {
                    noc_async_read_barrier();
                    // DPRINT << "num_pages_reserve_push " << num_pages_reserve_push << ENDL();
                    cb_push_back(fabric_sender_cb_id, num_pages_reserve_push);
                    num_pages_reserve_push = 0;
                }

                sender_read_addr += transfer_size;
                num_pages_read += read_size;
                shard_idx++;
                // total_push_tiles += read_size;
            }
        }
        DPRINT << "reader done" << ENDL();
        // DPRINT << "total_push_tiles " << total_push_tiles << ENDL();
    } else if (worker_core) {
        // Calculate base addresses once
        const uint32_t base_input_tensor_addr = get_read_ptr(input_tensor_cb_id);
        const uint32_t base_receiver_l1_addresses = get_read_ptr(fabric_receiver_cb_id) + chip_id_offset;

        for (uint32_t i = 0; i < num_pages_per_packet; i++) {
            const uint32_t rem = linear_input_packet_start_idx + i;
            const uint32_t linear_input_core_idcs = rem / per_core_width_in_pages;
            const uint32_t linear_input_page_offsets = rem % per_core_width_in_pages;

            DPRINT << "linear_input_core_idcs " << linear_input_core_idcs << ENDL();

            const uint32_t core_x = input_core_xy[linear_input_core_idcs][x_index];
            const uint32_t core_y = input_core_xy[linear_input_core_idcs][y_index];
            const uint32_t page_offset = linear_input_page_offsets * page_size_bytes;

            DPRINT << "core_x " << core_x << ENDL();
            DPRINT << "core_y " << core_y << ENDL();
            DPRINT << "base_input_tensor_addr " << base_input_tensor_addr << ENDL();
            DPRINT << "page_offset " << page_offset << ENDL();

            const uint64_t output_noc_address = get_noc_addr(core_x, core_y, base_input_tensor_addr + page_offset);
            const uint32_t receiver_l1_address = base_receiver_l1_addresses + i * page_size_bytes;

            noc_async_read(output_noc_address, receiver_l1_address, page_size_bytes);
        }
        if (receiver_core) {
            // Precompute multicast semaphore address once
#ifndef SKIP_MCAST
            const uint64_t multicast_semaphore_addr = static_noc_multicast_addr(
                packet_worker_start_x,
                packet_worker_start_y,
                packet_worker_end_x,
                packet_worker_end_y,
                local_semaphore_address);

            noc_semaphore_wait((uint32_t*)receiver_semaphore_address, total_senders);

            DPRINT << "num_dests " << num_dests << ENDL();
            DPRINT << "receiver_semaphore_address " << receiver_semaphore_address << ENDL();
            DPRINT << "local_semaphore_address " << local_semaphore_address << ENDL();
            DPRINT << "packet_worker_start_x " << packet_worker_start_x << ENDL();
            DPRINT << "packet_worker_start_y " << packet_worker_start_y << ENDL();
            DPRINT << "packet_worker_end_x " << packet_worker_end_x << ENDL();
            DPRINT << "packet_worker_end_y " << packet_worker_end_y << ENDL();
            // DPRINT << TSLICE(fabric_receiver_cb_id, 0, SliceRange::h0_w0_32(), true, true) << ENDL();
            noc_semaphore_set_multicast(
                receiver_semaphore_address,
                multicast_semaphore_addr,
                num_dests);  // could do different mcast for each device by having num_devices - 1 receiver cores
#else
            noc_semaphore_wait((uint32_t*)receiver_semaphore_address, total_senders);
#endif
        } else {
            noc_semaphore_wait((uint32_t*)local_semaphore_address, total_senders);
        }

        noc_async_read_barrier();
        cb_push_back(fabric_receiver_cb_id, num_pages_per_packet * num_devices);

        DPRINT << "receiver done" << ENDL();
    }
    noc_semaphore_set((uint32_t*)local_semaphore_address, INVALID);
    noc_semaphore_set((uint32_t*)receiver_semaphore_address, INVALID);
}
