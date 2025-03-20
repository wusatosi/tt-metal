// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "debug/dprint.h"

enum class CORE_TYPE : uint8_t { IDLE_CORE = 0, WORKER_CORE = 1, HOP_CORE = 2 };

struct core_type_and_noc_s {
    uint8_t core_type : 4;
    uint8_t noc : 4;
};

static_assert(sizeof(core_type_and_noc_s) == 1, "core_type_and_noc_s should be exactly 1 byte.");

struct ct_args_info {
    core_type_and_noc_s core_type_and_noc;
    uint8_t ring_index;
    uint8_t next_core_noc_x;
    uint8_t next_core_noc_y;
};

#ifdef RANGE_0  // 30 cores
constexpr std::array<uint32_t, 62> compile_time_arg_array{
    get_compile_time_arg_val(0),  get_compile_time_arg_val(1),  get_compile_time_arg_val(2),
    get_compile_time_arg_val(3),  get_compile_time_arg_val(4),  get_compile_time_arg_val(5),
    get_compile_time_arg_val(6),  get_compile_time_arg_val(7),  get_compile_time_arg_val(8),
    get_compile_time_arg_val(9),  get_compile_time_arg_val(10), get_compile_time_arg_val(11),
    get_compile_time_arg_val(12), get_compile_time_arg_val(13), get_compile_time_arg_val(14),
    get_compile_time_arg_val(15), get_compile_time_arg_val(16), get_compile_time_arg_val(17),
    get_compile_time_arg_val(18), get_compile_time_arg_val(19), get_compile_time_arg_val(20),
    get_compile_time_arg_val(21), get_compile_time_arg_val(22), get_compile_time_arg_val(23),
    get_compile_time_arg_val(24), get_compile_time_arg_val(25), get_compile_time_arg_val(26),
    get_compile_time_arg_val(27), get_compile_time_arg_val(28), get_compile_time_arg_val(29),
    get_compile_time_arg_val(30), get_compile_time_arg_val(31), get_compile_time_arg_val(32),
    get_compile_time_arg_val(33), get_compile_time_arg_val(34), get_compile_time_arg_val(35),
    get_compile_time_arg_val(36), get_compile_time_arg_val(37), get_compile_time_arg_val(38),
    get_compile_time_arg_val(39), get_compile_time_arg_val(40), get_compile_time_arg_val(41),
    get_compile_time_arg_val(42), get_compile_time_arg_val(43), get_compile_time_arg_val(44),
    get_compile_time_arg_val(45), get_compile_time_arg_val(46), get_compile_time_arg_val(47),
    get_compile_time_arg_val(48), get_compile_time_arg_val(49), get_compile_time_arg_val(50),
    get_compile_time_arg_val(51), get_compile_time_arg_val(52), get_compile_time_arg_val(53),
    get_compile_time_arg_val(54), get_compile_time_arg_val(55), get_compile_time_arg_val(56),
    get_compile_time_arg_val(57), get_compile_time_arg_val(58), get_compile_time_arg_val(59),
    get_compile_time_arg_val(60), get_compile_time_arg_val(61)};
#else  // 20
constexpr std::array<uint32_t, 52> compile_time_arg_array{
    get_compile_time_arg_val(0),  get_compile_time_arg_val(1),  get_compile_time_arg_val(2),
    get_compile_time_arg_val(3),  get_compile_time_arg_val(4),  get_compile_time_arg_val(5),
    get_compile_time_arg_val(6),  get_compile_time_arg_val(7),  get_compile_time_arg_val(8),
    get_compile_time_arg_val(9),  get_compile_time_arg_val(10), get_compile_time_arg_val(11),
    get_compile_time_arg_val(12), get_compile_time_arg_val(13), get_compile_time_arg_val(14),
    get_compile_time_arg_val(15), get_compile_time_arg_val(16), get_compile_time_arg_val(17),
    get_compile_time_arg_val(18), get_compile_time_arg_val(19), get_compile_time_arg_val(20),
    get_compile_time_arg_val(21), get_compile_time_arg_val(22), get_compile_time_arg_val(23),
    get_compile_time_arg_val(24), get_compile_time_arg_val(25), get_compile_time_arg_val(26),
    get_compile_time_arg_val(27), get_compile_time_arg_val(28), get_compile_time_arg_val(29),
    get_compile_time_arg_val(30), get_compile_time_arg_val(31), get_compile_time_arg_val(32),
    get_compile_time_arg_val(33), get_compile_time_arg_val(34), get_compile_time_arg_val(35),
    get_compile_time_arg_val(36), get_compile_time_arg_val(37), get_compile_time_arg_val(38),
    get_compile_time_arg_val(39), get_compile_time_arg_val(40), get_compile_time_arg_val(41),
    get_compile_time_arg_val(42), get_compile_time_arg_val(43), get_compile_time_arg_val(44),
    get_compile_time_arg_val(45), get_compile_time_arg_val(46), get_compile_time_arg_val(47),
    get_compile_time_arg_val(48), get_compile_time_arg_val(49), get_compile_time_arg_val(50),
    get_compile_time_arg_val(51)};
#endif

void kernel_main() {
    // Compile time args
    // constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(0);
    // constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(1);
    // constexpr uint32_t batch = get_compile_time_arg_val(2);

    // // All Gather specific
    // constexpr uint32_t ring_size = get_compile_time_arg_val(3);
    // uint32_t signal_semaphore_addr = get_semaphore(get_compile_time_arg_val(4));
    uint32_t ct_args_idx = 0;
    constexpr uint32_t shard_width_in_tiles = compile_time_arg_array[ct_args_idx++];
    constexpr uint32_t shard_height_in_tiles = compile_time_arg_array[ct_args_idx++];
    constexpr uint32_t batch = compile_time_arg_array[ct_args_idx++];

    // All Gather specific
    constexpr uint32_t ring_size = compile_time_arg_array[ct_args_idx++];
    uint32_t signal_semaphore_addr = get_semaphore(compile_time_arg_array[ct_args_idx++]);

    const uint32_t start_core_x = compile_time_arg_array[ct_args_idx++];
    const uint32_t start_core_y = compile_time_arg_array[ct_args_idx++];
    const uint32_t NUM_CORES_X = compile_time_arg_array[ct_args_idx++];

    volatile tt_l1_ptr uint32_t* unpadded_in0_shard_widths_in_tiles =
        (volatile tt_l1_ptr uint32_t*)(&compile_time_arg_array[ct_args_idx]);
    ct_args_idx += ring_size;
    volatile tt_l1_ptr uint8_t* ct_args_base = (volatile tt_l1_ptr uint8_t*)(&compile_time_arg_array[ct_args_idx]);
    uint32_t core_id =
        (get_absolute_logical_x() - start_core_x) + (get_absolute_logical_y() - start_core_y) * NUM_CORES_X;
    volatile tt_l1_ptr ct_args_info* info =
        (volatile tt_l1_ptr ct_args_info*)(ct_args_base + core_id * sizeof(ct_args_info));

    uint32_t core_type = (uint32_t)info->core_type;
    if (core_type == (uint32_t)CORE_TYPE::IDLE_CORE) {
        return;
    }
    bool is_hop_core = core_type == (uint32_t)CORE_TYPE::HOP_CORE;
    uint32_t ring_idx = (uint32_t)info->ring_index;
    uint32_t next_core_noc_x = (uint32_t)info->next_core_noc_x;
    uint32_t next_core_noc_y = (uint32_t)info->next_core_noc_y;
    uint32_t noc = (uint32_t)info->noc;

    // uint32_t common_rt_args_idx = 0;
    // const uint32_t start_core_x = get_common_arg_val<uint32_t>(common_rt_args_idx++);
    // const uint32_t start_core_y = get_common_arg_val<uint32_t>(common_rt_args_idx++);
    // const uint32_t NUM_CORES_X = get_common_arg_val<uint32_t>(common_rt_args_idx++);
    // uint32_t core_id =
    //     (get_absolute_logical_x() - start_core_x) + (get_absolute_logical_y() - start_core_y) * NUM_CORES_X;

    // volatile tt_l1_ptr uint32_t* unpadded_in0_shard_widths_in_tiles =
    //     (volatile tt_l1_ptr uint32_t*)(get_common_arg_addr(common_rt_args_idx));
    // common_rt_args_idx += ring_size;
    // volatile tt_l1_ptr uint8_t* common_rt_args_base =
    //     (volatile tt_l1_ptr uint8_t*)(get_common_arg_addr(common_rt_args_idx));
    // volatile tt_l1_ptr common_rt_args_info* common_rt_args =
    //     (volatile tt_l1_ptr common_rt_args_info*)(common_rt_args_base + core_id * sizeof(common_rt_args_info));

    // uint32_t core_type = (uint32_t)common_rt_args->core_type;

    // if (core_type == (uint32_t)CORE_TYPE::IDLE_CORE) {
    //     return;
    // }
    // bool is_hop_core = core_type == (uint32_t)CORE_TYPE::HOP_CORE;
    // uint32_t ring_idx = (uint32_t)common_rt_args->ring_index;
    // uint32_t next_core_noc_x = (uint32_t)common_rt_args->next_core_noc_x;
    // uint32_t next_core_noc_y = (uint32_t)common_rt_args->next_core_noc_y;
    // uint32_t noc = (uint32_t)common_rt_args->noc;

    volatile tt_l1_ptr uint32_t* l1_signal_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);
    uint64_t remote_signal_semaphore_addr = get_noc_addr(next_core_noc_x, next_core_noc_y, signal_semaphore_addr, noc);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in2 = tt::CBIndex::c_2;

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr uint32_t shard_size_in_tiles = shard_width_in_tiles * shard_height_in_tiles;
    constexpr uint32_t shard_size_bytes = shard_size_in_tiles * in0_single_tile_size_bytes;

    // Reserving/pushing the local shard is done in compute
    cb_reserve_back(cb_id_in2, batch * (ring_size - 1) * shard_size_in_tiles);

    uint32_t local_shard_read_addr = get_read_ptr(cb_id_in0);
    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in2);

    uint32_t hop_core_offset = static_cast<uint32_t>(is_hop_core);

    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t shard_cnt = hop_core_offset; shard_cnt < ring_size; shard_cnt++) {
            uint32_t curr_ring_idx = (ring_idx + shard_cnt) % ring_size;
            bool skip_send = unpadded_in0_shard_widths_in_tiles[curr_ring_idx] == 0 && !is_hop_core;

            uint32_t curr_shard_write_addr = l1_write_addr_in0 + shard_size_bytes * (shard_cnt - hop_core_offset);
            uint64_t remote_curr_shard_write_addr =
                get_noc_addr(next_core_noc_x, next_core_noc_y, curr_shard_write_addr, noc);
            uint32_t curr_shard_read_addr =
                shard_cnt == 0 ? local_shard_read_addr : l1_write_addr_in0 + shard_size_bytes * (shard_cnt - 1);

            // Wait for signal from previous core that data has been added to this core's in0
            noc_semaphore_wait_min(l1_signal_sem_addr, shard_cnt);

            // Send data to next core
            if (shard_cnt < ring_size - 1 || is_hop_core) {  // Skip sending the last shard
                if (!skip_send) {
                    noc_async_write(curr_shard_read_addr, remote_curr_shard_write_addr, shard_size_bytes, noc);
                }

                // Signal the next core that data is ready
                noc_semaphore_inc(remote_signal_semaphore_addr, 1, noc);
            }

            // Do stuff for matmul fusion here
            if (shard_cnt > 0) {
                cb_push_back(cb_id_in2, shard_size_in_tiles);
            }
        }
    }
    noc_async_atomic_barrier();
}
