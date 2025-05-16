#include "dataflow_api.h"

#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

using tt::data_movement::common::round_up;
using tt::data_movement::common::tt_memmove;

void kernel_main() {
    constexpr bool intermediate_is_dram = get_compile_time_arg_val(0);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t receiver_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t alignment = get_compile_time_arg_val(3);

    const auto packet_idx_start = get_arg_val<uint32_t>(0);
    const auto packet_idx_end = get_arg_val<uint32_t>(1);

    const auto max_pages_per_packet = get_arg_val<uint32_t>(2);
    const auto last_page_idx = get_arg_val<uint32_t>(3);

    const auto intermediate_base_addr = get_arg_val<uint32_t>(4);
    const auto packet_size_bytes = get_arg_val<uint32_t>(5);
    const auto page_size_bytes = get_arg_val<uint32_t>(6);

    auto semaphore_ptr = get_arg_val<volatile tt_l1_ptr uint32_t*>(7);

    const uint32_t aligned_page_size_bytes = round_up(page_size_bytes, alignment);

    InterleavedAddrGenFast<intermediate_is_dram> packet_buffer_addrgen{
        .bank_base_address = intermediate_base_addr,
        .page_size = packet_size_bytes,
        .data_format = get_dataformat(packet_cb_id)};

    cb_reserve_back(packet_cb_id, 1);
    const uint64_t packet_l1_addr = get_write_ptr(packet_cb_id);

    noc_semaphore_wait(semaphore_ptr, 1);

    uint32_t page_idx = packet_idx_start * max_pages_per_packet;
    for (uint32_t packet_idx = packet_idx_start; packet_idx < packet_idx_end; ++packet_idx) {
        const uint64_t packet_noc_addr = packet_buffer_addrgen.get_noc_addr(packet_idx);
        noc_async_read(packet_noc_addr, packet_l1_addr, packet_size_bytes);
        noc_async_read_barrier();

        const uint32_t curr_num_pages_packet = std::min(max_pages_per_packet, last_page_idx - page_idx);
        for (uint8_t packet_page_idx = 0; packet_page_idx < curr_num_pages_packet; ++packet_page_idx, ++page_idx) {
            cb_wait_front(receiver_cb_id, 1);
            const uint32_t page_l1_addr = get_write_ptr(receiver_cb_id);
            uint32_t packet_l1_page_addr = packet_l1_addr + aligned_page_size_bytes;
            tt_memmove<true, true, true, 0>(page_l1_addr, packet_l1_page_addr, page_size_bytes);
        }
        // async in chunks
        noc_async_write_barrier();
        cb_push_back(receiver_cb_id, curr_num_pages_packet);
    }
}
