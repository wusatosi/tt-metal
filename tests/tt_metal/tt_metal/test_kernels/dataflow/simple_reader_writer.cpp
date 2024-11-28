// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    constexpr std::uint32_t iteration = get_compile_time_arg_val(0);
    constexpr std::uint32_t page_size = get_compile_time_arg_val(1);

    std::uint32_t noc_x = get_arg_val<uint32_t>(0);
    std::uint32_t noc_y = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id = 0;
    uint32_t l1_read_addr = get_read_ptr(cb_id);

    uint64_t noc_addr = get_noc_addr(noc_x, noc_y, l1_read_addr);

#if defined(READ)
    for (uint32_t i = 0; i < iteration; i ++) {
        noc_async_read(noc_addr, l1_read_addr, page_size);
        noc_async_read_barrier();
    }
    DPRINT << "read done" <<ENDL();
#endif
#if defined(WRITE)
    for (uint32_t i = 0; i < iteration; i ++) {
        noc_async_write(l1_read_addr, noc_addr, page_size);
        noc_async_writes_flushed();
        noc_async_write_barrier();
    }
    DPRINT << "write done" <<ENDL();
#endif
#if defined(ATOMIC)
    for (uint32_t i = 0; i < iteration; i ++) {
        noc_semaphore_inc(noc_addr, 1);
        noc_async_atomic_barrier();
    }
    DPRINT << "atomic done" <<ENDL();
#endif
#if defined(POSTED_WRITE)
    for (uint32_t i = 0; i < iteration; i ++) {
        ncrisc_noc_fast_write_any_len<risc_type>(
            noc_index, write_cmd_buf, l1_read_addr, noc_addr, page_size, NOC_UNICAST_WRITE_VC, false, false, 1, true, true);

        while (!ncrisc_noc_posted_writes_sent<risc_type>(noc_index));
    }
    DPRINT << "posted write done" <<ENDL();
#endif
}
