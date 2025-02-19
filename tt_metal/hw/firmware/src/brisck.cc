// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <unistd.h>
#include <cstdint>

#include "risc_common.h"
#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "noc_nonblocking_api.h"
#include "firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include <kernel_includes.hpp>
#if defined ALIGN_LOCAL_CBS_TO_REMOTE_CBS
#include "remote_circular_buffer_api.h"
#endif

volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
volatile tt_l1_ptr uint32_t* profiler_data_buffer =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(GET_MAILBOX_ADDRESS_DEV(profiler.buffer));
void kernel_launch(uint32_t kernel_base_addr) {
#if defined(DEBUG_NULL_KERNELS) && !defined(DISPATCH_KERNEL)
    wait_for_go_message();
#ifdef KERNEL_RUN_TIME
    uint64_t end_time = c_tensix_core::read_wall_clock() + KERNEL_RUN_TIME;
    while (c_tensix_core::read_wall_clock() < end_time);
#endif
#else
    extern uint32_t __kernel_init_local_l1_base[];
    extern uint32_t __fw_export_end_text[];
    do_crt1((uint32_t tt_l1_ptr
                 *)(kernel_base_addr + (uint32_t)__kernel_init_local_l1_base - (uint32_t)__fw_export_end_text));

    if constexpr (NOC_MODE == DM_DEDICATED_NOC) {
        noc_local_state_init(NOC_INDEX);
    }
#ifdef ALIGN_LOCAL_CBS_TO_REMOTE_CBS
    ALIGN_LOCAL_CBS_TO_REMOTE_CBS
#endif
    wait_for_go_message();
    {
        DeviceZoneScopedMainChildN("BRISC-KERNEL");
        profiler_data_buffer[8] = p_reg[0];
        profiler_data_buffer[9] = p_reg[1];
        profiler_data_buffer[9] = p_reg[1];
        profiler_data_buffer[9] = p_reg[1];
        kernel_main();
        profiler_data_buffer[10] = p_reg[0];
        profiler_data_buffer[11] = p_reg[1];
        profiler_data_buffer[11] = p_reg[1];
        profiler_data_buffer[11] = p_reg[1];
    }
#endif
}
