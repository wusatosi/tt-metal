// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "risc_common.h"
#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "noc_nonblocking_api.h"
#include "stream_io_map.h"
#ifdef PERF_DUMP
#include "risc_perf.h"
#endif
#include "firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tensix_functions.h"
#include "c_tensix_core.h"
#include "kernel_includes.hpp"
#if defined ALIGN_LOCAL_CBS_TO_REMOTE_CBS or defined UPDATE_REMOTE_CB_CONFIGS_IN_L1
#include "circular_buffer_init.h"
#endif
#include "debug/dprint.h"

bool skip_kernel() {
#ifdef SKIP_KERNEL
    volatile tt_l1_ptr uint32_t* p_tensor = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(P_TENSOR_ADDR);
    uint32_t p_tensor_data = *p_tensor;

    if (p_tensor_data == 1) {
        DPRINT << "Skipping NCRISC kernel" << ENDL();
        return true;
    }
    return false;
#else
    return false;
#endif
}

uint32_t noc_reads_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_acked[NUM_NOCS];
uint32_t noc_nonposted_atomics_acked[NUM_NOCS];
uint32_t noc_posted_writes_num_issued[NUM_NOCS];

void kernel_launch(uint32_t kernel_base_addr) {
    DeviceZoneScopedMainChildN("NCRISC-KERNEL");
#if defined(DEBUG_NULL_KERNELS) && !defined(DISPATCH_KERNEL)
#ifdef KERNEL_RUN_TIME
    uint64_t end_time = c_tensix_core::read_wall_clock() + KERNEL_RUN_TIME;
    while (c_tensix_core::read_wall_clock() < KERNEL_RUN_TIME);
#endif
#else
    extern uint32_t __kernel_init_local_l1_base[];
    extern uint32_t __fw_export_end_text[];
    do_crt1((uint32_t tt_l1_ptr*)(kernel_base_addr + (uint32_t)__kernel_init_local_l1_base -
                                  (uint32_t)__fw_export_end_text));

    if constexpr (NOC_MODE == DM_DEDICATED_NOC) {
        noc_local_state_init(NOC_INDEX);
    }
#ifdef ALIGN_LOCAL_CBS_TO_REMOTE_CBS
    ALIGN_LOCAL_CBS_TO_REMOTE_CBS
#endif
    if (!skip_kernel()) {
        kernel_main();
    }
#ifdef UPDATE_REMOTE_CB_CONFIGS_IN_L1
    UPDATE_REMOTE_CB_CONFIGS_IN_L1
#endif
#endif
}
