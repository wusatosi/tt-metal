// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "noc_nonblocking_api.h"
#include "common_defines.h"

inline void (*rtos_context_switch_ptr)();
volatile inline uint32_t* flag_disable = (uint32_t*)(eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);

// #if COMPILE_FOR_IDLE_ERISC == 0
// constexpr uint8_t risc_type = static_cast<std::underlying_type_t<EthProcessorTypes>>(EthProcessorTypes::DM0);
// #elif COMPILE_FOR_IDLE_ERISC == 1
// constexpr uint8_t risc_type = static_cast<std::underlying_type_t<EthProcessorTypes>>(EthProcessorTypes::DM1);
// #elif defined(COMPILE_FOR_ERISC)
// constexpr uint8_t risc_type = static_cast<std::underlying_type_t<EthProcessorTypes>>(EthProcessorTypes::DM0);
// #endif

namespace internal_ {
inline __attribute__((always_inline)) void risc_context_switch() {
    ncrisc_noc_full_sync<risc_type>();
    rtos_context_switch_ptr();
    ncrisc_noc_counters_init<risc_type>();
}

inline __attribute__((always_inline)) void disable_erisc_app() { flag_disable[0] = 0; }
}  // namespace internal_
