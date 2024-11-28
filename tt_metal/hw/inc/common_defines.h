// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "core_config.h"

#if defined(COMPILE_FOR_BRISC)
constexpr uint8_t risc_type = static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM0);
#elif defined(COMPILE_FOR_NCRISC)
constexpr uint8_t risc_type = static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM1);
#elif COMPILE_FOR_IDLE_ERISC == 0
constexpr uint8_t risc_type = static_cast<std::underlying_type_t<EthProcessorTypes>>(EthProcessorTypes::DM0);
#elif COMPILE_FOR_IDLE_ERISC == 1
constexpr uint8_t risc_type = static_cast<std::underlying_type_t<EthProcessorTypes>>(EthProcessorTypes::DM1);
#elif defined(COMPILE_FOR_ERISC)
constexpr uint8_t risc_type = static_cast<std::underlying_type_t<EthProcessorTypes>>(EthProcessorTypes::DM0);
#endif
