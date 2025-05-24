// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef DM_COMMON_HPP
#define DM_COMMON_HPP

#include <cstdint>
#include "device_fixture.hpp"

namespace tt::tt_metal::unit_tests::dm {
// Unique id for each test run
extern uint32_t runtime_host_id;

// Function to obtain page size in bytes
uint32_t obtain_page_size_bytes(tt::ARCH arch);

}  // namespace tt::tt_metal::unit_tests::dm

#endif  // DM_COMMON_HPP
