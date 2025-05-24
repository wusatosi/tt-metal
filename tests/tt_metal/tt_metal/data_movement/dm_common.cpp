// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dm_common.hpp"

namespace tt::tt_metal::unit_tests::dm {

uint32_t runtime_host_id = 0;

uint32_t obtain_page_size_bytes(tt::ARCH arch) { return (arch == tt::ARCH::BLACKHOLE) ? 64 : 32; }

}  // namespace tt::tt_metal::unit_tests::dm
