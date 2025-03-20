// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <fmt/format.h>

#include <host_api.hpp>

namespace tt {

namespace tt_metal {

void running_program(uint32_t operation_id, uint32_t device_id) {
#if defined(TRACY_ENABLE)
    std::cout << fmt::format("op_id:{} on device:{}", operation_id, device_id) << std::endl;
#endif
}

}  // namespace tt_metal

}  // namespace tt
