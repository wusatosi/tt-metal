// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// KCM - This is an idea. But eventually this will be a class, and contain more stuff alongside lightmetal.hpp

#include <string>
#include <vector>

namespace tt::tt_metal {
inline namespace v0 {

// LightMetalTraceConfig struct
struct LightMetalTraceConfig {
    bool capture_enabled = false;
    bool auto_serialize_metal_trace = true;
    std::string filename = "/tmp/light_metal_trace.bin";
};

// LightMetalTrace struct
struct LightMetalTrace {
    std::vector<std::pair<uint32_t, detail::TraceDescriptor>> traces;
    LightMetalTraceConfig config;
};


}  // namespace v0
}  // namespace tt::tt_metal
