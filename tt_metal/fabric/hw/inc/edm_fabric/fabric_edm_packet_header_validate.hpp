// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric_edm_packet_header.hpp"
#include "debug/assert.h"
#include "debug/waypoint.h"

namespace tt::tt_fabric {

FORCE_INLINE void validate(const PacketHeader& packet_header) {
    ASSERT(packet_header.chip_send_type <= CHIP_SEND_TYPE_LAST);
}
FORCE_INLINE bool is_valid(const PacketHeader& packet_header) {
    WAYPOINT("TYPE");
    return (packet_header.chip_send_type <= CHIP_SEND_TYPE_LAST) && (packet_header.noc_send_type <= NOC_SEND_TYPE_LAST);
}

FORCE_INLINE void validate(const LowLatencyPacketHeader& packet_header) {}
FORCE_INLINE bool is_valid(const LowLatencyPacketHeader& packet_header) {
    WAYPOINT("PYPE");
    WATCHER_RING_BUFFER_PUSH((uint32_t)packet_header.noc_send_type);
    return (packet_header.noc_send_type <= NOC_SEND_TYPE_LAST);
}

FORCE_INLINE void validate(const LowLatencyMeshPacketHeader& packet_header) {}
FORCE_INLINE bool is_valid(const LowLatencyMeshPacketHeader& packet_header) {
    WAYPOINT("MYPE");
    return (packet_header.noc_send_type <= NOC_SEND_TYPE_LAST);
}

FORCE_INLINE void validate(const MeshPacketHeader& packet_header) {
    ASSERT(packet_header.noc_send_type <= NOC_SEND_TYPE_LAST);
}

FORCE_INLINE bool is_valid(const MeshPacketHeader& packet_header) {
    WAYPOINT("QYPE");
    return (packet_header.noc_send_type <= NOC_SEND_TYPE_LAST);
}

}  // namespace tt::tt_fabric
