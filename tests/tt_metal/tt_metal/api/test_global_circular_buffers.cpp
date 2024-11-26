// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "device_fixture.hpp"
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/global_circular_buffer.hpp"

TEST_F(DispatchFixture, CreateGlobalCircularBuffers) {
    CoreRangeSet cores(CoreRange({1, 1}, {1, 1}));
    CoreRangeSet cores2(CoreRange({1, 1}, {2, 2}));
    CoreRangeSet cores3(CoreRange({3, 3}, {3, 3}));

    for (auto device : devices_) {
        {
            std::unordered_map<CoreCoord, CoreRangeSet> sender_receiver_core_mapping;
            sender_receiver_core_mapping[CoreCoord(0, 0)] = cores;
            auto global_cb = tt::tt_metal::experimental::CreateGlobalCircularBuffer(device, sender_receiver_core_mapping, 3200, tt::tt_metal::BufferType::L1);
            auto buffer_address = global_cb->buffer_address();
            auto config_address = global_cb->config_address();
        }
        {
            std::unordered_map<CoreCoord, CoreRangeSet> sender_receiver_core_mapping;
            sender_receiver_core_mapping[CoreCoord(0, 0)] = cores;
            sender_receiver_core_mapping[CoreCoord(1, 1)] = cores3;
            // sender receiver cores overlap
            EXPECT_THROW(tt::tt_metal::experimental::CreateGlobalCircularBuffer(device, sender_receiver_core_mapping, 3200, tt::tt_metal::BufferType::L1), std::exception);
        }
        {
            std::unordered_map<CoreCoord, CoreRangeSet> sender_receiver_core_mapping;
            sender_receiver_core_mapping[CoreCoord(0, 0)] = cores;
            sender_receiver_core_mapping[CoreCoord(0, 1)] = cores2;
            // receiver cores overlap
            EXPECT_THROW(tt::tt_metal::experimental::CreateGlobalCircularBuffer(device, sender_receiver_core_mapping, 3200, tt::tt_metal::BufferType::L1), std::exception);
        }
    }
}
