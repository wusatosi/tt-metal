// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <math.h>
#include <algorithm>
#include <functional>
#include <random>

#include "gtest/gtest.h"

#include "ttnn/device.hpp"
#include "ttnn/types.hpp"
#include "tests/tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/host_api.hpp>
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/mesh_device.hpp>

namespace ttnn {

class TTNNFixture : public ::testing::Test {
protected:
    tt::ARCH arch_;
    size_t num_devices_;

    void SetUp() override {
        std::srand(0);
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
    }

    void TearDown() override {}
};

class TTNNFixtureWithDevice : public TTNNFixture {
private:
    int trace_region_size;
    int l1_small_size;

protected:
    tt::tt_metal::IDevice* device_ = nullptr;

    void SetUp() override {
        std::srand(0);
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        device_ = tt::tt_metal::CreateDevice(
            /*device_id=*/0,
            /*num_hw_cqs*/ 1,
            /*l1_small_size*/ l1_small_size,
            /*trace_region_size*/ trace_region_size);
    }

    void TearDown() override { tt::tt_metal::CloseDevice(device_); }

    tt::tt_metal::IDevice& getDevice() { return *device_; }

public:
    TTNNFixtureWithDevice() : trace_region_size(DEFAULT_TRACE_REGION_SIZE), l1_small_size(DEFAULT_L1_SMALL_SIZE) {}

    TTNNFixtureWithDevice(int trace_region_size, int l1_small_size) :
        trace_region_size(trace_region_size), l1_small_size(l1_small_size) {}
};

}  // namespace ttnn

namespace ttnn::distributed::test {

class T3kMultiDeviceFixture : public ::testing::Test {
protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (slow_dispatch) {
            GTEST_SKIP() << "Skipping Multi-Device test suite, since it can only be run in Fast Dispatch Mode.";
        }
        if (num_devices < 8 or arch != tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "Skipping T3K Multi-Device test suite on non T3K machine.";
        }
        mesh_device_ = MeshDevice::create(MeshDeviceConfig{.mesh_shape = MeshShape{2, 4}, .mesh_type = MeshType::Ring});
    }

    void TearDown() override {
        if (!mesh_device_) {
            return;
        }

        mesh_device_->close();
        mesh_device_.reset();
    }
    std::shared_ptr<MeshDevice> mesh_device_;
};

}  // namespace ttnn::distributed::test
