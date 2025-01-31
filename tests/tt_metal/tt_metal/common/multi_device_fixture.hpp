// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/host_api.hpp>
#include "llrt.hpp"
#include <tt-metalium/mesh_device.hpp>

#include "dispatch_fixture.hpp"
#include "umd/device/types/cluster_descriptor_types.h"
#include "tt_metal/test_utils/env_vars.hpp"

class MultiDeviceFixture : public DispatchFixture {
protected:
    void SetUp() override { this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()); }
};

class N300DeviceFixture : public MultiDeviceFixture {
protected:
    void SetUp() override {
        this->slow_dispatch_ = true;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch) {
            tt::log_info(tt::LogTest, "This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
            this->slow_dispatch_ = false;
            GTEST_SKIP();
        }

        MultiDeviceFixture::SetUp();

        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pci_devices = tt::tt_metal::GetNumPCIeDevices();
        if (this->arch_ == tt::ARCH::WORMHOLE_B0 && num_devices == 2 && num_pci_devices == 1) {
            std::vector<chip_id_t> ids;
            for (chip_id_t id = 0; id < num_devices; id++) {
                ids.push_back(id);
            }

            const auto& dispatch_core_config = tt::llrt::RunTimeOptions::get_instance().get_dispatch_core_config();
            tt::DevicePool::initialize(ids, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_config);
            this->devices_ = tt::DevicePool::instance().get_all_active_devices();
        } else {
            GTEST_SKIP();
        }
    }
};

template <int r, int c, int num_cqs>
class MeshDeviceFixture : public ::testing::Test {
protected:
    virtual void SetUp() override {
        using tt::tt_metal::distributed::MeshDevice;
        using tt::tt_metal::distributed::MeshDeviceConfig;
        using tt::tt_metal::distributed::MeshShape;

        constexpr uint32_t num_required_devices = r * c;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (slow_dispatch) {
            GTEST_SKIP() << "Skipping Mesh-Device test suite, since it can only be run in Fast Dispatch Mode.";
        }
        if (num_devices != num_required_devices or arch != tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << fmt::format(
                "Skipping Mesh-Device test suite on non-{} device machine.",
                num_required_devices == 8 ? "T3K" : "N300");
        }
        create_mesh_device();
    }

    void TearDown() override {
        if (!mesh_device_) {
            return;
        }

        mesh_device_->close();
        mesh_device_.reset();
    }

protected:
    virtual void create_mesh_device() {
        using tt::tt_metal::distributed::MeshDevice;
        using tt::tt_metal::distributed::MeshDeviceConfig;
        using tt::tt_metal::distributed::MeshShape;
        // Use ethernet dispatch for more than 1 CQ on T3K/N300
        DispatchCoreType core_type = (num_cqs >= 2) ? DispatchCoreType::ETH : DispatchCoreType::WORKER;
        mesh_device_ = MeshDevice::create(MeshDeviceConfig{.mesh_shape = MeshShape{r, c}}, 0, 0, num_cqs, core_type);
    }

    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;
};
// Individual test fixtures for T3K and N300 (single and multi-cq)
class T3000MeshDeviceFixture : public MeshDeviceFixture<2, 4, 1> {};
class N300MeshDeviceFixture : public MeshDeviceFixture<2, 1, 1> {};
class T3000MeshDeviceMultiCQFixture : public MeshDeviceFixture<2, 4, 2> {};
class N300MeshDeviceMultiCQFixture : public MeshDeviceFixture<2, 1, 2> {};
// Generic fixtures allowing a test to run on both N300 and T3K
using MeshFixtureTypes = ::testing::Types<T3000MeshDeviceFixture, N300MeshDeviceFixture>;
using MultiCQMeshFixtureTypes = ::testing::Types<T3000MeshDeviceMultiCQFixture, N300MeshDeviceMultiCQFixture>;
// Generic class that can be bound to specific test suites
template <typename T>
class MeshTestSuite : public T {};
