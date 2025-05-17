// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <unistd.h> // For gethostname

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/host_api.hpp>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tt_metal/impl/context/metal_context.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using namespace tt::tt_metal::distributed::multihost;

class DistributedContextFixture : public ::testing::Test {
protected:
    void SetUp() override { initialize_distributed_context(0, nullptr); }
};

class MultiHostT3000MeshDeviceFixture : public T3000MeshDeviceFixture {
protected:
    void SetUp() override {
        initialize_distributed_context(0, nullptr);
        T3000MeshDeviceFixture::SetUp();
    }

    void TearDown() override {
        T3000MeshDeviceFixture::TearDown();
    }
};

} // namespace

TEST_F(DistributedContextFixture, TestCreation) {
    EXPECT_NO_THROW(MetalContext::instance().get_distributed_context());
}

TEST_F(MultiHostT3000MeshDeviceFixture, TestMeshDeviceCreation) {
    std::cout << "mesh_device_->num_devices(): " << this->mesh_device_->num_devices() << std::endl;
    EXPECT_EQ(this->mesh_device_->num_devices(), 8);
}

TEST(ApiTargetDevicesTest, TestMeshDeviceCreation1x1) {
    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 1)));
    EXPECT_EQ(mesh_device->num_devices(), 1);
}

TEST(ApiTargetDevicesTest, TestMeshDeviceCreation2x2) {
    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 2)));
    EXPECT_EQ(mesh_device->num_devices(), 4);
}

TEST(ApiTargetDevicesTest, TestDeviceOpenClose) {
    auto device = CreateDevice(0);
    CloseDevice(device);
}
/*
TEST(DISABLED_ApiClusterTest, SeparateClusters) {
    //std::unique_ptr<tt_ClusterDescriptor> cluster_desc1 = tt_ClusterDescriptor::create_from_yaml("generated_cluster.yaml");
    // First, pregenerate a cluster descriptor and save it to a file.
    // This will run topology discovery and touch all the devices.
    std::filesystem::path cluster_path = tt::umd::Cluster::create_cluster_descriptor()->serialize_to_file();

    // Now, the user can create the cluster descriptor without touching the devices.
    std::unique_ptr<tt_ClusterDescriptor> cluster_desc1 = tt_ClusterDescriptor::create_from_yaml(cluster_path);
    // You can test the cluster descriptor here to see if the topology matched the one you'd expect.
    // For example, you can check if the number of chips is correct, or number of pci devices, or nature of eth
    // connections.
    std::unordered_set<chip_id_t> all_chips = cluster_desc1->get_all_chips();
    std::unordered_map<chip_id_t, chip_id_t> chips_with_pcie = cluster_desc1->get_chips_with_mmio();
    auto eth_connections = cluster_desc1->get_ethernet_connections();

    if (all_chips.empty()) {
        GTEST_SKIP() << "No chips present on the system. Skipping test.";
    }
    // Now we can choose which chips to open. This can be hardcoded if you already have expected topology.
    // The first cluster will open the first chip only, and the second cluster will open the rest of them.
    std::unordered_set<chip_id_t> cluster_2x2_a = {0, 1, 4, 5};
    std::unique_ptr<tt::umd::Cluster> umd_cluster1 = std::make_unique<tt::umd::Cluster>(tt::umd::ClusterOptions{
        .target_devices = cluster_2x2_a,
        .cluster_descriptor = cluster_desc1,
    });

    auto target_device_ids = umd_cluster1->get_target_device_ids();
    for (auto chip : target_device_ids) {
        std::cout << "first cluster chip: " << chip << std::endl;
    }
    EXPECT_EQ(target_device_ids.size(), 2);
}
*/

TEST_F(MultiHostT3000MeshDeviceFixture, TestMeshDeviceCreationWithTargetDevices) {
    std::cout << "mesh_device_->num_devices(): " << this->mesh_device_->num_devices() << std::endl;
    EXPECT_EQ(this->mesh_device_->num_devices(), 4);
}


} // namespace tt::tt_metal::distributed::test

int main(int argc, char** argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);
    
    // Run the tests
    int result = RUN_ALL_TESTS();
    
    return result;
}