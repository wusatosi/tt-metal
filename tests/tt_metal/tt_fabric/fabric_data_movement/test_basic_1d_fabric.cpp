// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/fabric_host_interface.h>

#include "fabric_fixture.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <tt-metalium/distributed.hpp>

namespace tt::tt_fabric {
using namespace tt::tt_metal::distributed;
using namespace tt_metal;

TEST_F(T3000MeshDeviceFixture, TestUnicastRaw) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};
    std::pair<mesh_id_t, chip_id_t> src_mesh_chip_id;
    chip_id_t src_physical_device_id;
    std::pair<mesh_id_t, chip_id_t> dst_mesh_chip_id;
    chip_id_t dst_physical_device_id;
    MeshCoordinateRange sender_coord(mesh_device_->shape());
    MeshCoordinateRange receiver_coord(mesh_device_->shape());
    auto control_plane = tt::Cluster::instance().get_control_plane();

    // Find a device with a neighbour in the East direction
    bool connection_found = false;

    for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
        auto device = mesh_device_->get_device(coord);
        src_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());
        // Get neighbours within a mesh in the East direction
        auto neighbors = control_plane->get_intra_chip_neighbors(
            src_mesh_chip_id.first, src_mesh_chip_id.second, RoutingDirection::E);
        if (neighbors.size() > 0) {
            src_physical_device_id = device->id();
            sender_coord = MeshCoordinateRange({coord}, {coord});
            dst_mesh_chip_id = {src_mesh_chip_id.first, neighbors[0]};
            dst_physical_device_id = control_plane->get_physical_chip_id_from_mesh_chip_id(dst_mesh_chip_id);

            for (const auto& candidate_coord : MeshCoordinateRange(mesh_device_->shape())) {
                if (mesh_device_->get_device(candidate_coord)->id() == dst_physical_device_id) {
                    receiver_coord = MeshCoordinateRange({candidate_coord}, {candidate_coord});
                    break;
                }
            }
            connection_found = true;
            break;
        }
    }
    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    // get a port to connect to
    std::vector<chan_id_t> eth_chans = control_plane->get_active_fabric_eth_channels_in_direction(
        src_mesh_chip_id.first, src_mesh_chip_id.second, RoutingDirection::E);
    if (eth_chans.size() == 0) {
        GTEST_SKIP() << "No active eth chans to connect to";
    }

    auto edm_port = eth_chans[0];
    CoreCoord edm_eth_core =
        tt::Cluster::instance().get_virtual_eth_core_from_channel(src_physical_device_id, edm_port);

    CoreCoord sender_virtual_core = mesh_device_->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = mesh_device_->worker_core_from_logical_core(receiver_logical_core);

    auto receiver_noc_encoding = tt::tt_metal::hal.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // test parameters
    uint32_t packet_header_address = 0x25000;
    uint32_t source_l1_buffer_address = 0x30000;
    uint32_t packet_payload_size_bytes = 4096;
    uint32_t num_packets = 10;
    uint32_t num_hops = 1;
    uint32_t test_results_address = 0x100000;
    uint32_t test_results_size_bytes = 128;
    uint32_t target_address = 0x30000;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    // common compile time args for sender and receiver
    std::vector<uint32_t> compile_time_args = {
        test_results_address, test_results_size_bytes, target_address, 0 /* mcast_mode */
    };

    auto ccl_mesh_workload = CreateMeshWorkload();
    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_tx.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    std::vector<uint32_t> sender_runtime_args = {
        packet_header_address,
        source_l1_buffer_address,
        packet_payload_size_bytes,
        num_packets,
        receiver_noc_encoding,
        time_seed,
        num_hops};

    // append the EDM connection rt args
    static constexpr std::size_t edm_buffer_size =
        tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes +
        sizeof(tt::tt_fabric::PacketHeader);
    const auto edm_config = tt::tt_fabric::FabricEriscDatamoverConfig(edm_buffer_size, 1, 2);

    tt::tt_fabric::SenderWorkerAdapterSpec edm_connection = {
        .edm_noc_x = edm_eth_core.x,
        .edm_noc_y = edm_eth_core.y,
        .edm_buffer_base_addr = edm_config.sender_channels_base_address[0],
        .num_buffers_per_channel = edm_config.sender_channels_num_buffers[0],
        .edm_l1_sem_addr = edm_config.sender_channels_local_flow_control_semaphore_address[0],
        .edm_connection_handshake_addr = edm_config.sender_channels_connection_semaphore_address[0],
        .edm_worker_location_info_addr = edm_config.sender_channels_worker_conn_info_base_address[0],
        .buffer_size_bytes = edm_config.channel_buffer_size_bytes,
        .buffer_index_semaphore_id = edm_config.sender_channels_buffer_index_semaphore_address[0],
        .persistent_fabric = true};

    auto worker_flow_control_semaphore_id = tt_metal::CreateSemaphore(sender_program, sender_logical_core, 0);
    auto worker_teardown_semaphore_id = tt_metal::CreateSemaphore(sender_program, sender_logical_core, 0);
    auto worker_buffer_index_semaphore_id = tt_metal::CreateSemaphore(sender_program, sender_logical_core, 0);

    append_worker_to_fabric_edm_sender_rt_args(
        edm_connection,
        worker_flow_control_semaphore_id,
        worker_teardown_semaphore_id,
        worker_buffer_index_semaphore_id,
        sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    std::vector<uint32_t> receiver_runtime_args = {packet_payload_size_bytes, num_packets, time_seed};

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    AddProgramToMeshWorkload(ccl_mesh_workload, std::move(sender_program), sender_coord);
    AddProgramToMeshWorkload(ccl_mesh_workload, std::move(receiver_program), receiver_coord);

    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), ccl_mesh_workload, true);

    // Validate the status and packets processed by sender and receiver
    std::vector<uint32_t> sender_status;
    std::vector<uint32_t> receiver_status;

    tt_metal::detail::ReadFromDeviceL1(
        DevicePool::instance().get_active_device(src_physical_device_id),
        sender_logical_core,
        test_results_address,
        test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    tt_metal::detail::ReadFromDeviceL1(
        DevicePool::instance().get_active_device(dst_physical_device_id),
        receiver_logical_core,
        test_results_address,
        test_results_size_bytes,
        receiver_status,
        CoreType::WORKER);

    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    EXPECT_EQ(receiver_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);

    uint64_t sender_bytes =
        ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];
    uint64_t receiver_bytes =
        ((uint64_t)receiver_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | receiver_status[TT_FABRIC_WORD_CNT_INDEX];
    EXPECT_EQ(sender_bytes, receiver_bytes);
}

TEST_F(T3000MeshDeviceFixture, TestUnicastConnAPI) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};
    std::pair<mesh_id_t, chip_id_t> src_mesh_chip_id;
    chip_id_t src_physical_device_id;
    std::pair<mesh_id_t, chip_id_t> dst_mesh_chip_id;
    chip_id_t dst_physical_device_id;
    MeshCoordinateRange sender_coord(mesh_device_->shape());
    MeshCoordinateRange receiver_coord(mesh_device_->shape());

    auto control_plane = tt::Cluster::instance().get_control_plane();

    // Find a device with a neighbour in the East direction
    bool connection_found = false;

    for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
        auto device = mesh_device_->get_device(coord);
        src_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());
        // Get neighbours within a mesh in the East direction
        auto neighbors = control_plane->get_intra_chip_neighbors(
            src_mesh_chip_id.first, src_mesh_chip_id.second, RoutingDirection::E);
        if (neighbors.size() > 0) {
            src_physical_device_id = device->id();
            sender_coord = MeshCoordinateRange({coord}, {coord});
            dst_mesh_chip_id = {src_mesh_chip_id.first, neighbors[0]};
            dst_physical_device_id = control_plane->get_physical_chip_id_from_mesh_chip_id(dst_mesh_chip_id);

            for (const auto& candidate_coord : MeshCoordinateRange(mesh_device_->shape())) {
                if (mesh_device_->get_device(candidate_coord)->id() == dst_physical_device_id) {
                    receiver_coord = MeshCoordinateRange({candidate_coord}, {candidate_coord});
                    ;
                    break;
                }
            }
            connection_found = true;
            break;
        }
    }
    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    CoreCoord sender_virtual_core = mesh_device_->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = mesh_device_->worker_core_from_logical_core(receiver_logical_core);

    auto receiver_noc_encoding = tt::tt_metal::hal.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // test parameters
    uint32_t packet_header_address = 0x25000;
    uint32_t source_l1_buffer_address = 0x30000;
    uint32_t packet_payload_size_bytes = 4096;
    uint32_t num_packets = 10;
    uint32_t num_hops = 1;
    uint32_t test_results_address = 0x100000;
    uint32_t test_results_size_bytes = 128;
    uint32_t target_address = 0x30000;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    // common compile time args for sender and receiver
    std::vector<uint32_t> compile_time_args = {
        test_results_address, test_results_size_bytes, target_address, 0 /* mcast_mode */
    };
    auto ccl_mesh_workload = CreateMeshWorkload();
    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_tx.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    std::vector<uint32_t> sender_runtime_args = {
        packet_header_address,
        source_l1_buffer_address,
        packet_payload_size_bytes,
        num_packets,
        receiver_noc_encoding,
        time_seed,
        num_hops};

    // append the EDM connection rt args
    append_fabric_connection_rt_args(
        src_physical_device_id, dst_physical_device_id, 0, sender_program, {sender_logical_core}, sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    std::vector<uint32_t> receiver_runtime_args = {packet_payload_size_bytes, num_packets, time_seed};

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    AddProgramToMeshWorkload(ccl_mesh_workload, std::move(sender_program), sender_coord);
    AddProgramToMeshWorkload(ccl_mesh_workload, std::move(receiver_program), receiver_coord);

    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), ccl_mesh_workload, true);

    // Validate the status and packets processed by sender and receiver
    std::vector<uint32_t> sender_status;
    std::vector<uint32_t> receiver_status;

    tt_metal::detail::ReadFromDeviceL1(
        DevicePool::instance().get_active_device(src_physical_device_id),
        sender_logical_core,
        test_results_address,
        test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    tt_metal::detail::ReadFromDeviceL1(
        DevicePool::instance().get_active_device(dst_physical_device_id),
        receiver_logical_core,
        test_results_address,
        test_results_size_bytes,
        receiver_status,
        CoreType::WORKER);

    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    EXPECT_EQ(receiver_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);

    uint64_t sender_bytes =
        ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];
    uint64_t receiver_bytes =
        ((uint64_t)receiver_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | receiver_status[TT_FABRIC_WORD_CNT_INDEX];
    EXPECT_EQ(sender_bytes, receiver_bytes);
}

TEST_F(T3000MeshDeviceFixture, TestMCastConnAPI) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};

    auto control_plane = tt::Cluster::instance().get_control_plane();

    // use control plane to find a mesh with 3 devices
    auto user_meshes = control_plane->get_user_physical_mesh_ids();
    mesh_id_t mesh_id;
    bool mesh_found = false;
    for (const auto& mesh : user_meshes) {
        auto mesh_shape = control_plane->get_physical_mesh_shape(mesh);
        if (mesh_shape.mesh_size() >= 3) {
            mesh_id = mesh;
            mesh_found = true;
            break;
        }
    }
    if (!mesh_found) {
        GTEST_SKIP() << "No mesh found for 3 chip mcast test";
    }

    // for this test, logical chip id 1 is the sender, 0 is the left receiver and 1 is the right receiver
    auto src_phys_chip_id = control_plane->get_physical_chip_id_from_mesh_chip_id(std::make_pair(mesh_id, 1));
    auto left_recv_phys_chip_id = control_plane->get_physical_chip_id_from_mesh_chip_id(std::make_pair(mesh_id, 0));
    ;
    auto right_recv_phys_chip_id = control_plane->get_physical_chip_id_from_mesh_chip_id(std::make_pair(mesh_id, 2));
    ;

    MeshCoordinateRange sender_coord(mesh_device_->shape());
    MeshCoordinateRange left_recv_coord(mesh_device_->shape());
    MeshCoordinateRange right_recv_coord(mesh_device_->shape());

    for (const auto& candidate_coord : MeshCoordinateRange(mesh_device_->shape())) {
        if (src_phys_chip_id == mesh_device_->get_device(candidate_coord)->id()) {
            sender_coord = MeshCoordinateRange({candidate_coord}, {candidate_coord});
        } else if (left_recv_phys_chip_id == mesh_device_->get_device(candidate_coord)->id()) {
            left_recv_coord = MeshCoordinateRange({candidate_coord}, {candidate_coord});
        } else if (right_recv_phys_chip_id == mesh_device_->get_device(candidate_coord)->id()) {
            right_recv_coord = MeshCoordinateRange({candidate_coord}, {candidate_coord});
        }
    }

    CoreCoord sender_virtual_core = mesh_device_->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = mesh_device_->worker_core_from_logical_core(receiver_logical_core);

    auto receiver_noc_encoding = tt::tt_metal::hal.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // test parameters
    uint32_t packet_header_address = 0x25000;
    uint32_t source_l1_buffer_address = 0x30000;
    uint32_t packet_payload_size_bytes = 4096;
    uint32_t num_packets = 100;
    uint32_t test_results_address = 0x100000;
    uint32_t test_results_size_bytes = 128;
    uint32_t target_address = 0x30000;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    // common compile time args for sender and receiver
    std::vector<uint32_t> compile_time_args = {
        test_results_address, test_results_size_bytes, target_address, 1 /* mcast_mode */
    };

    auto ccl_mesh_workload = CreateMeshWorkload();
    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_tx.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    std::vector<uint32_t> sender_runtime_args = {
        packet_header_address,
        source_l1_buffer_address,
        packet_payload_size_bytes,
        num_packets,
        receiver_noc_encoding,
        time_seed,
        1, /* mcast_fwd_hops */
        1, /* mcast_bwd_hops */
    };

    // append the EDM connection rt args for fwd connection
    append_fabric_connection_rt_args(
        src_phys_chip_id, right_recv_phys_chip_id, 0, sender_program, {sender_logical_core}, sender_runtime_args);
    append_fabric_connection_rt_args(
        src_phys_chip_id, left_recv_phys_chip_id, 0, sender_program, {sender_logical_core}, sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    std::vector<uint32_t> receiver_runtime_args = {packet_payload_size_bytes, num_packets, time_seed};

    // Create the receiver program for validation
    auto receiver_program_0 = tt_metal::CreateProgram();
    auto receiver_kernel_0 = tt_metal::CreateKernel(
        receiver_program_0,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    tt_metal::SetRuntimeArgs(receiver_program_0, receiver_kernel_0, receiver_logical_core, receiver_runtime_args);

    auto receiver_program_1 = tt_metal::CreateProgram();
    auto receiver_kernel_1 = tt_metal::CreateKernel(
        receiver_program_1,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    tt_metal::SetRuntimeArgs(receiver_program_1, receiver_kernel_1, receiver_logical_core, receiver_runtime_args);

    AddProgramToMeshWorkload(ccl_mesh_workload, std::move(sender_program), sender_coord);
    AddProgramToMeshWorkload(ccl_mesh_workload, std::move(receiver_program_0), left_recv_coord);
    AddProgramToMeshWorkload(ccl_mesh_workload, std::move(receiver_program_1), right_recv_coord);

    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), ccl_mesh_workload, true);

    // Validate the status and packets processed by sender and receiver
    std::vector<uint32_t> sender_status;
    std::vector<uint32_t> left_recv_status;
    std::vector<uint32_t> right_recv_status;

    tt_metal::detail::ReadFromDeviceL1(
        DevicePool::instance().get_active_device(src_phys_chip_id),
        sender_logical_core,
        test_results_address,
        test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    tt_metal::detail::ReadFromDeviceL1(
        DevicePool::instance().get_active_device(left_recv_phys_chip_id),
        receiver_logical_core,
        test_results_address,
        test_results_size_bytes,
        left_recv_status,
        CoreType::WORKER);

    tt_metal::detail::ReadFromDeviceL1(
        DevicePool::instance().get_active_device(right_recv_phys_chip_id),
        receiver_logical_core,
        test_results_address,
        test_results_size_bytes,
        right_recv_status,
        CoreType::WORKER);

    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    EXPECT_EQ(left_recv_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    EXPECT_EQ(right_recv_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);

    uint64_t sender_bytes =
        ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];
    uint64_t left_recv_bytes =
        ((uint64_t)left_recv_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | left_recv_status[TT_FABRIC_WORD_CNT_INDEX];
    uint64_t right_recv_bytes =
        ((uint64_t)right_recv_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | right_recv_status[TT_FABRIC_WORD_CNT_INDEX];

    EXPECT_EQ(sender_bytes, left_recv_bytes);
    EXPECT_EQ(left_recv_bytes, right_recv_bytes);
}

}  // namespace tt::tt_fabric
