// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/work_split.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <algorithm>
#include <random>
#include "gmock/gmock.h"

#include "tt_metal/hw/inc/socket.h"

namespace tt::tt_metal::distributed {

using MeshSocketTest = T3000MeshDeviceFixture;

// Sanity test with a single connection
TEST_F(MeshSocketTest, SingleConnectionSingleDeviceConfig) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    auto current_device_id = md0->get_device(MeshCoordinate(0, 0))->id();
    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 1);
    auto sender_virtual_coord = md0->worker_core_from_logical_core(sender_logical_coord);
    auto recv_virtual_coord = md0->worker_core_from_logical_core(recv_logical_coord);
    std::size_t socket_fifo_size = 1024;

    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);

    socket_connection_t socket_connection = {
        .sender_core = {MeshCoordinate(0, 0), sender_logical_coord},
        .receiver_core = {MeshCoordinate(0, 0), recv_logical_coord},
    };

    socket_memory_config_t socket_mem_config = {
        .socket_type = BufferType::L1,
        .fifo_size = socket_fifo_size,
    };

    socket_config_t socket_config = {
        .socket_connection_config = {socket_connection},
        .socket_mem_config = socket_mem_config,
    };
    auto [send_socket, recv_socket] = create_sockets(md0, md0, socket_config);

    std::vector<sender_socket_md> sender_config_readback;
    std::vector<receiver_socket_md> recv_config_readback;

    ReadShard(md0->mesh_command_queue(), sender_config_readback, send_socket.config_buffer, MeshCoordinate(0, 0));
    ReadShard(md0->mesh_command_queue(), recv_config_readback, recv_socket.config_buffer, MeshCoordinate(0, 0));

    EXPECT_EQ(sender_config_readback.size(), 1);
    EXPECT_EQ(recv_config_readback.size(), 1);

    const auto& sender_config = sender_config_readback[0];
    const auto& recv_config = recv_config_readback[0];

    // Validate Sender Config
    EXPECT_EQ(sender_config.bytes_acked, 0);
    EXPECT_EQ(sender_config.write_ptr, send_socket.data_buffer->address());
    EXPECT_EQ(sender_config.bytes_sent, 0);
    EXPECT_EQ(sender_config.downstream_mesh_id, 0);
    EXPECT_EQ(sender_config.downstream_chip_id, current_device_id);
    EXPECT_EQ(sender_config.downstream_noc_y, recv_virtual_coord.y);
    EXPECT_EQ(sender_config.downstream_noc_x, recv_virtual_coord.x);
    EXPECT_EQ(sender_config.downstream_bytes_sent_addr, recv_socket.config_buffer->address());
    EXPECT_EQ(sender_config.downstream_fifo_addr, send_socket.data_buffer->address());
    EXPECT_EQ(sender_config.downstream_fifo_total_size, socket_fifo_size);
    EXPECT_EQ(sender_config.is_sender, 1);
    EXPECT_EQ(sender_config.downstream_bytes_sent_addr % l1_alignment, 0);

    // Validate Receiver Config
    EXPECT_EQ(recv_config.bytes_sent, 0);
    EXPECT_EQ(recv_config.bytes_acked, 0);
    EXPECT_EQ(recv_config.read_ptr, recv_socket.data_buffer->address());
    EXPECT_EQ(recv_config.fifo_addr, recv_socket.data_buffer->address());
    EXPECT_EQ(recv_config.fifo_total_size, socket_fifo_size);
    EXPECT_EQ(recv_config.upstream_mesh_id, 0);
    EXPECT_EQ(recv_config.upstream_chip_id, current_device_id);
    EXPECT_EQ(recv_config.upstream_noc_y, sender_virtual_coord.y);
    EXPECT_EQ(recv_config.upstream_noc_x, sender_virtual_coord.x);
    EXPECT_EQ(recv_config.upstream_bytes_acked_addr, send_socket.config_buffer->address());
    EXPECT_EQ(recv_config.upstream_bytes_acked_addr % l1_alignment, 0);
}

// Test multiple connections
TEST_F(MeshSocketTest, MultiConnectionSingleDeviceConfig) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    auto current_device_id = md0->get_device(MeshCoordinate(0, 0))->id();
    std::size_t socket_fifo_size = 1024;
    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    const auto& worker_grid = md0->compute_with_storage_grid_size();
    std::vector<CoreCoord> sender_logical_coords;
    std::vector<CoreCoord> recv_logical_coords;

    for (std::size_t x = 0; x < worker_grid.x; x += 2) {
        if (x + 1 >= worker_grid.x) {
            continue;
        }
        for (std::size_t y = 0; y < worker_grid.y; y++) {
            sender_logical_coords.push_back(CoreCoord(x, y));
            recv_logical_coords.push_back(CoreCoord(x + 1, y));
        }
    }

    std::vector<socket_connection_t> socket_connections;

    for (std::size_t core_idx = 0; core_idx < sender_logical_coords.size(); core_idx++) {
        socket_connections.push_back(socket_connection_t{
            .sender_core = {MeshCoordinate(0, 0), sender_logical_coords[core_idx]},
            .receiver_core = {MeshCoordinate(0, 0), recv_logical_coords[core_idx]}});
    }

    socket_memory_config_t socket_mem_config = {
        .socket_type = BufferType::L1,
        .fifo_size = socket_fifo_size,
    };

    socket_config_t socket_config = {
        .socket_connection_config = socket_connections,
        .socket_mem_config = socket_mem_config,
    };

    auto [send_socket, recv_socket] = create_sockets(md0, md0, socket_config);

    std::vector<sender_socket_md> sender_configs;
    std::vector<receiver_socket_md> recv_configs;

    ReadShard(md0->mesh_command_queue(), sender_configs, send_socket.config_buffer, MeshCoordinate(0, 0));
    ReadShard(md0->mesh_command_queue(), recv_configs, recv_socket.config_buffer, MeshCoordinate(0, 0));

    EXPECT_EQ(sender_configs.size(), sender_logical_coords.size());
    EXPECT_EQ(recv_configs.size(), recv_logical_coords.size());

    const auto& sender_core_to_core_id =
        send_socket.config_buffer->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id_;

    const auto& recv_core_to_core_id =
        recv_socket.config_buffer->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id_;

    for (const auto& connection : socket_connections) {
        const auto& sender = connection.sender_core;
        const auto& recv = connection.receiver_core;
        auto sender_idx = sender_core_to_core_id.at(sender.second);
        auto recv_idx = recv_core_to_core_id.at(recv.second);

        const auto& sender_config = sender_configs[sender_idx];
        const auto& recv_config = recv_configs[recv_idx];

        auto sender_virtual_coord = md0->worker_core_from_logical_core(sender.second);
        auto recv_virtual_coord = md0->worker_core_from_logical_core(recv.second);

        // Validate Sender Configs
        EXPECT_EQ(sender_config.bytes_acked, 0);
        EXPECT_EQ(sender_config.write_ptr, send_socket.data_buffer->address());
        EXPECT_EQ(sender_config.bytes_sent, 0);
        EXPECT_EQ(sender_config.downstream_mesh_id, 0);
        EXPECT_EQ(sender_config.downstream_chip_id, current_device_id);
        EXPECT_EQ(sender_config.downstream_noc_y, recv_virtual_coord.y);
        EXPECT_EQ(sender_config.downstream_noc_x, recv_virtual_coord.x);
        EXPECT_EQ(sender_config.downstream_bytes_sent_addr, recv_socket.config_buffer->address());
        EXPECT_EQ(sender_config.downstream_fifo_addr, send_socket.data_buffer->address());
        EXPECT_EQ(sender_config.downstream_fifo_total_size, socket_fifo_size);
        EXPECT_EQ(sender_config.is_sender, 1);
        EXPECT_EQ(sender_config.downstream_bytes_sent_addr % l1_alignment, 0);

        // Validate Recv Configs
        EXPECT_EQ(recv_config.bytes_sent, 0);
        EXPECT_EQ(recv_config.bytes_acked, 0);
        EXPECT_EQ(recv_config.read_ptr, recv_socket.data_buffer->address());
        EXPECT_EQ(recv_config.fifo_addr, recv_socket.data_buffer->address());
        EXPECT_EQ(recv_config.fifo_total_size, socket_fifo_size);
        EXPECT_EQ(recv_config.upstream_mesh_id, 0);
        EXPECT_EQ(recv_config.upstream_chip_id, current_device_id);
        EXPECT_EQ(recv_config.upstream_noc_y, sender_virtual_coord.y);
        EXPECT_EQ(recv_config.upstream_noc_x, sender_virtual_coord.x);
        EXPECT_EQ(recv_config.upstream_bytes_acked_addr, send_socket.config_buffer->address());
        EXPECT_EQ(recv_config.upstream_bytes_acked_addr % l1_alignment, 0);
    }
}

TEST_F(MeshSocketTest, MultiConnectionMultiDeviceTest) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 4), MeshCoordinate(0, 0));
    auto md1 = mesh_device_->create_submesh(MeshShape(1, 4), MeshCoordinate(1, 0));
    std::unordered_map<MeshCoordinate, chip_id_t> sender_device_coord_to_id;
    std::unordered_map<MeshCoordinate, chip_id_t> receiver_device_coord_to_id;

    for (const auto& coord : MeshCoordinateRange(md0->shape())) {
        sender_device_coord_to_id[coord] = md0->get_device(coord)->id();
    }

    for (const auto& coord : MeshCoordinateRange(md1->shape())) {
        receiver_device_coord_to_id[coord] = md1->get_device(coord)->id();
    }
    std::size_t socket_fifo_size = 1024;
    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    const auto& worker_grid = md0->compute_with_storage_grid_size();

    std::vector<CoreCoord> sender_logical_coords;
    std::vector<CoreCoord> recv_logical_coords;
    std::vector<MeshCoordinate> sender_device_coords;
    std::vector<MeshCoordinate> recv_device_coords;
    uint32_t core_idx = 0;
    for (std::size_t x = 0; x < worker_grid.x; x++) {
        for (std::size_t y = 0; y < worker_grid.y; y++) {
            sender_logical_coords.push_back(CoreCoord(x, y));
            recv_logical_coords.push_back(CoreCoord(x, y));
            sender_device_coords.push_back(MeshCoordinate(0, core_idx % 4));
            recv_device_coords.push_back(MeshCoordinate(0, core_idx % 4));
            core_idx++;
        }
    }

    // Shuffle core coordinates to randomize the connections
    std::random_device rd;
    std::mt19937 generator(rd());
    std::shuffle(sender_logical_coords.begin(), sender_logical_coords.end(), generator);
    std::shuffle(recv_logical_coords.begin(), recv_logical_coords.end(), generator);
    std::shuffle(sender_device_coords.begin(), sender_device_coords.end(), generator);
    std::shuffle(recv_device_coords.begin(), recv_device_coords.end(), generator);

    std::vector<socket_connection_t> socket_connections;

    for (std::size_t coord_idx = 0; coord_idx < sender_logical_coords.size(); coord_idx++) {
        std::cout << "Create Connection: "
                  << "Sender: (" << sender_device_coords[coord_idx] << ", " << sender_logical_coords[coord_idx].str()
                  << ") "
                  << "Receiver: (" << recv_device_coords[coord_idx] << ", " << recv_logical_coords[coord_idx].str()
                  << ")" << std::endl;
        socket_connection_t socket_connection = {
            .sender_core = {sender_device_coords[coord_idx], sender_logical_coords[coord_idx]},
            .receiver_core = {recv_device_coords[coord_idx], recv_logical_coords[coord_idx]}};
        socket_connections.push_back(socket_connection);
    }

    socket_config_t socket_config_l1 = {
        .socket_connection_config = socket_connections,
        .socket_mem_config =
            {
                .socket_type = BufferType::L1,
                .fifo_size = socket_fifo_size,
            },
    };
    socket_config_t socket_config_dram = {
        .socket_connection_config = socket_connections,
        .socket_mem_config =
            {
                .socket_type = BufferType::DRAM,
                .fifo_size = socket_fifo_size,
            },
    };

    auto [send_socket_l1, recv_socket_l1] = create_sockets(md0, md1, socket_config_l1);
    auto [send_socket_dram, recv_socket_dram] = create_sockets(md0, md1, socket_config_dram);

    const auto& sender_core_to_core_id =
        send_socket_l1.config_buffer->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id_;

    const auto& recv_core_to_core_id =
        recv_socket_l1.config_buffer->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id_;

    std::unordered_map<MeshCoordinate, std::vector<sender_socket_md>> sender_configs_per_dev_coord;
    std::unordered_map<MeshCoordinate, std::vector<receiver_socket_md>> recv_configs_per_dev_coord;

    for (const auto& device_coord : MeshCoordinateRange(md0->shape())) {
        std::vector<sender_socket_md> sender_configs;
        std::vector<receiver_socket_md> recv_configs;

        ReadShard(md0->mesh_command_queue(), sender_configs, send_socket_l1.config_buffer, device_coord);
        ReadShard(md1->mesh_command_queue(), recv_configs, recv_socket_l1.config_buffer, device_coord);

        sender_configs_per_dev_coord[device_coord] = std::move(sender_configs);
        recv_configs_per_dev_coord[device_coord] = std::move(recv_configs);
    }

    for (const auto& connection : socket_connections) {
        const auto& sender_core = connection.sender_core;
        const auto& recv_core = connection.receiver_core;
        const auto& sender_device_coord = sender_core.first;
        const auto& recv_device_coord = recv_core.first;
        const auto& sender_core_coord = sender_core.second;
        const auto& recv_core_coord = recv_core.second;

        auto sender_idx = sender_core_to_core_id.at(sender_core_coord);
        auto recv_idx = recv_core_to_core_id.at(recv_core_coord);

        auto sender_virtual_coord = md0->worker_core_from_logical_core(sender_core_coord);
        auto recv_virtual_coord = md1->worker_core_from_logical_core(recv_core_coord);
        auto sender_device_id = sender_device_coord_to_id[sender_device_coord];
        auto receiver_device_id = receiver_device_coord_to_id[recv_device_coord];

        const auto& sender_config = sender_configs_per_dev_coord[sender_device_coord][sender_idx];
        const auto& recv_config = recv_configs_per_dev_coord[recv_device_coord][recv_idx];

        // Validate Sender Configs
        EXPECT_EQ(sender_config.bytes_acked, 0);
        EXPECT_EQ(sender_config.write_ptr, send_socket_l1.data_buffer->address());
        EXPECT_EQ(sender_config.bytes_sent, 0);
        EXPECT_EQ(sender_config.downstream_mesh_id, 0);
        EXPECT_EQ(sender_config.downstream_chip_id, receiver_device_id);
        EXPECT_EQ(sender_config.downstream_noc_y, recv_virtual_coord.y);
        EXPECT_EQ(sender_config.downstream_noc_x, recv_virtual_coord.x);
        EXPECT_EQ(sender_config.downstream_bytes_sent_addr, recv_socket_l1.config_buffer->address());
        EXPECT_EQ(sender_config.downstream_fifo_addr, send_socket_l1.data_buffer->address());
        EXPECT_EQ(sender_config.downstream_fifo_total_size, socket_fifo_size);
        EXPECT_EQ(sender_config.is_sender, 1);
        EXPECT_EQ(sender_config.downstream_bytes_sent_addr % l1_alignment, 0);

        // Validate Recv Configs
        EXPECT_EQ(recv_config.bytes_sent, 0);
        EXPECT_EQ(recv_config.bytes_acked, 0);
        EXPECT_EQ(recv_config.read_ptr, recv_socket_l1.data_buffer->address());
        EXPECT_EQ(recv_config.fifo_addr, recv_socket_l1.data_buffer->address());
        EXPECT_EQ(recv_config.fifo_total_size, socket_fifo_size);
        EXPECT_EQ(recv_config.upstream_mesh_id, 0);
        EXPECT_EQ(recv_config.upstream_chip_id, sender_device_id);
        EXPECT_EQ(recv_config.upstream_noc_y, sender_virtual_coord.y);
        EXPECT_EQ(recv_config.upstream_noc_x, sender_virtual_coord.x);
        EXPECT_EQ(recv_config.upstream_bytes_acked_addr, send_socket_l1.config_buffer->address());
        EXPECT_EQ(recv_config.upstream_bytes_acked_addr % l1_alignment, 0);
    }
}

void test_single_connection_single_device_socket(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> md0,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size) {
    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 1);
    auto sender_virtual_coord = md0->worker_core_from_logical_core(sender_logical_coord);
    auto recv_virtual_coord = md0->worker_core_from_logical_core(recv_logical_coord);

    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);

    socket_connection_t socket_connection = {
        .sender_core = {MeshCoordinate(0, 0), sender_logical_coord},
        .receiver_core = {MeshCoordinate(0, 0), recv_logical_coord},
    };

    socket_memory_config_t socket_mem_config = {
        .socket_type = BufferType::L1,
        .fifo_size = socket_fifo_size,
    };

    socket_config_t socket_config = {
        .socket_connection_config = {socket_connection},
        .socket_mem_config = socket_mem_config,
    };
    auto [send_socket, recv_socket] = create_sockets(md0, md0, socket_config);

    auto sender_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(sender_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig sender_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = sender_data_shard_params,
        .bottom_up = false};

    auto recv_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(recv_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig recv_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = recv_data_shard_params,
        .bottom_up = false};

    const ReplicatedBufferConfig buffer_config{.size = data_size};

    auto sender_data_buffer = MeshBuffer::create(buffer_config, sender_device_local_config, md0.get());

    auto recv_data_buffer = MeshBuffer::create(buffer_config, recv_device_local_config, md0.get());

    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);

    WriteShard(md0->mesh_command_queue(), sender_data_buffer, src_vec, MeshCoordinate(0, 0));

    auto send_recv_program = CreateProgram();
    auto sender_kernel = CreateKernel(
        send_recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/sender.cpp",
        sender_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(send_socket.config_buffer->address()),
                static_cast<uint32_t>(sender_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size)}});

    auto recv_kernel = CreateKernel(
        send_recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_worker.cpp",
        recv_logical_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(recv_socket.config_buffer->address()),
                static_cast<uint32_t>(recv_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size)}});

    auto mesh_workload = CreateMeshWorkload();
    MeshCoordinateRange devices(md0->shape());

    AddProgramToMeshWorkload(mesh_workload, std::move(send_recv_program), devices);

    EnqueueMeshWorkload(md0->mesh_command_queue(), mesh_workload, false);

    std::vector<uint32_t> recv_data_readback;
    ReadShard(md0->mesh_command_queue(), recv_data_readback, recv_data_buffer, MeshCoordinate(0, 0));
    EXPECT_EQ(src_vec, recv_data_readback);
}

TEST_F(MeshSocketTest, SingleConnectionSingleDeviceSocket) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    // No wrap
    test_single_connection_single_device_socket(md0, 1024, 64, 1024);
    // Even wrap
    test_single_connection_single_device_socket(md0, 1024, 64, 2048);
    // Uneven wrap
    test_single_connection_single_device_socket(md0, 4096, 1088, 9792);
}

struct socket_core_mapping {
    CoreCoord sender_core;
    CoreCoord receiver_core;
    CoreRange worker_cores;
    CoreRangeSet data_cores;
    CoreRangeSet output_cores;
};

void test_single_connection_single_device_socket_with_workers(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> md0,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    tt::stl::Span<socket_core_mapping> socket_core_mappings,
    bool final_ack) {
    CoreRangeSet used_cores;
    uint32_t num_used_cores = 0;
    for (const auto& mapping : socket_core_mappings) {
        if (mapping.worker_cores.size() != mapping.output_cores.num_cores() ||
            mapping.worker_cores.size() != mapping.output_cores.num_cores()) {
            GTEST_SKIP() << "Worker and data/output core ranges must be the same size";
        }
        // TODO: Update this check for multi device
        used_cores = used_cores.merge(CoreRangeSet(mapping.sender_core));
        used_cores = used_cores.merge(CoreRangeSet(mapping.receiver_core));
        used_cores = used_cores.merge(CoreRangeSet(mapping.worker_cores));
        num_used_cores += mapping.worker_cores.size() + 2;
        if (used_cores.num_cores() != num_used_cores) {
            GTEST_SKIP() << "Socket core ranges must not overlap" << used_cores.num_cores() << " != " << num_used_cores;
        }
    }

    if (final_ack && socket_fifo_size < data_size) {
        GTEST_SKIP() << "Socket FIFO size must be greater than data size for final ack";
    }
    if (!final_ack && socket_fifo_size < 2 * page_size) {
        GTEST_SKIP() << "Socket FIFO size must be greater than 2 * page size for loop ack";
    }

    CoreCoord sender_logical_data_core = CoreCoord(0, 0);
    CoreCoord sender_virtual_data_core = md0->worker_core_from_logical_core(sender_logical_data_core);

    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);

    std::vector<socket_connection_t> socket_connections;
    socket_connections.reserve(socket_core_mappings.size());

    CoreRangeSet sender_crs;
    CoreRangeSet recv_worker_crs;
    CoreRangeSet output_crs;
    uint32_t num_data_cores = 0;
    uint32_t num_output_cores = 0;
    {
        std::vector<CoreRange> sender_logical_cr;
        sender_logical_cr.reserve(socket_core_mappings.size());
        std::vector<CoreRange> recv_worker_logical_cr;
        recv_worker_logical_cr.reserve(socket_core_mappings.size() * 2);
        std::vector<CoreRange> output_logical_cr;

        for (const auto& mapping : socket_core_mappings) {
            socket_connection_t socket_connection = {
                .sender_core = {MeshCoordinate(0, 0), mapping.sender_core},
                .receiver_core = {MeshCoordinate(0, 0), mapping.receiver_core}};
            socket_connections.push_back(socket_connection);

            sender_logical_cr.push_back(CoreRange(mapping.sender_core));
            recv_worker_logical_cr.push_back(CoreRange(mapping.receiver_core));
            recv_worker_logical_cr.push_back(mapping.worker_cores);

            output_logical_cr.insert(
                output_logical_cr.end(), mapping.output_cores.ranges().begin(), mapping.output_cores.ranges().end());
            num_data_cores += mapping.data_cores.num_cores();
            num_output_cores += mapping.output_cores.num_cores();
        }
        sender_crs = CoreRangeSet(sender_logical_cr);
        recv_worker_crs = CoreRangeSet(recv_worker_logical_cr);
        output_crs = CoreRangeSet(output_logical_cr);
    }

    socket_memory_config_t socket_mem_config = {
        .socket_type = BufferType::L1,
        .fifo_size = socket_fifo_size,
    };

    socket_config_t socket_config = {
        .socket_connection_config = socket_connections,
        .socket_mem_config = socket_mem_config,
    };

    auto [send_socket, recv_socket] = create_sockets(md0, md0, socket_config);

    auto sender_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(sender_logical_data_core), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig sender_device_local_config{
        .page_size = data_size * num_data_cores,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = sender_data_shard_params,
        .bottom_up = false};

    const ReplicatedBufferConfig sender_buffer_config{.size = data_size * num_data_cores};

    // Only used for allocation
    auto output_shard_params = ShardSpecBuffer(output_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig output_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = output_shard_params,
        .bottom_up = false};

    const ReplicatedBufferConfig output_buffer_config{.size = data_size * num_output_cores};

    auto sender_data_buffer = MeshBuffer::create(sender_buffer_config, sender_device_local_config, md0.get());

    auto output_buffer = MeshBuffer::create(output_buffer_config, output_device_local_config, md0.get());

    std::vector<uint32_t> src_vec(sender_buffer_config.size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);

    WriteShard(md0->mesh_command_queue(), sender_data_buffer, src_vec, MeshCoordinate(0, 0));

    auto send_recv_program = CreateProgram();

    auto sender_cb_index = tt::CBIndex::c_0;
    auto sender_cb_config = CircularBufferConfig(data_size, {{sender_cb_index, tt::DataFormat::UInt32}})
                                .set_page_size(sender_cb_index, data_size);
    auto sender_cb = CreateCircularBuffer(send_recv_program, sender_crs, sender_cb_config);

    // Create CB on both receiver and worker so that receiver knows the address
    auto config_cb_index = tt::CBIndex::c_0;
    auto config_cb_config =
        CircularBufferConfig(sizeof(receiver_socket_md), {{config_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(config_cb_index, sizeof(receiver_socket_md));
    auto config_cb = CreateCircularBuffer(send_recv_program, recv_worker_crs, config_cb_config);

    auto data_cb_index = tt::CBIndex::c_1;
    auto data_cb_config = CircularBufferConfig(2 * page_size, {{data_cb_index, tt::DataFormat::UInt32}})
                              .set_page_size(data_cb_index, page_size);
    // No need to create on recv core, but better dispatch to do so
    auto data_cb = CreateCircularBuffer(send_recv_program, recv_worker_crs, data_cb_config);

    auto config_sem = CreateSemaphore(send_recv_program, recv_worker_crs, 0);
    auto credits0_sem = CreateSemaphore(send_recv_program, recv_worker_crs, 0);

    uint32_t data_offset = 0;
    for (const auto& mapping : socket_core_mappings) {
        const auto& sender_logical_coord = mapping.sender_core;
        const auto& recv_logical_coord = mapping.receiver_core;
        auto recv_virtual_coord = md0->worker_core_from_logical_core(recv_logical_coord);
        auto worker_logical_coords = corerange_to_cores(mapping.worker_cores, std::nullopt, true);
        auto worker_virtual_coords = md0->worker_cores_from_logical_cores(worker_logical_coords);
        auto data_logical_coords = corerange_to_cores(mapping.data_cores, std::nullopt, true);
        auto data_virtual_coords = md0->worker_cores_from_logical_cores(data_logical_coords);
        auto output_logical_coords = corerange_to_cores(mapping.output_cores, std::nullopt, true);
        auto output_virtual_coords = md0->worker_cores_from_logical_cores(output_logical_coords);
        std::vector<uint32_t> data_virtual_xys;
        std::vector<uint32_t> output_virtual_xys;
        data_virtual_xys.reserve(data_virtual_coords.size() * 2);
        output_virtual_xys.reserve(output_virtual_coords.size() * 2);
        for (uint32_t j = 0; j < data_virtual_coords.size(); ++j) {
            data_virtual_xys.push_back(data_virtual_coords[j].x);
            data_virtual_xys.push_back(data_virtual_coords[j].y);
            output_virtual_xys.push_back(output_virtual_coords[j].x);
            output_virtual_xys.push_back(output_virtual_coords[j].y);
        }

        auto sender_kernel = CreateKernel(
            send_recv_program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/sender_multi_data.cpp",
            sender_logical_coord,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {
                    static_cast<uint32_t>(send_socket.config_buffer->address()),
                    static_cast<uint32_t>(sender_data_buffer->address() + data_offset),
                    static_cast<uint32_t>(page_size),
                    static_cast<uint32_t>(data_size),
                    static_cast<uint32_t>(data_logical_coords.size()),
                    static_cast<uint32_t>(sender_virtual_data_core.x),
                    static_cast<uint32_t>(sender_virtual_data_core.y),
                    static_cast<uint32_t>(sender_cb_index),
                }});

        const auto& sender_rtas = data_virtual_xys;
        SetRuntimeArgs(send_recv_program, sender_kernel, sender_logical_coord, sender_rtas);
        data_offset += data_logical_coords.size() * data_size;

        if (final_ack) {
            auto recv_kernel = CreateKernel(
                send_recv_program,
                "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_final_ack.cpp",
                recv_logical_coord,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = {
                        static_cast<uint32_t>(recv_socket.config_buffer->address()),
                        static_cast<uint32_t>(config_cb_index),
                        static_cast<uint32_t>(config_sem),
                        static_cast<uint32_t>(credits0_sem),
                        static_cast<uint32_t>(page_size),
                        static_cast<uint32_t>(data_size),
                        static_cast<uint32_t>(worker_virtual_coords.begin()->x),
                        static_cast<uint32_t>(worker_virtual_coords.begin()->y),
                        static_cast<uint32_t>(worker_virtual_coords.rbegin()->x),
                        static_cast<uint32_t>(worker_virtual_coords.rbegin()->y),
                        static_cast<uint32_t>(worker_virtual_coords.size()),
                    }});
            auto worker_kernel = CreateKernel(
                send_recv_program,
                "tests/tt_metal/tt_metal/test_kernels/misc/socket/worker_final_ack.cpp",
                mapping.worker_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = {
                        static_cast<uint32_t>(config_cb_index),
                        static_cast<uint32_t>(config_sem),
                        static_cast<uint32_t>(credits0_sem),
                        static_cast<uint32_t>(data_cb_index),
                        static_cast<uint32_t>(page_size),
                        static_cast<uint32_t>(data_size),
                        static_cast<uint32_t>(recv_virtual_coord.x),
                        static_cast<uint32_t>(recv_virtual_coord.y),
                        static_cast<uint32_t>(output_buffer->address()),
                    }});
            for (uint32_t j = 0; j < worker_logical_coords.size(); ++j) {
                std::vector<uint32_t> worker_rtas = {
                    static_cast<uint32_t>(data_virtual_coords[j].x),
                    static_cast<uint32_t>(data_virtual_coords[j].y),
                    static_cast<uint32_t>(output_virtual_coords[j].x),
                    static_cast<uint32_t>(output_virtual_coords[j].y),
                };
                SetRuntimeArgs(send_recv_program, worker_kernel, worker_logical_coords[j], worker_rtas);
            }
        } else {
            auto credits1_sem = CreateSemaphore(send_recv_program, recv_worker_crs, 0);
            auto recv_kernel = CreateKernel(
                send_recv_program,
                "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_loop_ack.cpp",
                recv_logical_coord,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = {
                        static_cast<uint32_t>(recv_socket.config_buffer->address()),
                        static_cast<uint32_t>(config_cb_index),
                        static_cast<uint32_t>(config_sem),
                        static_cast<uint32_t>(credits0_sem),
                        static_cast<uint32_t>(credits1_sem),
                        static_cast<uint32_t>(page_size),
                        static_cast<uint32_t>(data_size),
                        static_cast<uint32_t>(worker_virtual_coords.begin()->x),
                        static_cast<uint32_t>(worker_virtual_coords.begin()->y),
                        static_cast<uint32_t>(worker_virtual_coords.rbegin()->x),
                        static_cast<uint32_t>(worker_virtual_coords.rbegin()->y),
                        static_cast<uint32_t>(worker_virtual_coords.size()),
                    }});

            auto worker_kernel = CreateKernel(
                send_recv_program,
                "tests/tt_metal/tt_metal/test_kernels/misc/socket/worker_loop_ack.cpp",
                mapping.worker_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = {
                        static_cast<uint32_t>(config_cb_index),
                        static_cast<uint32_t>(config_sem),
                        static_cast<uint32_t>(credits0_sem),
                        static_cast<uint32_t>(credits1_sem),
                        static_cast<uint32_t>(data_cb_index),
                        static_cast<uint32_t>(page_size),
                        static_cast<uint32_t>(data_size),
                        static_cast<uint32_t>(recv_virtual_coord.x),
                        static_cast<uint32_t>(recv_virtual_coord.y),
                        static_cast<uint32_t>(output_buffer->address())}});
            for (uint32_t j = 0; j < worker_logical_coords.size(); ++j) {
                std::vector<uint32_t> worker_rtas = {
                    static_cast<uint32_t>(data_virtual_coords[j].x),
                    static_cast<uint32_t>(data_virtual_coords[j].y),
                    static_cast<uint32_t>(output_virtual_coords[j].x),
                    static_cast<uint32_t>(output_virtual_coords[j].y),
                };
                SetRuntimeArgs(send_recv_program, worker_kernel, worker_logical_coords[j], worker_rtas);
            }
        }
    }

    auto mesh_workload = CreateMeshWorkload();
    MeshCoordinateRange devices(md0->shape());

    AddProgramToMeshWorkload(mesh_workload, std::move(send_recv_program), devices);

    EnqueueMeshWorkload(md0->mesh_command_queue(), mesh_workload, false);

    uint8_t* src_ptr = reinterpret_cast<uint8_t*>(src_vec.data());
    uint32_t pages_per_core = data_size / page_size;
    for (const auto& mapping : socket_core_mappings) {
        std::vector<uint32_t> output_data_readback;
        uint32_t num_local_output_cores = mapping.output_cores.num_cores();
        auto local_output_shard_params = ShardSpecBuffer(
            mapping.output_cores,
            {pages_per_core, 1},
            ShardOrientation::ROW_MAJOR,
            {1, 1},
            {pages_per_core, num_local_output_cores});

        const DeviceLocalBufferConfig local_output_device_local_config{
            .page_size = page_size,
            .buffer_type = BufferType::L1,
            .buffer_layout = TensorMemoryLayout::WIDTH_SHARDED,
            .shard_parameters = local_output_shard_params,
            .bottom_up = false};

        const ReplicatedBufferConfig local_buffer_config{.size = data_size * num_local_output_cores};

        auto local_output_buffer = MeshBuffer::create(
            local_buffer_config, local_output_device_local_config, md0.get(), output_buffer->address());

        ReadShard(md0->mesh_command_queue(), output_data_readback, local_output_buffer, MeshCoordinate(0, 0));

        EXPECT_EQ(std::memcmp(src_ptr, output_data_readback.data(), local_buffer_config.size), 0);
        src_ptr += local_buffer_config.size;
    }
}

TEST_F(MeshSocketTest, SingleConnectionSingleDeviceSocketWithWorkersFinalAck) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    std::vector<socket_core_mapping> socket_core_mappings = {
        {.sender_core = {0, 0},
         .receiver_core = {1, 0},
         .worker_cores = CoreRange({0, 1}, {3, 2}),
         .data_cores = CoreRangeSet(CoreRange({0, 3}, {3, 4})),
         .output_cores = CoreRangeSet(CoreRange({0, 4}, {3, 5}))},
    };
    // These tests must not wrap and continue sending data
    test_single_connection_single_device_socket_with_workers(md0, 1024, 64, 512, socket_core_mappings, true);
    test_single_connection_single_device_socket_with_workers(md0, 1024, 64, 1024, socket_core_mappings, true);
}

TEST_F(MeshSocketTest, SingleConnectionSingleDeviceSocketWithWorkersLoopAck) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    std::vector<socket_core_mapping> socket_core_mappings = {
        {.sender_core = {0, 0},
         .receiver_core = {1, 0},
         .worker_cores = CoreRange({0, 1}, {3, 2}),
         .data_cores = CoreRangeSet(CoreRange({0, 3}, {3, 4})),
         .output_cores = CoreRangeSet(CoreRange({0, 4}, {3, 5}))},
    };
    // No wrap
    test_single_connection_single_device_socket_with_workers(md0, 1024, 64, 512, socket_core_mappings, false);
    test_single_connection_single_device_socket_with_workers(md0, 1024, 64, 1024, socket_core_mappings, false);
    // Even wrap
    test_single_connection_single_device_socket_with_workers(md0, 1024, 64, 2048, socket_core_mappings, false);
    // Uneven wrap
    test_single_connection_single_device_socket_with_workers(md0, 4096, 1088, 9792, socket_core_mappings, false);
}

TEST_F(MeshSocketTest, MultiConnectionSingleDeviceSocketWithWorkersFinalAck) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    std::vector<socket_core_mapping> socket_core_mappings = {
        {.sender_core = {0, 0},
         .receiver_core = {1, 0},
         .worker_cores = CoreRange({0, 1}, {1, 2}),
         .data_cores = CoreRangeSet(CoreRange({0, 3}, {1, 4})),
         .output_cores = CoreRangeSet(CoreRange({0, 4}, {1, 5}))},
        {.sender_core = {2, 0},
         .receiver_core = {3, 0},
         .worker_cores = CoreRange({2, 1}, {4, 2}),
         .data_cores = CoreRangeSet(CoreRange({2, 3}, {4, 4})),
         .output_cores = CoreRangeSet(CoreRange({2, 4}, {4, 5}))},
    };
    // These tests must not wrap and continue sending data
    test_single_connection_single_device_socket_with_workers(md0, 1024, 64, 512, socket_core_mappings, true);
    test_single_connection_single_device_socket_with_workers(md0, 1024, 64, 1024, socket_core_mappings, true);
}

TEST_F(MeshSocketTest, MultiConnectionSingleDeviceSocketWithWorkersLoopAck) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    std::vector<socket_core_mapping> socket_core_mappings = {
        {.sender_core = {0, 0},
         .receiver_core = {1, 0},
         .worker_cores = CoreRange({0, 1}, {1, 2}),
         .data_cores = CoreRangeSet(CoreRange({0, 3}, {1, 4})),
         .output_cores = CoreRangeSet(CoreRange({0, 4}, {1, 5}))},
        {.sender_core = {2, 0},
         .receiver_core = {3, 0},
         .worker_cores = CoreRange({2, 1}, {4, 2}),
         .data_cores = CoreRangeSet(CoreRange({2, 3}, {4, 4})),
         .output_cores = CoreRangeSet(CoreRange({2, 4}, {4, 5}))},
    };
    // No wrap
    test_single_connection_single_device_socket_with_workers(md0, 1024, 64, 512, socket_core_mappings, false);
    test_single_connection_single_device_socket_with_workers(md0, 1024, 64, 1024, socket_core_mappings, false);
    // Even wrap
    test_single_connection_single_device_socket_with_workers(md0, 1024, 64, 2048, socket_core_mappings, false);
    // Uneven wrap
    test_single_connection_single_device_socket_with_workers(md0, 4096, 1088, 9792, socket_core_mappings, false);
}

}  // namespace tt::tt_metal::distributed
