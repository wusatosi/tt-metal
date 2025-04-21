// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/logger.hpp>
#include <tt-metalium/metal_soc_descriptor.h>
#include <tt-metalium/program.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <stdint.h>
#include <numeric>
#include <algorithm>
#include <random>
#include <exception>
#include <optional>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>
#include <string>

#include "tests/tt_metal/test_utils/test_common.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "llrt.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_fabric_test_common.hpp"

using namespace tt;
using chan_id_t = std::uint8_t;

// Defaults
constexpr uint32_t default_prng_seed = 0xFFFFFFFF;
constexpr uint32_t default_num_packets = 5000;
constexpr uint32_t default_packet_payload_size_kb = 1;
const std::string default_board_type = "tg";

constexpr uint32_t default_num_routing_planes = 4;
constexpr uint32_t max_num_routing_planes = 4;

// test mode if the traffic is to be restricted b/w a pair of devices
constexpr uint32_t default_test_device_id_l = 0xFFFFFFFF;
constexpr uint32_t default_test_device_id_r = 0xFFFFFFFF;

constexpr uint32_t default_num_sender_chips = 0xFFFFFFFF;

constexpr uint32_t default_num_hops = 0xFFFFFFFF;

// Kernels
const std::string controller_kernel_src =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_controller_worker.cpp";
const std::string sender_kernel_src =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_sender_worker.cpp";
const std::string receiver_kernel_src =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_receiver_worker.cpp";
const std::string edm_kernel_src = "tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_datamover.cpp";

const auto routing_directions = {
    tt_fabric::RoutingDirection::N,
    tt_fabric::RoutingDirection::S,
    tt_fabric::RoutingDirection::E,
    tt_fabric::RoutingDirection::W};

// global random number generator
std::mt19937 global_rng;

// benchmark mode
bool benchmark_mode = false;

// mcast mode
bool mcast_mode = false;

// bidirectional traffic
bool bidirectional_mode = false;

// all-to-all mode -> a chip sends packets to all possible destinations
bool all_to_all_mode = false;

// extra logging
bool verbose_mode = false;

// number of routing planes determine the number of fabric kernels to launch on a given device
// launching on a single routing plane for instance can reduce setup time
uint32_t num_routing_planes;

// global test params
uint32_t tensix_unreserved_base_address = 0;
uint32_t packet_payload_size_kb = 0;
uint32_t packet_payload_size_bytes = 0;
uint32_t num_packets = 0;
uint32_t time_seed = 0;

// fold fabric connections at the corner chips to make an outer ring (could be useful for T3K)
bool wrap_around_mesh = false;

bool enable_ring = false;

// Metal fabric initialization level
// 0: No fabric initialization (setup via test framework)
// 1: Initialize metal fabric with default settings
uint32_t metal_fabric_init_level;

tt::tt_fabric::FabricEriscDatamoverConfig get_edm_config() {
    constexpr std::size_t edm_buffer_size =
        tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes +
        sizeof(tt::tt_fabric::PacketHeader);
    return tt::tt_fabric::FabricEriscDatamoverConfig(edm_buffer_size, tt_fabric::Topology::Linear);
}

template <typename T>
std::vector<T> select_random_elements(const std::vector<T>& input, uint32_t num_elements) {
    if (num_elements > input.size()) {
        throw std::runtime_error("Num elements out of bounds");
    }

    std::vector<T> copy_vector(input.begin(), input.end());
    std::shuffle(copy_vector.begin(), copy_vector.end(), global_rng);
    return std::vector<T>(copy_vector.begin(), copy_vector.begin() + num_elements);
}

inline std::vector<uint32_t> get_random_numbers_from_range(uint32_t start, uint32_t end, uint32_t count) {
    std::vector<uint32_t> range(end - start + 1);

    // generate and shuffle
    std::iota(range.begin(), range.end(), start);
    std::shuffle(range.begin(), range.end(), global_rng);
    return std::vector<uint32_t>(range.begin(), range.begin() + count);
}

struct TestBoard {
    std::vector<chip_id_t> physical_chip_ids;
    std::vector<chip_id_t> handshake_seq_chip_ids;
    std::map<chip_id_t, IDevice*> IDevice_handle_map;
    tt::tt_fabric::ControlPlane* control_plane;
    tt::tt_fabric::mesh_id_t mesh_id;  // currently supporting only one mesh

    // physical chip layout with the top left chip being the NW chip
    std::vector<std::vector<chip_id_t>> physical_chip_matrix;
    std::unordered_map<chip_id_t, std::pair<uint32_t, uint32_t>> physical_chip_coords;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;

    void init(std::string& board_type_) {
        auto num_available_devices = tt_metal::GetNumAvailableDevices();
        uint32_t num_expected_devices;
        uint32_t chip_id_offset = 0;

        if ("n300" == board_type_) {
            num_expected_devices = 2;
        } else if ("t3k" == board_type_) {
            num_expected_devices = 8;
        } else if ("tg" == board_type_) {
            num_expected_devices = 32;
            chip_id_offset = 4;
        } else if ("ubb" == board_type_) {
            num_expected_devices = 32;
        } else {
            throw std::runtime_error("Unsupported board type");
        }

        if (num_available_devices != num_expected_devices) {
            log_fatal(
                LogTest,
                "Expected {} devices for board type {}, got {}",
                num_expected_devices,
                board_type_,
                num_available_devices);
            throw std::runtime_error("Unexpected number of devices");
        }

        physical_chip_ids.resize(num_available_devices);
        std::iota(physical_chip_ids.begin(), physical_chip_ids.end(), chip_id_offset);

        if (metal_fabric_init_level == 0) {
            tt::tt_metal::detail::InitializeFabricConfig(tt::tt_metal::FabricConfig::CUSTOM);
        } else if (metal_fabric_init_level == 1) {
            tt::tt_metal::detail::InitializeFabricConfig(
                enable_ring ? tt::tt_metal::FabricConfig::FABRIC_1D_RING : tt::tt_metal::FabricConfig::FABRIC_1D);
        }

        IDevice_handle_map = tt::tt_metal::detail::CreateDevices(physical_chip_ids);
        tensix_unreserved_base_address =
            IDevice_handle_map.begin()->second->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

        control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();

        // Only 1 mesh is supported as of now
        mesh_id = control_plane->get_user_physical_mesh_ids()[0];

        _generate_chip_neighbors_map();

        if (metal_fabric_init_level == 0) {
            _generate_router_handshake_chip_sequence();
        }
    }

    bool is_valid_chip_id(chip_id_t physical_chip_id) {
        auto it = std::find(physical_chip_ids.begin(), physical_chip_ids.end(), physical_chip_id);
        return (it != physical_chip_ids.end());
    }

    std::vector<chip_id_t> get_available_chips() const { return physical_chip_ids; }

    IDevice* get_IDevice_handle(chip_id_t chip_id) { return IDevice_handle_map[chip_id]; }

    // return Manhattan distance b/w the 2 chips
    std::pair<int, int> get_distance_between_chips(chip_id_t src_chip_id, chip_id_t dst_chip_id) {
        auto src_chip_coords = _get_chip_coords(src_chip_id);
        auto dst_chip_coords = _get_chip_coords(dst_chip_id);
        return std::make_pair(
            (dst_chip_coords.first - src_chip_coords.first), (dst_chip_coords.second - src_chip_coords.second));
    }

    std::unordered_map<tt_fabric::RoutingDirection, uint32_t> get_hops_between_chips(
        chip_id_t src_chip_id, chip_id_t dst_chip_id) {
        std::unordered_map<tt_fabric::RoutingDirection, uint32_t> hops_counts;
        auto [x_dist, y_dist] = get_distance_between_chips(src_chip_id, dst_chip_id);

        // TODO: remove the warning once 2D support is merged
        if (std::abs(x_dist) > 0 && std::abs(y_dist) > 0) {
            log_warning(LogTest, "Src chip: {} and dst chip: {} are not on the same line", src_chip_id, dst_chip_id);
        }

        hops_counts[tt_fabric::RoutingDirection::N] = y_dist < 0 ? std::abs(y_dist) : 0;
        hops_counts[tt_fabric::RoutingDirection::S] = y_dist > 0 ? y_dist : 0;
        hops_counts[tt_fabric::RoutingDirection::E] = x_dist > 0 ? x_dist : 0;
        hops_counts[tt_fabric::RoutingDirection::W] = x_dist < 0 ? std::abs(x_dist) : 0;
        return hops_counts;
    }

    uint32_t get_max_hops_in_direction(chip_id_t src_chip_id, tt_fabric::RoutingDirection dir) {
        auto [x, y] = _get_chip_coords(src_chip_id);
        switch (dir) {
            case tt_fabric::RoutingDirection::N: return y == 0 ? 0 : y;
            case tt_fabric::RoutingDirection::S: return y == (num_rows - 1) ? 0 : (num_rows - 1 - y);
            case tt_fabric::RoutingDirection::E: return x == (num_cols - 1) ? 0 : (num_cols - 1 - x);
            case tt_fabric::RoutingDirection::W: return x == 0 ? 0 : x;
            default: throw std::runtime_error("Unexpected direction");
        }
        return 0;
    }

    uint32_t get_random_hops_in_direction(chip_id_t src_chip_id, tt_fabric::RoutingDirection dir) {
        uint32_t max_hops = get_max_hops_in_direction(src_chip_id, dir);
        return max_hops == 0 ? 0 : get_random_numbers_from_range(1, max_hops, 1)[0];
    }

    chip_id_t get_unicast_dest_chip(chip_id_t src_chip_id, tt_fabric::RoutingDirection dir, uint32_t num_hops) {
        auto [src_x, src_y] = _get_chip_coords(src_chip_id);
        uint32_t dst_x = src_x;
        uint32_t dst_y = src_y;

        switch (dir) {
            case tt_fabric::RoutingDirection::N: dst_y -= num_hops; break;
            case tt_fabric::RoutingDirection::S: dst_y += num_hops; break;
            case tt_fabric::RoutingDirection::E: dst_x += num_hops; break;
            case tt_fabric::RoutingDirection::W: dst_x -= num_hops; break;
            default: throw std::runtime_error("Unexpected direction");
        }

        return _get_chip_from_coords(dst_x, dst_y);
    }

    std::vector<chip_id_t> get_mcast_dest_chips(
        chip_id_t src_chip_id, tt_fabric::RoutingDirection dir, uint32_t num_hops) {
        std::vector<chip_id_t> dest_chips;
        for (uint32_t hops = 1; hops <= num_hops; hops++) {
            dest_chips.push_back(get_unicast_dest_chip(src_chip_id, dir, hops));
        }

        return dest_chips;
    }

    void close_devices() { tt::tt_metal::detail::CloseDevices(IDevice_handle_map); }

    void _generate_chip_neighbors_map() {
        // find all neighbors of chip 0 (logical chip ids)
        std::unordered_map<tt_fabric::RoutingDirection, chip_id_t> chip_0_neigbors;
        std::optional<tt_fabric::RoutingDirection> chip_1_direction = std::nullopt;
        for (const auto& direction : routing_directions) {
            auto neighbors = control_plane->get_intra_chip_neighbors(mesh_id, 0, direction);
            if (neighbors.empty()) {
                continue;
            }
            // assuming same neighbor per direction
            chip_0_neigbors[direction] = neighbors[0];
            if (neighbors[0] == 1) {
                chip_1_direction = direction;
            }
        }

        if (chip_0_neigbors.size() > 2) {
            log_fatal(
                LogTest, "Expected 2 or less than 2 neigbors for a corner chip, but found {}", chip_0_neigbors.size());
            throw std::runtime_error("Unexpected number of neigbord for corner chip");
        }

        if (!chip_1_direction.has_value()) {
            throw std::runtime_error("Logical chip 1 is not a neighbor of logical chip 0");
        }

        int row_offset = 0, col_offset = 0;

        // determine the row and col offset which will be used while filling up the chip matrix
        switch (chip_1_direction.value()) {
            case tt_fabric::RoutingDirection::N: col_offset = -1; break;
            case tt_fabric::RoutingDirection::S: col_offset = 1; break;
            case tt_fabric::RoutingDirection::E: row_offset = 1; break;
            case tt_fabric::RoutingDirection::W: row_offset = -1; break;
            default: throw std::runtime_error("Unexpected direction");
        }

        if (chip_0_neigbors.size() == 2) {
            // find the other neighbor chip and direction
            tt_fabric::RoutingDirection other_chip_dir;
            chip_id_t other_chip;
            for (const auto& [dir, chip] : chip_0_neigbors) {
                if (chip != 0) {
                    other_chip_dir = dir;
                    other_chip = chip;
                }
            }

            switch (other_chip_dir) {
                case tt_fabric::RoutingDirection::N: col_offset = 0 - other_chip; break;  // chip 0 could be in E/W
                case tt_fabric::RoutingDirection::S: col_offset = other_chip; break;      // chip 0 could be in E/W
                case tt_fabric::RoutingDirection::E: row_offset = other_chip; break;      // chip 0 could be in N/S
                case tt_fabric::RoutingDirection::W: row_offset = 0 - other_chip; break;  // chip 0 could be in N/S
                default: throw std::runtime_error("Unexpected direction");
            }

            if (row_offset == 0 || col_offset == 0) {
                throw std::runtime_error("Unexpected error while setting up neighbor map");
            }
        }

        if (std::abs(row_offset) > 1 || col_offset == 0) {
            num_rows = std::abs(row_offset);
            num_cols = physical_chip_ids.size() / num_rows;
        } else if (std::abs(col_offset) > 1 || row_offset == 0) {
            num_cols = std::abs(col_offset);
            num_rows = physical_chip_ids.size() / num_cols;
        }

        if (num_rows == 0 || num_cols == 0) {
            throw std::runtime_error("Unable to determine number of rows or columns while setting up neighbor map");
        }

        // determine the chip for the NW corner of the matrix
        chip_id_t start_logical_chip_id = (num_rows - 1) * (col_offset < 0 ? std::abs(col_offset) : 0) +
                                          (num_cols - 1) * (row_offset < 0 ? std::abs(row_offset) : 0);

        // populate the physical chip matrix
        physical_chip_matrix.resize(num_rows, std::vector<chip_id_t>(num_cols));
        for (uint32_t i = 0; i < num_rows; i++) {
            chip_id_t logical_chip_id = start_logical_chip_id;
            for (uint32_t j = 0; j < num_cols; j++) {
                if (logical_chip_id > physical_chip_ids.size()) {
                    throw std::runtime_error("Failed to setup neighbor map, logical chip id exceeding bounds");
                }
                chip_id_t phys_chip_id =
                    control_plane->get_physical_chip_id_from_mesh_chip_id({mesh_id, logical_chip_id});
                physical_chip_matrix[i][j] = phys_chip_id;
                logical_chip_id += row_offset;
            }
            start_logical_chip_id += col_offset;
        }

        // cache the chip coords
        for (uint32_t i = 0; i < num_rows; i++) {
            for (uint32_t j = 0; j < num_cols; j++) {
                // flipped indices here since we will later reference them as x,y
                physical_chip_coords[physical_chip_matrix[i][j]] = std::make_pair(j, i);
            }
        }
    }

    std::pair<uint32_t, uint32_t> _get_chip_coords(chip_id_t chip_id) {
        if (physical_chip_coords.find(chip_id) == physical_chip_coords.end()) {
            log_fatal(LogTest, "Unknown chip id: {}", chip_id);
            throw std::runtime_error("Unexpected chip id for coord lookup");
        }
        return physical_chip_coords.at(chip_id);
    }

    chip_id_t _get_chip_from_coords(uint32_t x, uint32_t y) {
        if (x >= num_cols) {
            log_fatal(LogTest, "Chip x coords exceeds bounds, expected less than {}, got {}", num_cols, x);
            throw std::runtime_error("Chip x coords exceeds bounds");
        }
        if (y >= num_rows) {
            log_fatal(LogTest, "Chip y coords exceeds bounds, expected less than {}, got {}", num_rows, y);
            throw std::runtime_error("Chip y coords exceeds bounds");
        }

        return physical_chip_matrix[y][x];
    }

    void _generate_router_handshake_chip_sequence() {
        std::set<chip_id_t> mmio_chip_ids;
        for (const auto& chip_id : physical_chip_ids) {
            mmio_chip_ids.insert(
                tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(chip_id));
        }

        for (const auto& chip_id : mmio_chip_ids) {
            auto tunnels_from_mmio =
                tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(chip_id);
            for (auto i = 0; i < tunnels_from_mmio.size(); i++) {
                for (auto j = tunnels_from_mmio[i].size() - 1; j > 0; j--) {
                    auto remote_chip_id = tunnels_from_mmio[i][j];
                    if (!is_valid_chip_id(remote_chip_id)) {
                        log_fatal(
                            LogTest, "Found unexpected chip id {} while generating handshake seq", remote_chip_id);
                        throw std::runtime_error("Unexpected chip while generating handshake seq");
                    }
                    handshake_seq_chip_ids.push_back(remote_chip_id);
                }
            }
            // insert the mmio chip only if its a valid chip, for instance on TG, skip the gateways
            if (is_valid_chip_id(chip_id)) {
                handshake_seq_chip_ids.push_back(chip_id);
            }
        }
    }

} test_board;

struct TestDevice {
    chip_id_t physical_chip_id;
    tt_metal::IDevice* IDevice_handle;
    tt_metal::Program program_handle;
    metal_SocDescriptor soc_desc;
    std::unordered_map<chan_id_t, CoreCoord> fabric_eth_chan_to_logical_core;
    std::unordered_map<tt_fabric::RoutingDirection, std::vector<chan_id_t>> active_fabric_eth_channels;
    std::unordered_map<tt_fabric::RoutingDirection, chip_id_t> chip_neighbors;
    std::vector<CoreCoord> worker_logical_cores;

    uint32_t edm_status_address;
    uint32_t edm_local_sync_address;
    uint32_t edm_termination_signal_address;
    chan_id_t master_edm_chan;

    // [routing_plane_id][dir] -> (hops, mcast_mode)
    std::unordered_map<uint32_t, std::unordered_map<tt_fabric::RoutingDirection, std::pair<uint32_t, bool>>>
        sender_workers_config;
    std::vector<CoreCoord> sender_worker_virtual_cores;
    // [sender_chip_id] -> {(routing_plane_id)}
    std::unordered_map<chip_id_t, std::set<uint32_t>> receiver_workers_config;
    std::vector<CoreCoord> receiver_worker_virtual_cores;

    // For now, max_num_routing_planes workers are reserved as senders
    // and 2 as misc workers, rest can be used as receivers
    static constexpr uint8_t SENDER_WORKER_RESERVED_INDEX = max_num_routing_planes;
    static constexpr uint8_t MISC_WORKER_RESERVED_INDEX = SENDER_WORKER_RESERVED_INDEX + 2;

    TestDevice(chip_id_t chip_id_) {
        physical_chip_id = chip_id_;
        IDevice_handle = test_board.get_IDevice_handle(physical_chip_id);
        program_handle = tt_metal::CreateProgram();
        soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(physical_chip_id);

        auto fabric_eth_chans =
            tt::tt_metal::MetalContext::instance().get_cluster().get_fabric_ethernet_channels(physical_chip_id);
        for (const auto& chan : fabric_eth_chans) {
            fabric_eth_chan_to_logical_core[chan] = soc_desc.get_eth_core_for_channel(chan, CoordSystem::LOGICAL);
        }

        auto grid_size = IDevice_handle->logical_grid_size();
        for (auto i = 0; i < grid_size.x; i++) {
            for (auto j = 0; j < grid_size.y; j++) {
                worker_logical_cores.push_back(CoreCoord({i, j}));
            }
        }

        // determine the fabric routers and neighbors in each direction
        auto* control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();
        auto mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(physical_chip_id);

        for (const auto& direction : routing_directions) {
            auto neighbors =
                control_plane->get_intra_chip_neighbors(mesh_chip_id.first, mesh_chip_id.second, direction);
            if (neighbors.empty()) {
                continue;
            }

            auto active_eth_chans = control_plane->get_active_fabric_eth_channels_in_direction(
                mesh_chip_id.first, mesh_chip_id.second, direction);
            if (active_eth_chans.empty()) {
                continue;
            }

            std::pair<tt_fabric::mesh_id_t, chip_id_t> neighbor_mesh_chip_id;
            // assume same neighbor per direction
            neighbor_mesh_chip_id = {mesh_chip_id.first, neighbors[0]};
            chip_neighbors[direction] = control_plane->get_physical_chip_id_from_mesh_chip_id(neighbor_mesh_chip_id);

            // !!! TODO: sort the eth chans as per the setup in toplogy.cpp !!!
            std::vector<chan_id_t> ordered_fabric_eth_chans(active_eth_chans.begin(), active_eth_chans.end());

            // only insert the number of eth channels as per active routing planes
            auto num_active_fabric_channels =
                std::min(num_routing_planes, static_cast<uint32_t>(ordered_fabric_eth_chans.size()));
            active_fabric_eth_channels.insert(
                {direction,
                 std::vector<chan_id_t>(
                     ordered_fabric_eth_chans.begin(), ordered_fabric_eth_chans.begin() + num_active_fabric_channels)});
        }

        if (active_fabric_eth_channels.empty()) {
            log_fatal(LogTest, "Could not find any eth channels for fabric kernels for device: {}", physical_chip_id);
            throw std::runtime_error("Could not find any eth channels for fabric kernels");
        }
    }

    // TODO: add support for ring topology
    void compile_fabric_router_kernels() {
        const auto& edm_config = get_edm_config();
        std::unordered_map<chan_id_t, tt_fabric::FabricEriscDatamoverBuilder> edm_builders;

        edm_status_address = edm_config.edm_status_address;
        edm_local_sync_address = edm_config.edm_local_sync_address;
        edm_termination_signal_address = edm_config.termination_signal_address;

        for (const auto& [direction, remote_chip_id] : chip_neighbors) {
            // TODO: add checks for ring support/dateline
            bool is_dateline = false;

            for (const auto& eth_chan : active_fabric_eth_channels[direction]) {
                auto edm_builder = tt_fabric::FabricEriscDatamoverBuilder::build(
                    IDevice_handle,
                    program_handle,
                    fabric_eth_chan_to_logical_core.at(eth_chan),
                    physical_chip_id,
                    remote_chip_id,
                    edm_config,
                    true /* enable persistent mode */,
                    false /* build_in_worker_connection_mode */,
                    is_dateline);
                edm_builders.insert({eth_chan, edm_builder});
            }
        }

        auto connect_downstream_builders = [&](tt_fabric::RoutingDirection dir1, tt_fabric::RoutingDirection dir2) {
            bool can_connect = (chip_neighbors.find(dir1) != chip_neighbors.end()) &&
                               (chip_neighbors.find(dir2) != chip_neighbors.end());
            if (can_connect) {
                auto& eth_chans_dir1 = active_fabric_eth_channels.at(dir1);
                auto& eth_chans_dir2 = active_fabric_eth_channels.at(dir2);

                auto eth_chans_dir1_it = eth_chans_dir1.begin();
                auto eth_chans_dir2_it = eth_chans_dir2.begin();

                // since tunneling cores are not guaraneteed to be reserved on the same routing plane, iterate through
                // the sorted eth channels in both directions
                while (eth_chans_dir1_it != eth_chans_dir1.end() && eth_chans_dir2_it != eth_chans_dir2.end()) {
                    auto eth_chan_dir1 = *eth_chans_dir1_it;
                    auto eth_chan_dir2 = *eth_chans_dir2_it;

                    auto& edm_builder1 = edm_builders.at(eth_chan_dir1);
                    auto& edm_builder2 = edm_builders.at(eth_chan_dir2);
                    edm_builder1.connect_to_downstream_edm(edm_builder2);
                    edm_builder2.connect_to_downstream_edm(edm_builder1);

                    eth_chans_dir1_it++;
                    eth_chans_dir2_it++;
                }
            }
        };

        if (wrap_around_mesh && chip_neighbors.size() == 2) {
            auto it = chip_neighbors.begin();
            auto dir1 = it->first;
            it++;
            auto dir2 = it->first;
            connect_downstream_builders(dir1, dir2);
        } else {
            connect_downstream_builders(tt_fabric::RoutingDirection::N, tt_fabric::RoutingDirection::S);
            connect_downstream_builders(tt_fabric::RoutingDirection::E, tt_fabric::RoutingDirection::W);
        }

        uint32_t num_edm_chans = edm_builders.size();
        master_edm_chan = active_fabric_eth_channels.begin()->second[0];
        uint32_t edm_channels_mask = 0;
        for (const auto& [eth_chan, _] : edm_builders) {
            edm_channels_mask += 0x1 << (uint32_t)eth_chan;
        }

        for (auto& [eth_chan, edm_builder] : edm_builders) {
            edm_builder.set_wait_for_host_signal(true);
            const std::vector<uint32_t> edm_kernel_rt_args = edm_builder.get_runtime_args();
            std::vector<uint32_t> edm_kernel_ct_args = edm_builder.get_compile_time_args();
            uint32_t is_local_handshake_master = eth_chan == master_edm_chan;
            const auto eth_logical_core = fabric_eth_chan_to_logical_core.at(eth_chan);

            edm_kernel_ct_args.push_back(is_local_handshake_master);
            edm_kernel_ct_args.push_back(master_edm_chan);
            edm_kernel_ct_args.push_back(num_edm_chans);
            edm_kernel_ct_args.push_back(edm_channels_mask);

            if (is_local_handshake_master) {
                std::vector<uint32_t> zero_buf(1, 0);
                tt_metal::detail::WriteToDeviceL1(
                    IDevice_handle, eth_logical_core, edm_local_sync_address, zero_buf, CoreType::ETH);
            }

            auto edm_kernel = tt::tt_metal::CreateKernel(
                program_handle,
                edm_kernel_src,
                eth_logical_core,
                tt::tt_metal::EthernetConfig{
                    .noc = tt_metal::NOC::NOC_0,
                    .compile_args = edm_kernel_ct_args,
                    .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

            tt::tt_metal::SetRuntimeArgs(program_handle, edm_kernel, eth_logical_core, edm_kernel_rt_args);
        }
    }

    void wait_for_fabric_router_sync() {
        const auto fabric_master_router_core = fabric_eth_chan_to_logical_core.at(master_edm_chan);
        std::vector<std::uint32_t> master_router_status{0};
        while (master_router_status[0] != tt::tt_fabric::EDMStatus::LOCAL_HANDSHAKE_COMPLETE) {
            tt_metal::detail::ReadFromDeviceL1(
                IDevice_handle, fabric_master_router_core, edm_status_address, 4, master_router_status, CoreType::ETH);
        }
    }

    void notify_fabric_routers() {
        const auto fabric_master_router_core = fabric_eth_chan_to_logical_core.at(master_edm_chan);
        std::vector<uint32_t> signal(1, tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC);
        tt_metal::detail::WriteToDeviceL1(
            IDevice_handle, fabric_master_router_core, edm_status_address, signal, CoreType::ETH);
    }

    void terminate_fabric_routers() {
        std::vector<uint32_t> signal(1, tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE);
        tt_metal::detail::WriteToDeviceL1(
            IDevice_handle,
            fabric_eth_chan_to_logical_core.at(master_edm_chan),
            edm_termination_signal_address,
            signal,
            CoreType::ETH);
    }

    void update_sender_worker_hops(
        uint32_t routing_plane_id,
        std::unordered_map<tt_fabric::RoutingDirection, uint32_t>& hops_counts,
        bool worker_mcast_mode) {
        if (sender_workers_config.find(routing_plane_id) == sender_workers_config.end()) {
            for (const auto& dir : routing_directions) {
                // initlaize with num_hops = 0 and mcast_mode = false
                sender_workers_config[routing_plane_id][dir] = std::make_pair(0, false);
            }
        }

        for (const auto& dir : routing_directions) {
            // reset hop counts if routing plane id exceeds the available in that direction
            if (active_fabric_eth_channels.find(dir) == active_fabric_eth_channels.end() ||
                routing_plane_id >= active_fabric_eth_channels[dir].size()) {
                hops_counts[dir] = 0;
                continue;
            }

            // update the current hop counts for this routing plane id
            // hop counts always increase
            // mcast mode once set, will not be disabled later
            if (hops_counts[dir] > 0) {
                // if hops in this direction are already set, it means we are adding a new receiver
                // enable mcast mode in that case
                if (sender_workers_config[routing_plane_id][dir].first > 0) {
                    sender_workers_config[routing_plane_id][dir].second = true;
                }
                uint32_t new_hops = std::max(hops_counts[dir], sender_workers_config[routing_plane_id][dir].first);
                bool new_mcast_mode = worker_mcast_mode || sender_workers_config[routing_plane_id][dir].second;
                sender_workers_config[routing_plane_id][dir] = std::make_pair(new_hops, new_mcast_mode);
            }
        }
    }

    void add_receiver_worker_for_sender(uint32_t sender_physical_chip_id, uint32_t routing_plane_id) {
        receiver_workers_config[sender_physical_chip_id].insert(routing_plane_id);
    }

    void compile_controller_worker_kernel() {
        auto worker_core = _get_misc_worker_core();
        auto kernel_handle = tt_metal::CreateKernel(
            program_handle,
            controller_kernel_src,
            {worker_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        if (verbose_mode) {
            log_info(
                LogTest, "[Device: Phys: {}] Controller kernel running on core: {}", physical_chip_id, worker_core);
        }

        std::unordered_map<tt_fabric::RoutingDirection, uint32_t> hops_count;
        for (const auto& dir : routing_directions) {
            hops_count[dir] = test_board.get_max_hops_in_direction(physical_chip_id, dir);
        }

        // controller only needs to wait for the host signal for custom fabric setup
        // otherwise the fabric routers are already up and running by the time these kernels start
        uint32_t wait_for_host_signal = metal_fabric_init_level == 0 ? 1 : 0;
        auto runtime_args = _get_controller_worker_rtas(wait_for_host_signal, hops_count);

        // loop in the order of the routing directions since map isnt guaranteed to preserve order
        for (const auto& dir : routing_directions) {
            if (hops_count.find(dir) == hops_count.end() || hops_count.at(dir) == 0) {
                continue;
            }

            if (chip_neighbors.find(dir) == chip_neighbors.end()) {
                log_fatal(LogTest, "Could not find expected neighbor for chip {} in dir {}", physical_chip_id, dir);
                throw std::runtime_error("Could not find neighbor chip");
            }
            auto remote_chip_id = chip_neighbors.at(dir);
            // connect the controller to routing plane (link idx) 0 always
            tt_fabric::append_fabric_connection_rt_args(
                physical_chip_id, remote_chip_id, 0 /* link_idx */, program_handle, {worker_core}, runtime_args);
        }

        std::vector<uint32_t> zero_buf(SEMAPHORE_SIZE_BYTES / 4, 0);
        if (wait_for_host_signal) {
            tt_metal::detail::WriteToDeviceL1(
                IDevice_handle,
                worker_core,
                tensix_unreserved_base_address + HOST_TO_CONTROLLER_SEM_OFFSET,
                zero_buf,
                CoreType::WORKER);
        }
        tt_metal::detail::WriteToDeviceL1(
            IDevice_handle,
            worker_core,
            tensix_unreserved_base_address + CONTROLLER_TO_CONTROLLER_SEM_OFFSET,
            zero_buf,
            CoreType::WORKER);
        tt_metal::detail::WriteToDeviceL1(
            IDevice_handle,
            worker_core,
            tensix_unreserved_base_address + SENDERS_TO_CONTROLLER_SEM_OFFSET,
            zero_buf,
            CoreType::WORKER);

        tt_metal::SetRuntimeArgs(program_handle, kernel_handle, worker_core, runtime_args);
    }

    void notify_controller_worker() {
        auto controller_worker_core = _get_misc_worker_core();
        std::vector<uint32_t> signal(1, 1);
        tt_metal::detail::WriteToDeviceL1(
            IDevice_handle,
            controller_worker_core,
            tensix_unreserved_base_address + HOST_TO_CONTROLLER_SEM_OFFSET,
            signal,
            CoreType::WORKER);
    }

    void compile_sender_worker_kernels() {
        for (const auto& [routing_plane_id, hops_counts] : sender_workers_config) {
            // if hops in no directions are set, skip this sender
            uint32_t skip_sender = std::all_of(
                hops_counts.begin(),
                hops_counts.end(),
                [](const std::pair<tt_fabric::RoutingDirection, std::pair<uint32_t, bool>>& pair) {
                    return pair.second.first == 0;
                });
            if (skip_sender) {
                continue;
            }

            auto worker_core = _get_sender_worker_core(routing_plane_id);
            sender_worker_virtual_cores.push_back(IDevice_handle->worker_core_from_logical_core(worker_core));
            auto kernel_handle = tt_metal::CreateKernel(
                program_handle,
                sender_kernel_src,
                {worker_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

            if (verbose_mode) {
                log_info(
                    LogTest,
                    "[Device: Phys: {}] Sender kernel running on x={},y={}",
                    physical_chip_id,
                    worker_core.x,
                    worker_core.y);
            }

            auto runtime_args = _get_sender_worker_rtas(routing_plane_id, hops_counts);
            // loop in the order of the routing directions since map isnt guaranteed to preserve order
            for (const auto& dir : routing_directions) {
                if (hops_counts.find(dir) == hops_counts.end() || hops_counts.at(dir).first == 0) {
                    continue;
                }

                if (chip_neighbors.find(dir) == chip_neighbors.end()) {
                    log_fatal(LogTest, "Could not find expected neighbor for chip {} in dir {}", physical_chip_id, dir);
                    throw std::runtime_error("Could not find neighbor chip");
                }
                auto remote_chip_id = chip_neighbors.at(dir);
                tt_fabric::append_fabric_connection_rt_args(
                    physical_chip_id, remote_chip_id, routing_plane_id, program_handle, {worker_core}, runtime_args);
            }

            std::vector<uint32_t> zero_buf(SEMAPHORE_SIZE_BYTES / 4, 0);
            tt_metal::detail::WriteToDeviceL1(
                IDevice_handle,
                worker_core,
                tensix_unreserved_base_address + CONTROLLER_TO_WORKERS_SEM_OFFSET,
                zero_buf,
                CoreType::WORKER);

            tt_metal::SetRuntimeArgs(program_handle, kernel_handle, worker_core, runtime_args);
        }
    }

    void compile_receiever_worker_kernels() {
        for (const auto& [sender_physical_chip_id, routing_plane_ids] : receiver_workers_config) {
            auto worker_core = _get_receiver_worker_core(sender_physical_chip_id);
            receiver_worker_virtual_cores.push_back(IDevice_handle->worker_core_from_logical_core(worker_core));
            auto kernel_handle = tt_metal::CreateKernel(
                program_handle,
                receiver_kernel_src,
                {worker_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

            if (verbose_mode) {
                log_info(
                    LogTest,
                    "[Device: Phys: {}] Receiver kernel running on x={},y={}",
                    physical_chip_id,
                    worker_core.x,
                    worker_core.y);
            }

            auto runtime_args = _get_receiver_worker_rtas(sender_physical_chip_id, routing_plane_ids);
            tt_metal::SetRuntimeArgs(program_handle, kernel_handle, worker_core, runtime_args);
        }
    }

    void compile_program() {
        if (metal_fabric_init_level == 0) {
            compile_fabric_router_kernels();
        }
        compile_sender_worker_kernels();
        compile_receiever_worker_kernels();
        // compile controller at the end, since it needs info about number of senders
        compile_controller_worker_kernel();
    }

    void launch_program() { tt_metal::detail::LaunchProgram(IDevice_handle, program_handle, false); }

    void wait_for_sender_workers() {
        // poll the test results register
        // should poll the controller kernel instead?
        for (const auto& core : sender_worker_virtual_cores) {
            while (true) {
                auto status = tt::llrt::read_hex_vec_from_core(
                    physical_chip_id, core, tensix_unreserved_base_address + TEST_RESULTS_ADDRESS_OFFSET, 4);
                if ((status[0] & 0xFFFF) != 0) {
                    break;
                }
            }
        }
    }

    void wait_for_receiver_workers() {
        // poll the test results register
        // should poll the controller kernel instead?
        for (const auto& core : receiver_worker_virtual_cores) {
            while (true) {
                auto status = tt::llrt::read_hex_vec_from_core(
                    physical_chip_id, core, tensix_unreserved_base_address + TEST_RESULTS_ADDRESS_OFFSET, 4);
                if ((status[0] & 0xFFFF) != 0) {
                    break;
                }
            }
        }
    }

    void collect_results() {
        // status should be 'pass'
        // number of packets should match the global settings

        // for rx kernels it should be multiplied by the number of senders (one chip but multiple links)
    }

    void validate_results() {}

    void wait_for_program_done() { tt_metal::detail::WaitProgramDone(IDevice_handle, program_handle); }

    uint32_t _get_noc_encoding(CoreCoord& logical_core) {
        CoreCoord virt_core = IDevice_handle->worker_core_from_logical_core(logical_core);
        return tt_metal::MetalContext::instance().hal().noc_xy_encoding(virt_core.x, virt_core.y);
    }

    uint32_t _get_sender_chip_encoding(chip_id_t chip_id, uint32_t routing_plane_id) {
        return (((uint32_t)chip_id) << 16) | routing_plane_id;
    }

    // returns logcial coordinates
    CoreCoord _get_sender_worker_core(uint32_t routing_plane_id) {
        // first few worker cores are reserved as the senders
        return worker_logical_cores[routing_plane_id];
    }

    // returns logcial coordinates
    CoreCoord _get_misc_worker_core(uint32_t offset = 0) {
        return worker_logical_cores[SENDER_WORKER_RESERVED_INDEX + offset];
    }

    // returns logcial coordinates
    CoreCoord _get_receiver_worker_core(chip_id_t sender_physical_chip_id) {
        return worker_logical_cores[MISC_WORKER_RESERVED_INDEX + sender_physical_chip_id];
    }

    std::vector<uint32_t> _get_controller_worker_rtas(
        uint32_t wait_for_host_signal, std::unordered_map<tt_fabric::RoutingDirection, uint32_t> hops_count) {
        std::vector<uint32_t> controller_rtas = {
            tensix_unreserved_base_address, wait_for_host_signal, sender_worker_virtual_cores.size()};

        // mcast encoding
        auto core_range_virtual_start = IDevice_handle->worker_core_from_logical_core(worker_logical_cores.front());
        auto core_range_virtual_end = IDevice_handle->worker_core_from_logical_core(worker_logical_cores.back());
        uint32_t mcast_encoding = tt::tt_metal::MetalContext::instance().hal().noc_multicast_encoding(
            core_range_virtual_start.x, core_range_virtual_start.y, core_range_virtual_end.x, core_range_virtual_end.y);
        controller_rtas.push_back(worker_logical_cores.size());
        controller_rtas.push_back(mcast_encoding);

        std::vector<uint32_t> hops;
        for (const auto& dir : routing_directions) {
            hops.push_back(hops_count[dir]);
        }
        controller_rtas.insert(controller_rtas.end(), hops.begin(), hops.end());

        return controller_rtas;
    }

    std::vector<uint32_t> _get_sender_worker_rtas(
        uint32_t routing_plane_id,
        std::unordered_map<tt_fabric::RoutingDirection, std::pair<uint32_t, bool>> hops_counts) {
        auto controller_worker_core = _get_misc_worker_core();
        auto receiver_worker_core = _get_receiver_worker_core(physical_chip_id);
        std::vector<uint32_t> sender_rtas = {
            tensix_unreserved_base_address,
            routing_plane_id,
            packet_payload_size_bytes,
            num_packets,
            time_seed,
            _get_sender_chip_encoding(physical_chip_id, routing_plane_id),
            _get_noc_encoding(controller_worker_core),
            _get_noc_encoding(receiver_worker_core)};

        // iterate over the map in the order of the routing directions since order of insertion may not be preserved
        std::vector<uint32_t> is_mcast_enabled;
        std::vector<uint32_t> hops;
        for (const auto& dir : routing_directions) {
            hops.push_back(hops_counts[dir].first);
            is_mcast_enabled.push_back(hops_counts[dir].second);
        }
        sender_rtas.insert(sender_rtas.end(), is_mcast_enabled.begin(), is_mcast_enabled.end());
        sender_rtas.insert(sender_rtas.end(), hops.begin(), hops.end());

        return sender_rtas;
    }

    std::vector<uint32_t> _get_receiver_worker_rtas(
        chip_id_t sender_physical_chip_id, std::set<uint32_t> routing_plane_ids) {
        std::vector<uint32_t> receiver_rtas = {
            tensix_unreserved_base_address, packet_payload_size_bytes, num_packets, time_seed};

        // sender ids
        // TODO: unify max_num_routing_planes and MAX_NUM_SENDERS_PER_RECEIVER
        for (uint32_t plane_id = 0; plane_id < max_num_routing_planes; plane_id++) {
            uint32_t sender_id;
            if (routing_plane_ids.contains(plane_id)) {
                sender_id = _get_sender_chip_encoding(sender_physical_chip_id, plane_id);
            } else {
                sender_id = INVALID_SENDER_ID;
            }
            receiver_rtas.push_back(sender_id);
        }

        return receiver_rtas;
    }
};

/*
struct TestFabricTraffic {
    chan_id_t tx_eth_chan;
    uint32_t num_hops;
    TestDevice* tx_device;
    std::vector<TestDevice*> rx_devices;
    CoreCoord tx_logical_core;
    CoreCoord tx_virtual_core;
    std::vector<CoreCoord> rx_logical_cores;
    std::vector<CoreCoord> rx_virtual_cores;
    std::vector<uint32_t> tx_results;
    std::vector<std::vector<uint32_t>> rx_results;

    TestFabricTraffic(
        chan_id_t tx_eth_chan_, uint32_t num_hops_, TestDevice* tx_device_, std::vector<TestDevice*> rx_devices_) {
        tx_eth_chan = tx_eth_chan_;
        num_hops = num_hops_;
        tx_device = tx_device_;
        rx_devices = rx_devices_;

        // TODO: select the optimal tx/rx worker based on the proximity from the ethernet core
        tx_logical_core = tx_device->select_random_worker_cores(1)[0];
        tx_virtual_core = tx_device->IDevice_handle->worker_core_from_logical_core(tx_logical_core);

        // TODO: for mcast, choose the same rx core
        for (auto& rx_device : rx_devices) {
            CoreCoord rx_logical_core = rx_device->select_random_worker_cores(1)[0];
            rx_logical_cores.push_back(rx_logical_core);
            rx_virtual_cores.push_back(rx_device->IDevice_handle->worker_core_from_logical_core(rx_logical_core));
        }
    }

    void build_worker_kernels() {
        // TODO get these propagated from the command line args
        uint32_t packet_header_address = 0x25000;
        uint32_t source_l1_buffer_address = 0x30000;
        uint32_t packet_payload_size_bytes = 4096;
        uint32_t num_packets = 5;
        uint32_t test_results_address = 0x100000;
        uint32_t test_results_size_bytes = 128;
        uint32_t target_address = 0x30000;
        uint32_t notfication_address = 0x24000;

        std::map<string, string> defines;
        std::vector<uint32_t> zero_buf(1, 0);

        // build sender kernel
        std::vector<uint32_t> compile_args = {test_results_address, test_results_size_bytes, target_address};

        std::vector<uint32_t> tx_runtime_args = {
            packet_header_address,
            source_l1_buffer_address,
            packet_payload_size_bytes,
            num_packets,
            num_hops,
            rx_devices[0]->get_noc_encoding(rx_logical_cores[0]),
            time_seed,
        };

        auto edm_rt_args = tx_device->generate_edm_connection_rt_args(tx_eth_chan, {tx_logical_core});
        for (auto& arg : edm_rt_args) {
            tx_runtime_args.push_back(arg);
        }

        // zero out host notification address
        tt::llrt::write_hex_vec_to_core(tx_device->chip_id, tx_virtual_core, zero_buf, notfication_address);

        auto tx_kernel = tt_metal::CreateKernel(
            tx_device->program_handle,
            tx_kernel_src,
            {tx_logical_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = compile_args,
                .defines = defines});

        tt_metal::SetRuntimeArgs(tx_device->program_handle, tx_kernel, tx_logical_core, tx_runtime_args);

        std::cout << "num hops: " << num_hops << std::endl;

        log_info(
            LogTest,
            "[Device: Phys: {}] TX running on: logical: x={},y={}, virtual: x{},y={}, Eth chan: {}",
            tx_device->chip_id,
            tx_logical_core.x,
            tx_logical_core.y,
            tx_virtual_core.x,
            tx_virtual_core.y,
            (uint32_t)tx_eth_chan);

        // build receiver kernel(s)
        std::vector<uint32_t> rx_runtime_args = {packet_payload_size_bytes, num_packets, time_seed};

        auto rx_kernel = tt_metal::CreateKernel(
            rx_devices[0]->program_handle,
            rx_kernel_src,
            {rx_logical_cores[0]},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = compile_args,
                .defines = defines});

        tt_metal::SetRuntimeArgs(rx_devices[0]->program_handle, rx_kernel, rx_logical_cores[0], rx_runtime_args);

        log_info(
            LogTest,
            "[Device: Phys: {}] RX running on: logical: x={},y={}, virtual: x{},y={}",
            rx_devices[0]->chip_id,
            rx_logical_cores[0].x,
            rx_logical_cores[0].y,
            rx_virtual_cores[0].x,
            rx_virtual_cores[0].y);
    }

    void notify_tx_worker() {
        uint32_t notfication_address = 0x24000;
        std::vector<uint32_t> start_signal(1, 1);
        tt::llrt::write_hex_vec_to_core(tx_device->chip_id, tx_virtual_core, start_signal, notfication_address);
    }

    bool collect_results() {
        uint32_t test_results_address = 0x100000;
        bool pass = true;

        // collect tx results
        // TODO: avoid invoking the device handle directly
        CoreCoord tx_virtual_core = tx_device->IDevice_handle->worker_core_from_logical_core(tx_logical_core);
        tx_results = tt::llrt::read_hex_vec_from_core(tx_device->chip_id, tx_virtual_core, test_results_address, 128);
        log_info(
            LogTest,
            "[Device: Phys: {}] TX status = {}",
            tx_device->chip_id,
            tt_fabric_status_to_string(tx_results[TT_FABRIC_STATUS_INDEX]));
        pass &= (tx_results[TT_FABRIC_STATUS_INDEX] == TT_FABRIC_STATUS_PASS);

        // collect rx results
        // TODO: avoid invoking the device handle directly
        for (auto i = 0; i < rx_devices.size(); i++) {
            CoreCoord rx_virtual_core =
                rx_devices[i]->IDevice_handle->worker_core_from_logical_core(rx_logical_cores[i]);
            rx_results.push_back(
                tt::llrt::read_hex_vec_from_core(rx_devices[i]->chip_id, rx_virtual_core, test_results_address, 128));
            log_info(
                LogTest,
                "[Device: Phys: {}] RX{} status = {}",
                rx_devices[i]->chip_id,
                i,
                tt_fabric_status_to_string(rx_results[i][TT_FABRIC_STATUS_INDEX]));
            pass &= (rx_results[i][TT_FABRIC_STATUS_INDEX] == TT_FABRIC_STATUS_PASS);
        }

        return pass;
    }

    bool validate_results() {
        bool pass = true;
        uint64_t num_tx_bytes =
            ((uint64_t)tx_results[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | tx_results[TT_FABRIC_WORD_CNT_INDEX];
        uint64_t num_rx_bytes;

        // tally-up data words and number of packets from rx and tx
        for (auto i = 0; i < rx_results.size(); i++) {
            num_rx_bytes =
                ((uint64_t)rx_results[i][TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | rx_results[i][TT_FABRIC_WORD_CNT_INDEX];
            pass &= (num_tx_bytes == num_rx_bytes);

            if (!pass) {
                break;
            }
        }

        return pass;
    }

    void print_result_summary() {
        uint64_t num_tx_bytes =
            ((uint64_t)tx_results[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | tx_results[TT_FABRIC_WORD_CNT_INDEX];
        uint64_t tx_elapsed_cycles =
            ((uint64_t)tx_results[TT_FABRIC_CYCLES_INDEX + 1] << 32) | tx_results[TT_FABRIC_CYCLES_INDEX];
        double tx_bw = ((double)num_tx_bytes) / tx_elapsed_cycles;

        log_info(
            LogTest,
            "[Device: Phys: {}] TX bytes sent: {}, elapsed cycles: {} -> BW: {:.2f} B/cycle",
            tx_device->chip_id,
            num_tx_bytes,
            tx_elapsed_cycles,
            tx_bw);
    }
}; */

int main(int argc, char** argv) {
    std::vector<std::string> input_args(argv, argv + argc);
    uint32_t prng_seed = test_args::get_command_option_uint32(input_args, "--prng_seed", default_prng_seed);
    num_packets = test_args::get_command_option_uint32(input_args, "--num_packets", default_num_packets);
    packet_payload_size_kb =
        test_args::get_command_option_uint32(input_args, "--packet_payload_size_kb", default_packet_payload_size_kb);
    num_routing_planes =
        test_args::get_command_option_uint32(input_args, "--num_routing_planes", default_num_routing_planes);
    std::string board_type = test_args::get_command_option(input_args, "--board_type", std::string(default_board_type));

    uint32_t test_device_id_l =
        test_args::get_command_option_uint32(input_args, "--device_id", default_test_device_id_l);
    uint32_t test_device_id_r =
        test_args::get_command_option_uint32(input_args, "--device_id_r", default_test_device_id_r);

    uint32_t num_sender_chips =
        test_args::get_command_option_uint32(input_args, "--num_sender_chips", default_num_sender_chips);

    std::unordered_map<tt_fabric::RoutingDirection, uint32_t> global_hops_counts;
    global_hops_counts[tt_fabric::RoutingDirection::N] =
        test_args::get_command_option_uint32(input_args, "--n_hops", default_num_hops);
    global_hops_counts[tt_fabric::RoutingDirection::S] =
        test_args::get_command_option_uint32(input_args, "--s_hops", default_num_hops);
    global_hops_counts[tt_fabric::RoutingDirection::E] =
        test_args::get_command_option_uint32(input_args, "--e_hops", default_num_hops);
    global_hops_counts[tt_fabric::RoutingDirection::W] =
        test_args::get_command_option_uint32(input_args, "--w_hops", default_num_hops);

    mcast_mode = test_args::has_command_option(input_args, "--mcast_mode");
    bidirectional_mode = test_args::has_command_option(input_args, "--bidirectional_mode");
    benchmark_mode = test_args::has_command_option(input_args, "--benchmark_mode");
    verbose_mode = test_args::has_command_option(input_args, "--verbose");

    metal_fabric_init_level = test_args::get_command_option_uint32(input_args, "--metal_fabric_init_level", 0);

    bool pass = true;

    if (default_prng_seed == prng_seed) {
        std::random_device rd;
        prng_seed = rd();
    }
    global_rng.seed(prng_seed);
    time_seed = std::chrono::system_clock::now().time_since_epoch().count();
    packet_payload_size_bytes = packet_payload_size_kb * 1024;

    // if hops in any direction is specified, reset the ones that are not specified
    bool hops_set = std::any_of(
        global_hops_counts.begin(),
        global_hops_counts.end(),
        [](const std::pair<tt_fabric::RoutingDirection, uint32_t>& pair) { return pair.second != default_num_hops; });
    if (hops_set) {
        for (auto& [dir, hops] : global_hops_counts) {
            hops = hops == default_num_hops ? 0 : hops;
        }
    }

    try {
        test_board.init(board_type);

        if (metal_fabric_init_level == 0) {
            log_info(LogTest, "Running test with custom fabric setup mode");
        } else if (metal_fabric_init_level == 1) {
            log_info(LogTest, "Running test with fabric setup at device init");
        }

        // TODO: query the upper limit on the number of routing planes
        if (num_routing_planes > max_num_routing_planes) {
            log_fatal(
                LogTest,
                "Specified num_routing_planes {}, exceeds the maximum value {}",
                num_routing_planes,
                max_num_routing_planes);
            throw std::runtime_error("Unexpected num_routing_planes");
        }

        // validate left and right device IDs
        if ((default_test_device_id_l != test_device_id_l) && !test_board.is_valid_chip_id(test_device_id_l)) {
            log_fatal(LogTest, "Invalid left chip id: {}", test_device_id_l);
            throw std::runtime_error("Invalid left chip id");
        }
        if ((default_test_device_id_r != test_device_id_r) && !test_board.is_valid_chip_id(test_device_id_r)) {
            log_fatal(LogTest, "Invalid right chip id: {}", test_device_id_r);
            throw std::runtime_error("Invalid right chip id");
        }

        std::vector<chip_id_t> sender_physical_chip_ids;

        //  if both left and right device IDs are specified, launch traffic only b/w them
        if ((default_test_device_id_l != test_device_id_l) && (default_test_device_id_r != test_device_id_r)) {
            // throw warning if num_sender devices is set and is not equal to 1
            if (num_sender_chips != default_num_sender_chips) {
                num_sender_chips = 1;
                log_warning(LogTest, "Overriding num_sender_chips to 1 since both the src and dst chips are specified");
            }

            sender_physical_chip_ids.push_back(test_device_id_l);
            global_hops_counts = test_board.get_hops_between_chips(test_device_id_l, test_device_id_r);
        } else {
            auto available_chips = test_board.get_available_chips();
            auto num_available_chips = available_chips.size();
            if (num_sender_chips == default_num_sender_chips) {
                num_sender_chips = num_available_chips;
            } else if (num_sender_chips > num_available_chips) {
                log_fatal(
                    LogTest,
                    "Expected num_sender_chips to be less than or equal to {}, but got {}",
                    num_available_chips,
                    num_sender_chips);
                throw std::runtime_error("Num sender devices higher than num available devices");
            }

            sender_physical_chip_ids = select_random_elements(available_chips, num_sender_chips);
        }

        std::unordered_map<chip_id_t, std::shared_ptr<TestDevice>> test_devices;
        // init test devices from the list of chips
        for (auto& chip_id : test_board.physical_chip_ids) {
            test_devices[chip_id] = std::make_shared<TestDevice>(chip_id);
        }

        // Create mappings for traffic by adding sender and receiver worker cores
        for (const auto& sender_chip_id : sender_physical_chip_ids) {
            for (uint32_t routing_plane_id = 0; routing_plane_id < num_routing_planes; routing_plane_id++) {
                // if hops are not set, pick random hops
                std::unordered_map<tt_fabric::RoutingDirection, uint32_t> sender_chip_hops_counts;
                for (const auto& [dir, hops] : global_hops_counts) {
                    sender_chip_hops_counts[dir] =
                        hops == default_num_hops ? test_board.get_random_hops_in_direction(sender_chip_id, dir) : hops;
                }

                // the sender operates in the globally set mode mcast mode (either unicast or mcast)
                test_devices[sender_chip_id]->update_sender_worker_hops(
                    routing_plane_id, sender_chip_hops_counts, mcast_mode);

                // sender device updates the hops to 0 in the directions it cant send traffic when it runs out of
                // fabric routers in that direction. If hops in all directions are 0, skip this sender chip.
                bool sender_attached = std::any_of(
                    sender_chip_hops_counts.begin(),
                    sender_chip_hops_counts.end(),
                    [](const std::pair<tt_fabric::RoutingDirection, uint32_t>& pair) { return pair.second != 0; });
                if (!sender_attached) {
                    break;
                }

                // get the receiver chip for each of the hops so that we can add corresponding receiver worker cores
                std::vector<chip_id_t> receiver_chips;
                for (const auto& [dir, hops] : sender_chip_hops_counts) {
                    if (hops == 0) {
                        continue;
                    }

                    if (mcast_mode) {
                        auto mcast_dest_chips = test_board.get_mcast_dest_chips(sender_chip_id, dir, hops);
                        receiver_chips.insert(receiver_chips.end(), mcast_dest_chips.begin(), mcast_dest_chips.end());
                    } else {
                        receiver_chips.push_back(test_board.get_unicast_dest_chip(sender_chip_id, dir, hops));
                    }
                }

                for (const auto& receiver_chip_id : receiver_chips) {
                    test_devices[receiver_chip_id]->add_receiver_worker_for_sender(sender_chip_id, routing_plane_id);

                    // for bidirectional mode, add a sender on the reciever chip in unicast mode
                    // and a corresponding receiever on the sender
                    if (bidirectional_mode) {
                        auto hops_counts = test_board.get_hops_between_chips(receiver_chip_id, sender_chip_id);
                        test_devices[receiver_chip_id]->update_sender_worker_hops(
                            routing_plane_id, hops_counts, false /* mcast_mode */);
                        test_devices[sender_chip_id]->add_receiver_worker_for_sender(
                            receiver_chip_id, routing_plane_id);
                    }
                }
            }
        }

        log_info(LogTest, "Compiling programs");
        for (const auto& [_, test_device] : test_devices) {
            test_device->compile_program();
        }

        log_info(LogTest, "Launching programs");
        for (const auto& [_, test_device] : test_devices) {
            test_device->launch_program();
        }

        if (metal_fabric_init_level == 0) {
            log_info(LogTest, "Programs launched, waiting for fabric routers sync");
            for (const auto& chip_id : test_board.handshake_seq_chip_ids) {
                test_devices[chip_id]->wait_for_fabric_router_sync();
            }

            // do the signalling from host before the routers are kicked off to listen for traffic
            log_info(LogTest, "Router sync done, notifying controllers");
            for (const auto& [_, test_device] : test_devices) {
                test_device->notify_controller_worker();
            }

            log_info(LogTest, "Controllers notified, notifying routers");
            for (const auto& chip_id : test_board.handshake_seq_chip_ids) {
                test_devices[chip_id]->notify_fabric_routers();
            }
        }

        log_info(LogTest, "Routers running, waiting for sender workers to finish");
        for (const auto& [_, test_device] : test_devices) {
            test_device->wait_for_sender_workers();
        }

        log_info(LogTest, "Sender workers done, waiting for receiver workers to finish");
        for (const auto& [_, test_device] : test_devices) {
            test_device->wait_for_receiver_workers();
        }

        if (metal_fabric_init_level == 0) {
            log_info(LogTest, "Receiver workers done, terminating fabric routers");
            for (const auto& [_, test_device] : test_devices) {
                test_device->terminate_fabric_routers();
            }
        }

        log_info(LogTest, "Waiting for programs to finish");
        for (const auto& [_, test_device] : test_devices) {
            test_device->wait_for_program_done();
        }

        log_info(LogTest, "Programs finished, collecting results");
        for (const auto& [_, test_device] : test_devices) {
            test_device->collect_results();
        }

        log_info(LogTest, "Results collected, closing devices");
        test_board.close_devices();

        log_info(LogTest, "Closed devices, validating results");
        for (const auto& [_, test_device] : test_devices) {
            test_device->validate_results();
        }

        // TODO: print result summary

    } catch (const std::exception& e) {
        pass = false;
        log_fatal(e.what());
    }

    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(false);

    if (pass) {
        log_info(LogTest, "Test Passed");
        return 0;
    } else {
        log_fatal(LogTest, "Test Failed\n");
        return 1;
    }

    return 0;
}
