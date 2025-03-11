// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include "fabric_fixture.hpp"
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/routing_table_generator.hpp>

namespace tt::tt_fabric {
namespace fabric_router_tests {

TEST_F(ControlPlaneFixture, TestUBBConnectivity) {
      const auto& eth_connections = tt::Cluster::instance().get_ethernet_connections();
      for (const auto &[chip, connections] : eth_connections) {
        std::cout <<std::dec << "Chip: " << chip << " connects to " << connections.size() << std::endl;
          for (const auto &[channel, remote_chip_and_channel] : connections) {
              std::cout <<"Chip: " << chip << " Channel: " << channel << " Remote Chip: " << std::get<0>(remote_chip_and_channel) << " Remote Channel: " << std::get<1>(remote_chip_and_channel) << std::endl;
          }
      }

}


}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
