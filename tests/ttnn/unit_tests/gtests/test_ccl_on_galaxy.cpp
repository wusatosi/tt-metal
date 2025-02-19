// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/bfloat16.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_op.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/async_runtime.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn_multi_command_queue_fixture.hpp"

#include "tt_cluster.hpp"

using namespace tt;
using namespace tt_metal;

// We use this to dispatch a single device operation asynchronously
// Needed to reproduce the deadlock scenario with a very specific pattern of commands
// This can go away once device_operation::run will be made async and ccl op is moved to the new tmp-based
// DeviceOperation
namespace async_detail {
template <typename OpConfig>
std::vector<Tensor> run_operation(
    QueueId cq_id,
    OpConfig devop,
    const operation::Tensors& input_tensors,
    const operation::OptionalConstTensors& optional_input_tensors = {},
    const operation::OptionalTensors& optional_output_tensors = {}) {
    static_assert(
        operation::detail::is_device_operation<OpConfig>(), "ttnn::run_operation can only dispatch Device Operations!");
    // Create output tensor vector by examining the number of output shapes created by the device operation
    auto output_specs = operation::DeviceOperation<operation::Tensors>(devop).compute_output_specs(input_tensors, {});
    std::vector<Tensor> outputs(output_specs.size());
    // Populate the workers of the output tensors, based on the input tensors. This is needed for the async engine.
    for (int i = 0; i < outputs.size(); i++) {
        outputs[i] = Tensor(operation::get_workers_for_op_output(input_tensors, optional_input_tensors));
    }
    // Send the operation to the async engine, which will populate the output tensors.
    for (auto worker : outputs.at(0).workers) {
        tt::tt_metal::operation::launch_op(
            [devop, worker, cq_id](
                const std::vector<Tensor>& input_tensors,
                const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                return operation::run(
                    std::move(devop), input_tensors, optional_input_tensors, optional_output_tensors, cq_id);
            },
            input_tensors,
            outputs,
            optional_input_tensors,
            optional_output_tensors);
    }
    return outputs;
}
}  // namespace async_detail

bool is_tg_system() {
    const bool is_galaxy_system = tt::Cluster::instance().is_galaxy_cluster();
    const size_t num_mmio_devices = tt::Cluster::instance().number_of_pci_devices();
    const size_t num_devices = tt::Cluster::instance().number_of_user_devices();
    return is_galaxy_system && (num_mmio_devices == 4) && (num_devices == 32);
}

bool is_tgg_system() {
    const bool is_galaxy_system = tt::Cluster::instance().is_galaxy_cluster();
    const size_t num_mmio_devices = tt::Cluster::instance().number_of_pci_devices();
    const size_t num_devices = tt::Cluster::instance().number_of_user_devices();
    return is_galaxy_system && (num_mmio_devices == 8) && (num_devices == 64);
}

ttnn::MeshShape get_mesh_shape() {
    ttnn::MeshShape shape;
    if (is_tg_system()) {
        shape = {8, 4};
    } else {
        TT_FATAL(is_tgg_system(), "Unsupported Galaxy system");
        shape = {8, 8};
    }
    return shape;
}

void validate_num_tunnels_and_tunnel_depth() {
    const uint32_t num_devices_in_tunnel = tt::Cluster::instance().get_mmio_device_max_tunnel_depth(0);
    const uint32_t num_mmio_devices = tt::Cluster::instance().number_of_pci_devices();
    const uint32_t cluster_tunnel_count = tt::Cluster::instance().get_mmio_device_tunnel_count(0);
    TT_FATAL(
        num_devices_in_tunnel == 4,
        "Expected Galaxy to have tunnel depth of 4, detected tunnel depth of {}",
        num_devices_in_tunnel);
    const uint32_t num_tunnels = num_mmio_devices * cluster_tunnel_count;
    if (is_tg_system()) {
        TT_FATAL(num_tunnels == 8, "Expected 8 tunnels in a TG system, detected {} tunnels", num_tunnels);
    } else if (is_tgg_system()) {
        TT_FATAL(num_tunnels == 16, "Expected 16 tunnels in a TGG system, detected {} tunnels", num_tunnels);
    }
}

std::shared_ptr<bfloat16[]> create_container_for_readback_data(const uint32_t buf_size_datums) {
    if (is_tg_system()) {
        return std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums * 4]);
    } else {
        TT_FATAL(is_tgg_system(), "Unsupported Galaxy system");
        return std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums * 8]);
    }
}
