// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sliced_reduce_scatter_async_op.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {

// Implementation of SlicedReduceScatterAsync methods

void SlicedReduceScatterAsync::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(
        input_tensors.size() == 1,
        "SlicedReduceScatterAsync: Input tensor size must be 1, but is {}",
        input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensor.get_layout();
    const auto& dtype = input_tensor.get_dtype();
    const auto& page_size = input_tensor.buffer()->page_size();
    TT_FATAL(
        page_size % input_tensor.buffer()->alignment() == 0,
        "SlicedReduceScatterAsync currently requires aligned pages");

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Operands to sliced_reduce_scatter_async must be on device");
    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "Operands to sliced_reduce_scatter_async must be allocated in buffers on device");
    TT_FATAL(this->num_links > 0, "Number of links must be greater than 0, but is {}", this->num_links);
    TT_FATAL(
        this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelized over rows, num_links ({}) exceeds available rows ({})",
        this->num_links,
        input_tensor.device()->compute_with_storage_grid_size().y);

    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Unsupported input memory layout {}.",
        input_tensor.memory_config().memory_layout);
    TT_FATAL(
        input_tensor.memory_config().buffer_type == BufferType::DRAM,
        "SlicedReduceScatterAsync: Input tensor must be in DRAM, but is in {}",
        input_tensor.memory_config().buffer_type);

    TT_FATAL(this->scatter_dim == 3, "SlicedReduceScatterAsync: scatter_dim must be 3, but is {}", this->scatter_dim);

    TT_FATAL(
        input_tensor.get_padded_shape()[this->scatter_dim] % this->ring_size == 0,
        "SlicedReduceScatterAsync: input tensor dimension {} must be divisible by ring_size {}",
        input_tensor.get_padded_shape()[this->scatter_dim],
        this->ring_size);

    // Output tensor validation
    TT_FATAL(
        output_tensors.size() == 2,
        "SlicedReduceScatterAsync: Number of output tensors must be 2, but is {}",
        output_tensors.size());

    for (const auto& maybe_output_tensor : output_tensors) {
        TT_FATAL(maybe_output_tensor.has_value(), "Output tensor must be provided");
        const auto& output_tensor = maybe_output_tensor.value();
        TT_FATAL(
            output_tensor.storage_type() == StorageType::DEVICE,
            "Output tensor for sliced_reduce_scatter_async must be on device");
        TT_FATAL(
            output_tensor.memory_config().buffer_type == BufferType::DRAM,
            "Output tensor for sliced_reduce_scatter_async must be in DRAM, but is in {}",
            output_tensor.memory_config().buffer_type);
        TT_FATAL(output_tensor.get_dtype() == dtype, "Output tensor dtype must match input tensor dtype");
        TT_FATAL(
            output_tensor.memory_config() == this->output_mem_config,
            "Output tensor memory config must match specified output_mem_config");
    }

    const auto& persistent_intermediate_buffer = output_tensors.at(0).value();
    const auto& persistent_output_buffer = output_tensors.at(1).value();

    TT_FATAL(
        persistent_intermediate_buffer.get_padded_shape() == input_tensor.get_padded_shape(),
        "SlicedReduceScatterAsync: persistent_intermediate_buffer and persistent_output_buffer must have the same "
        "shape");
    auto expected_output_shape = input_tensor.get_padded_shape();
    expected_output_shape[this->scatter_dim] /= this->ring_size;
    TT_FATAL(
        persistent_output_buffer.get_padded_shape() == expected_output_shape,
        "SlicedReduceScatterAsync: persistent_output_buffer and persistent_intermediate_buffer must have the same "
        "shape");

    TT_FATAL(this->num_links == 1, "SlicedReduceScatterAsync: num_links must be 1, but is {}", this->num_links);
}

std::vector<ttnn::TensorSpec> SlicedReduceScatterAsync::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto output_shape = input_tensor.get_padded_shape();
    output_shape[this->scatter_dim] *= this->ring_size;
    auto intermediate_tensor_spec = TensorSpec(
        input_tensor.get_padded_shape(),
        TensorLayout(input_tensor.get_dtype(), input_tensor.get_tensor_spec().page_config(), this->output_mem_config));
    auto output_tensor_spec = TensorSpec(
        output_shape,
        TensorLayout(input_tensor.get_dtype(), input_tensor.get_tensor_spec().page_config(), this->output_mem_config));
    return {intermediate_tensor_spec, output_tensor_spec};
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks SlicedReduceScatterAsync::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks SlicedReduceScatterAsync::create_program_at(
    const MeshCoordinate& coord, const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    tt::log_debug(tt::LogOp, "DEBUG: create_program_at is called");
    auto mesh_device = input_tensors[0].mesh_device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(coord) : input_tensors[0].device();

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = this->ring_size;  // Initialize device index

    TT_FATAL(this->topology == ttnn::ccl::Topology::Ring, "DEBUG: topology: {}", this->topology);

    std::vector<IDevice*> devices_to_use = input_tensors[0].mesh_device()->get_view().get_ring_devices();

    for (uint32_t i = 0; i < this->ring_size; ++i) {
        if (devices_to_use.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices_to_use.at(this->ring_size - 1);
            }
            if (i != this->ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices_to_use.at(0);
            }
        }
    }

    TT_FATAL(device_index < this->ring_size, "DEBUG: device_index: {}", device_index);
    TT_FATAL(
        forward_device.value()->id() != backward_device.value()->id(),
        "DEBUG: forward and backward devices are the same: {}, {}",
        forward_device.value()->id(),
        backward_device.value()->id());
    TT_FATAL(
        forward_device.value()->id() != target_device->id(),
        "DEBUG: forward device is the same as target device: {}, {}",
        forward_device.value()->id(),
        target_device->id());
    TT_FATAL(
        backward_device.value()->id() != target_device->id(),
        "DEBUG: backward device is the same as target device: {}, {}",
        backward_device.value()->id(),
        target_device->id());

    return sliced_reduce_scatter_async_minimal(
        input_tensors[0],
        output_tensors.at(0),
        output_tensors.at(1),
        target_device,
        forward_device,
        backward_device,
        this->scatter_dim,
        this->num_links,
        this->ring_size,
        device_index,
        this->topology,
        this->semaphore,
        this->sub_device_id);
}

tt::tt_metal::operation::Hash SlicedReduceScatterAsync::compute_program_hash(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto input_shape = input_tensor.get_padded_shape();
    auto input_memory_layout = input_tensor.get_layout();
    auto input_dtype = input_tensor.get_dtype();
    auto input_memory_config = input_tensor.memory_config();
    std::vector<uint32_t> semaphore_addresses;
    for (const auto& semaphore : this->semaphore) {
        semaphore_addresses.push_back(semaphore.address());
    }

    return tt::tt_metal::operation::hash_operation<SlicedReduceScatterAsync>(
        this->scatter_dim,
        this->num_links,
        this->ring_size,
        this->output_mem_config,
        this->topology,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config,
        semaphore_addresses);
}

namespace operations {
namespace experimental {
namespace ccl {

// Top-level API function for SlicedReduceScatterAsync
Tensor sliced_reduce_scatter_async(
    const Tensor& input_tensor,
    Tensor& persistent_intermediate_buffer,
    Tensor& persistent_output_buffer,
    const int32_t scatter_dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "sliced_reduce_scatter_async op is only supported for Fast Dispatch");

    std::vector<IDevice*> devices;
    for (const auto& spec : input_tensor.device_storage().specs) {
        devices.push_back(input_tensor.mesh_device()->get_device(spec.first));
    }

    uint32_t num_devices = devices.size();
    TT_FATAL(num_devices > 0, "sliced_reduce_scatter_async requires at least one device, but has {}", num_devices);

    ttnn::ccl::Topology ccl_topology = topology;
    if (num_devices == 1) {
        TT_THROW("sliced_reduce_scatter_async is a collective operation and requires more than 1 device.");
    }
    if (num_devices == 2 && topology == ttnn::ccl::Topology::Ring) {
        log_warning(tt::LogOp, "Using Linear topology for SlicedReduceScatterAsync with 2 devices instead of Ring.");
        ccl_topology = ttnn::ccl::Topology::Linear;
    }

    std::vector<std::optional<Tensor>> optional_output_tensors = {
        persistent_intermediate_buffer, persistent_output_buffer};

    // Normalizing dims here before passing to the struct/op implementation
    int32_t rank = input_tensor.get_logical_shape().rank();
    int32_t norm_scatter_dim = (scatter_dim < 0) ? rank + scatter_dim : scatter_dim;

    TT_FATAL(norm_scatter_dim >= 0 && norm_scatter_dim < rank, "Invalid scatter_dim: {}", scatter_dim);

    return tt::tt_metal::operation::run(
               ttnn::SlicedReduceScatterAsync(
                   devices,
                   norm_scatter_dim,
                   num_links,
                   num_devices,
                   memory_config.value_or(input_tensor.memory_config()),
                   ccl_topology,
                   multi_device_global_semaphore,
                   sub_device_id),
               {input_tensor},
               {},
               optional_output_tensors)
        .at(1);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations
}  // namespace ttnn
