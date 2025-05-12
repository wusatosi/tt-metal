#include "sample_op.hpp"
#include "tt-metalium/core_coord.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/core.hpp"
#include "cpp/ttnn/global_semaphore.hpp"
#include <iostream>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/fabric.hpp>

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {

tt::tt_metal::operation::Hash Sample::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    log_trace(tt::LogOp, "compute_program_hash is called");
    auto input_shape = input_tensors[0].get_padded_shape();
    auto input_memory_layout = input_tensors[0].get_layout();
    auto input_dtype = input_tensors[0].get_dtype();
    auto input_memory_config = input_tensors[0].memory_config();

    return tt::tt_metal::operation::hash_operation<Sample>(
        input_shape, input_memory_layout, input_dtype, input_memory_config);
}

void Sample::validate(const std::vector<Tensor>& input_tensors) const {
    // Basic validation
    TT_FATAL(input_tensors.size() == 1, "Sample operation requires exactly one input tensor");
}

std::vector<ttnn::TensorSpec> Sample::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    // Return the same spec as the input tensor
    std::vector<ttnn::TensorSpec> output_specs;
    output_specs.push_back(input_tensors[0].get_tensor_spec());
    return output_specs;
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks Sample::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors, this->semaphore);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks sample(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors,
    IDevice* device,
    IDevice* fwd_device,
    IDevice* bwd_device,
    const ttnn::GlobalSemaphore& semaphore) {
    tt::tt_metal::Program program{};
    std::optional<tt::tt_metal::operation::OverrideRuntimeArgumentsCallback<std::vector<Tensor>>>
        override_runtime_arguments_callback = std::nullopt;
    std::cout << "DEBUG: sample program created" << std::endl;

    auto input_tensor = input_tensors[0];
    auto output_tensor = output_tensors[0];
    auto mesh_device = input_tensor.mesh_device();
    // Common
    tt::DataFormat data_format = tt::DataFormat::Bfp8_b;
    uint32_t src0_cb_index = tt::CB::c_in0;
    uint32_t dst0_cb_index = tt::CB::c_in1;
    uint32_t header_cb_index = tt::CB::c_in2;
    uint32_t num_pages_per_packet = 1;
    uint32_t tile_size = 1088;

    static constexpr auto num_packet_headers_storable = 8;
    static constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);

    // Reader
    auto reader_cores = CoreCoord(0, 2);
    auto writer_cores = CoreCoord(0, 0);

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(2 * tile_size, {{src0_cb_index, data_format}})
            .set_page_size(src0_cb_index, tile_size);
    tt::tt_metal::CBHandle cb_src0_workers = tt::tt_metal::CreateCircularBuffer(program, writer_cores, cb_src0_config);

    tt::tt_metal::CircularBufferConfig cb_dst0_config =
        tt::tt_metal::CircularBufferConfig(2 * tile_size, {{dst0_cb_index, data_format}})
            .set_page_size(dst0_cb_index, tile_size);
    tt::tt_metal::CBHandle cb_dst0_workers = tt::tt_metal::CreateCircularBuffer(program, writer_cores, cb_dst0_config);

    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2, {{header_cb_index, tt::DataFormat::RawUInt32}})
            .set_page_size(header_cb_index, packet_header_size_bytes);
    tt::tt_metal::CBHandle reserved_packet_header_CB_handle =
        tt::tt_metal::CreateCircularBuffer(program, writer_cores, cb_reserved_packet_header_config);

    auto reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};

    CoreCoord drain_sync_core = device->worker_core_from_logical_core(reader_cores);

    std::cout << "Device: " << device->id() << " Logical core: " << reader_cores.str()
              << " Drain sync core: " << drain_sync_core.str() << std::endl;

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/sample/device/kernels/"
        "reader.cpp",
        reader_cores,
        reader_kernel_config);

    std::vector<uint32_t> reader_rt_args = {
        input_tensor.buffer()->address(),  // tensor_address0
        semaphore.address(),
        device->id()};

    tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {reader_cores}, reader_rt_args);

    // Writer
    auto writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    writer_kernel_config.compile_args = {header_cb_index};

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/sample/device/kernels/"
        "writer.cpp",
        writer_cores,
        writer_kernel_config);

    std::vector<uint32_t> writer_rt_args = {
        header_cb_index,
        input_tensor.buffer()->address(),
        semaphore.address(),
        drain_sync_core.x,  // out_ready_sem_noc0_x
        drain_sync_core.y,  // out_ready_sem_noc0_y
        device->id()};

    // link is set to 0
    writer_rt_args.push_back(1);
    tt::tt_fabric::append_fabric_connection_rt_args(
        device->id(), fwd_device->id(), 0, program, writer_cores, writer_rt_args);
    writer_rt_args.push_back(1);
    tt::tt_fabric::append_fabric_connection_rt_args(
        device->id(), bwd_device->id(), 0, program, writer_cores, writer_rt_args);

    tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, writer_cores, writer_rt_args);

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

tt::tt_metal::operation::ProgramWithCallbacks Sample::create_program_at(
    const ttnn::MeshCoordinate& coord,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors,
    const ttnn::GlobalSemaphore& semaphore) const {
    const auto& mesh_view = input_tensors[0].mesh_device()->get_view();
    auto devices = mesh_view.get_devices();
    // print each device id

    auto current_device = devices[(coord[0] * 4) + coord[1]];

    for (auto device : devices) {
        std::cout << "Device id: " << device->id() << std::endl;
    }
    IDevice* forward_device = nullptr;
    IDevice* backward_device = nullptr;
    for (uint32_t i = 0; i < 8; ++i) {
        if (devices.at(i) != current_device) {
            continue;
        }
        backward_device = i != 0 ? devices.at(i - 1) : devices.at(7);
        forward_device = i != 7 ? devices.at(i + 1) : devices.at(0);
    }

    std::cout << "Coords: " << coord[0] << ", " << coord[1] << std::endl;
    std::cout << "calc idx: " << (coord[0] * 4) + coord[1] << std::endl;
    std::cout << "Current device id: " << current_device->id() << std::endl;
    std::cout << "Fwd device id: " << forward_device->id() << std::endl;
    std::cout << "Bwd device id: " << backward_device->id() << std::endl;
    return sample(input_tensors, output_tensors, current_device, forward_device, backward_device, semaphore);
};

namespace operations::experimental::ccl {
ttnn::Tensor sample(const ttnn::Tensor& input_tensor, const ttnn::GlobalSemaphore& semaphore) {
    auto result = tt::tt_metal::operation::run(ttnn::Sample{semaphore}, {input_tensor});
    // Return the first tensor from the result vector
    if (!result.empty()) {
        return result[0];
    }
    // Return an empty tensor if the result is empty
    return input_tensor;  // Return input as fallback
}
}  // namespace operations::experimental::ccl
}  // namespace ttnn
