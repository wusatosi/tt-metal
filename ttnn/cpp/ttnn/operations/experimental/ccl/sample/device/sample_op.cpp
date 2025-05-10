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
    GlobalSemaphore* global_semaphore = new GlobalSemaphore(
        input_tensors[0].device(), CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(1, 1))), 0, BufferType::L1);
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors, global_semaphore);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks sample(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors,
    IDevice* device,
    IDevice* fwd_device,
    bool should_read,
    bool should_write,
    GlobalSemaphore* global_semaphore) {
    tt::tt_metal::Program program{};
    std::optional<tt::tt_metal::operation::OverrideRuntimeArgumentsCallback<std::vector<Tensor>>>
        override_runtime_arguments_callback = std::nullopt;
    std::cout << "DEBUG: sample program created" << std::endl;

    // Common
    tt::DataFormat data_format = tt::DataFormat::Bfp8_b;
    uint32_t src0_cb_index = tt::CB::c_in0;
    uint32_t dst0_cb_index = tt::CB::c_in1;
    uint32_t num_pages_per_packet = 1;
    uint32_t tile_size = 1088;

    const auto reserved_packet_header_CB_index = tt::CB::c_in2;
    static constexpr auto num_packet_headers_storable = 8;
    static constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);

    auto input_tensor = input_tensors[0];
    auto output_tensor = output_tensors[0];

    // GlobalSemaphore(
    //     IDevice* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);

    // // Reader
    auto reader_cores = CoreCoord(0, 0);
    auto writer_cores = CoreCoord(0, 0);

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(2 * tile_size, {{src0_cb_index, data_format}})
            .set_page_size(src0_cb_index, tile_size);
    tt::tt_metal::CBHandle cb_src0_workers = CreateCircularBuffer(program, {reader_cores}, cb_src0_config);

    tt::tt_metal::CircularBufferConfig cb_dst0_config =
        tt::tt_metal::CircularBufferConfig(2 * tile_size, {{dst0_cb_index, data_format}})
            .set_page_size(dst0_cb_index, tile_size);
    tt::tt_metal::CBHandle cb_dst0_workers = CreateCircularBuffer(program, {writer_cores}, cb_dst0_config);

    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    auto reserved_packet_header_CB_handle =
        CreateCircularBuffer(program, {reader_cores}, cb_reserved_packet_header_config);

    // auto semaphore = tt::tt_metal::CreateSemaphore(program, {reader_cores}, 0);
    if (should_read) {
        auto reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
        // reader_kernel_config.compile_args = {src0_cb_index, dst0_cb_index, tile_size};

        auto reader_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/sample/device/kernels/"
            "reader.cpp",
            reader_cores,
            reader_kernel_config);

        std::vector<uint32_t> reader_rt_args = {
            // input_tensor.buffer()->address(),  // tensor_address0
            global_semaphore->address(),
        };

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {reader_cores}, reader_rt_args);
    }

    // Writer
    // auto writer_core_range = CoreRange(writer_cores, writer_cores);

    if (should_write) {
        std::cout << "Writer starting..." << std::endl;
        auto writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
        // writer_kernel_config.compile_args = {src0_cb_index, dst0_cb_index, tile_size};

        auto writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/sample/device/kernels/"
            "writer.cpp",
            writer_cores,
            writer_kernel_config);

        std::vector<uint32_t> writer_rt_args = {
            reserved_packet_header_CB_index,
            global_semaphore->address(),
            reader_cores.x,  // out_ready_sem_noc0_x
            reader_cores.y   // out_ready_sem_noc0_y
        };

        // link is set to 0
        writer_rt_args.push_back(1);
        tt::tt_fabric::append_fabric_connection_rt_args(
            device->id(), fwd_device->id(), 0, program, writer_cores, writer_rt_args);
        writer_rt_args.push_back(0);

        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, writer_cores, writer_rt_args);
    }

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

tt::tt_metal::operation::ProgramWithCallbacks Sample::create_program_at(
    const ttnn::MeshCoordinate& coord,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors,
    GlobalSemaphore* global_semaphore) const {
    const auto& mesh_view = input_tensors[0].mesh_device()->get_view();
    auto devices = mesh_view.get_devices();
    std::cout << "COORD:" << std::endl;
    std::cout << coord[0] << ", " << coord[1] << std::endl;

    if (coord[0] == 0 && coord[1] == 0) {
        auto to_device = devices[1];
        std::cout << "DEBUG: device id: " << devices[0]->id() << std::endl;
        std::cout << "DEBUG: to_device id: " << to_device->id() << std::endl;
        return sample(input_tensors, output_tensors, devices[0], to_device, false, true, global_semaphore);
        // return tt::tt_metal::operation::ProgramWithCallbacks{};
    }
    if (coord[0] == 0 && coord[1] == 1) {
        auto to_device = devices[0];
        std::cout << "DEBUG: device id: " << devices[1]->id() << std::endl;
        std::cout << "DEBUG: to_device id: " << to_device->id() << std::endl;
        return sample(input_tensors, output_tensors, devices[1], to_device, true, false, global_semaphore);
    }
    // return sample(input_tensors, output_tensors, devices[0], devices[0], false, false);
    return {.program = tt::tt_metal::Program(), .override_runtime_arguments_callback = std::nullopt};
    // std::optional<IDevice*> forward_device = std::nullopt;
    // std::optional<IDevice*> backward_device = std::nullopt;
    // uint32_t device_index = 0;  // Initialize device index
    // for (uint32_t i = 0; i < 8; ++i) {
    //     if (devices_to_use.at(i) == target_device) {
    //         device_index = i;
    //         if (i != 0) {
    //             backward_device = devices_to_use.at(i - 1);
    //         } else if (topology == ttnn::ccl::Topology::Ring) {
    //             backward_device = devices_to_use.at(this->ring_size - 1);
    //         }
    //         if (i != this->ring_size - 1) {
    //             forward_device = devices_to_use.at(i + 1);
    //         } else if (topology == ttnn::ccl::Topology::Ring) {
    //             forward_device = devices_to_use.at(0);
    //         }
    //     }
    // }
    // return sample(input_tensors, output_tensors);
};

namespace operations::experimental::ccl {
ttnn::Tensor sample(const ttnn::Tensor& input_tensor) {
    auto result = tt::tt_metal::operation::run(ttnn::Sample{}, {input_tensor});
    // Return the first tensor from the result vector
    if (!result.empty()) {
        return result[0];
    }
    // Return an empty tensor if the result is empty
    return input_tensor;  // Return input as fallback
}
}  // namespace operations::experimental::ccl
}  // namespace ttnn
