#include "sample_op.hpp"
#include <sys/types.h>
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/semaphore.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/core.hpp"
#include "cpp/ttnn/global_semaphore.hpp"
#include <iostream>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/fabric.hpp>

#include "ttnn/tensor/tensor.hpp"
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
    auto input_shape = input_tensors[0].get_padded_shape();
    auto output_shape = ttnn::Shape({input_shape[0], input_shape[1] * 8});
    output_specs.push_back(ttnn::TensorSpec(output_shape, input_tensors[0].tensor_spec().tensor_layout()));
    return output_specs;
}

std::vector<Tensor> Sample::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (!output_tensors.empty() && output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }
    auto spec = compute_output_specs(input_tensors)[0];
    return {create_device_tensor(spec, input_tensors.at(0).device())};
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks Sample::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors, this->semaphores);
        });
}

void createReader(
    tt::tt_metal::Program& program,
    IDevice* device,
    CoreCoord drain_sync_core,
    Tensor& input_tensor,
    Tensor& output_tensor,
    const std::vector<ttnn::GlobalSemaphore>& semaphores,
    CoreCoord reader_core,
    uint32_t device_order,
    const uint32_t local_semaphore_id) {
    auto reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    reader_kernel_config.compile_args = {tt::CB::c_in0, tt::CB::c_in1};

    constexpr static uint32_t tile_size = 1088;

    constexpr static uint32_t num_tiles_per_buffer = 100;

    uint32_t num_tiles = input_tensor.padded_shape().volume() / tt::constants::TILE_HW;
    uint32_t tiles_per_row = input_tensor.padded_shape()[0] / tt::constants::TILE_HEIGHT;
    uint32_t tiles_per_col = input_tensor.padded_shape()[1] / tt::constants::TILE_WIDTH;

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/sample/device/kernels/"
        "reader.cpp",
        reader_core,
        reader_kernel_config);

    std::vector<uint32_t> reader_rt_args = {
        input_tensor.buffer()->address(),   // tensor_address0
        output_tensor.buffer()->address(),  // tensor_address1
        num_tiles,
        tiles_per_row,
        tiles_per_col,
        num_tiles_per_buffer,
        device->id(),
        device_order,
        tile_size,
        local_semaphore_id};

    tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {reader_core}, reader_rt_args);
}

void createWriter(
    tt::tt_metal::Program& program,
    IDevice* device,
    IDevice* fwd_device,
    IDevice* bwd_device,
    CoreCoord drain_sync_core,
    Tensor& input_tensor,
    Tensor& output_tensor,
    const std::vector<ttnn::GlobalSemaphore>& semaphores,
    CoreCoord writer_core,
    int link,
    uint32_t device_order,
    const uint32_t local_semaphore_id) {
    tt::DataFormat data_format = tt::DataFormat::Bfp8_b;
    uint32_t src0_cb_index = tt::CB::c_in0;
    uint32_t src1_cb_index = tt::CB::c_in1;
    uint32_t dst0_cb_index = tt::CB::c_in2;
    uint32_t dst1_cb_index = tt::CB::c_in3;
    uint32_t header_cb_index = tt::CB::c_in4;
    uint32_t num_pages_per_packet = 1;
    uint32_t num_tiles = input_tensor.padded_shape().volume() / tt::constants::TILE_HW;
    uint32_t tiles_per_row = input_tensor.padded_shape()[0] / tt::constants::TILE_HEIGHT;
    uint32_t tiles_per_col = input_tensor.padded_shape()[1] / tt::constants::TILE_WIDTH;

    constexpr static uint32_t tile_size = 1088;

    static constexpr auto num_packet_headers_storable = 8;
    static constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    static constexpr auto num_tiles_per_buffer = 100;

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_tiles_per_buffer * tile_size, {{src0_cb_index, data_format}})
            .set_page_size(src0_cb_index, tile_size);
    tt::tt_metal::CBHandle cb_src0_workers = tt::tt_metal::CreateCircularBuffer(program, writer_core, cb_src0_config);

    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(num_tiles_per_buffer * tile_size, {{src1_cb_index, data_format}})
            .set_page_size(src1_cb_index, tile_size);
    tt::tt_metal::CBHandle cb_src1_workers = tt::tt_metal::CreateCircularBuffer(program, writer_core, cb_src1_config);

    tt::tt_metal::CircularBufferConfig cb_dst0_config =
        tt::tt_metal::CircularBufferConfig(num_tiles_per_buffer * tile_size, {{dst0_cb_index, data_format}})
            .set_page_size(dst0_cb_index, tile_size);
    tt::tt_metal::CBHandle cb_dst0_workers = tt::tt_metal::CreateCircularBuffer(program, writer_core, cb_dst0_config);

    tt::tt_metal::CircularBufferConfig cb_dst1_config =
        tt::tt_metal::CircularBufferConfig(num_tiles_per_buffer * tile_size, {{dst1_cb_index, data_format}})
            .set_page_size(dst1_cb_index, tile_size);
    tt::tt_metal::CBHandle cb_dst1_workers = tt::tt_metal::CreateCircularBuffer(program, writer_core, cb_dst1_config);

    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2, {{header_cb_index, tt::DataFormat::RawUInt32}})
            .set_page_size(header_cb_index, packet_header_size_bytes);
    tt::tt_metal::CBHandle reserved_packet_header_CB_handle =
        tt::tt_metal::CreateCircularBuffer(program, writer_core, cb_reserved_packet_header_config);

    auto writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    writer_kernel_config.compile_args = {src0_cb_index, src1_cb_index, dst0_cb_index, dst1_cb_index, header_cb_index};

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/sample/device/kernels/"
        "writer.cpp",
        writer_core,
        writer_kernel_config);

    auto semaphore_sent = semaphores.at(0);
    auto semaphore_can_receive = semaphores.at(1);

    std::vector<uint32_t> writer_rt_args = {
        input_tensor.buffer()->address(),
        output_tensor.buffer()->address(),
        semaphore_sent.address(),
        semaphore_can_receive.address(),
        drain_sync_core.x,  // out_ready_sem_noc0_x
        drain_sync_core.y,  // out_ready_sem_noc0_y
        device->id(),
        device_order,
        num_tiles,
        tiles_per_row,
        tiles_per_col,
        num_tiles_per_buffer,
        tile_size,
        local_semaphore_id,
    };

    // fabric connection
    writer_rt_args.push_back(1);
    tt::tt_fabric::append_fabric_connection_rt_args(
        device->id(), fwd_device->id(), link, program, writer_core, writer_rt_args);
    writer_rt_args.push_back(1);
    tt::tt_fabric::append_fabric_connection_rt_args(
        device->id(), bwd_device->id(), link, program, writer_core, writer_rt_args);

    tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, writer_core, writer_rt_args);
}

tt::tt_metal::operation::ProgramWithCallbacks sample(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors,
    IDevice* device,
    IDevice* fwd_device,
    IDevice* bwd_device,
    const std::vector<ttnn::GlobalSemaphore>& semaphores,
    uint32_t device_order) {
    tt::tt_metal::Program program{};
    std::optional<tt::tt_metal::operation::OverrideRuntimeArgumentsCallback<std::vector<Tensor>>>
        override_runtime_arguments_callback = std::nullopt;

    auto input_tensor = input_tensors[0];
    auto output_tensor = output_tensors[0];
    auto mesh_device = input_tensor.mesh_device();

    CoreCoord core_coord = CoreCoord(0, 0);
    CoreCoord fwd_drain_sync_core = device->worker_core_from_logical_core(core_coord);

    // Forward
    // Reader
    auto fwd_local_semaphore = tt::tt_metal::CreateSemaphore(program, core_coord, 0);
    // auto bwd_local_semaphore = tt::tt_metal::CreateSemaphore(program, core_coord, 0);
    createReader(
        program,
        device,
        fwd_drain_sync_core,
        input_tensor,
        output_tensor,
        semaphores,
        core_coord,
        device_order,
        fwd_local_semaphore);

    // Writer
    createWriter(
        program,
        device,
        fwd_device,
        bwd_device,
        fwd_drain_sync_core,
        input_tensor,
        output_tensor,
        semaphores,
        core_coord,
        0,
        device_order,
        fwd_local_semaphore);

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

// create map of ids to order
std::map<uint32_t, uint32_t> device_id_order_map = {{4, 0}, {0, 1}, {2, 2}, {6, 3}, {7, 4}, {3, 5}, {1, 6}, {5, 7}};

tt::tt_metal::operation::ProgramWithCallbacks Sample::create_program_at(
    const ttnn::MeshCoordinate& coord,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors,
    const std::vector<ttnn::GlobalSemaphore>& semaphores) const {
    const auto& mesh_view = input_tensors[0].mesh_device()->get_view();
    auto devices = mesh_view.get_devices();

    auto current_device = devices[(coord[0] * 4) + coord[1]];

    IDevice* forward_device = nullptr;
    IDevice* backward_device = nullptr;
    for (uint32_t i = 0; i < 8; ++i) {
        if (devices.at(i) != current_device) {
            continue;
        }
        backward_device = i != 0 ? devices.at(i - 1) : devices.at(7);
        forward_device = i != 7 ? devices.at(i + 1) : devices.at(0);
        break;
    }

    auto device_order = device_id_order_map.at(current_device->id());
    return sample(
        input_tensors, output_tensors, current_device, forward_device, backward_device, semaphores, device_order);
};

namespace operations::experimental::ccl {
ttnn::Tensor sample(const ttnn::Tensor& input_tensor, const std::vector<ttnn::GlobalSemaphore>& semaphores) {
    auto result = tt::tt_metal::operation::run(ttnn::Sample{semaphores}, {input_tensor});
    // Return the first tensor from the result vector
    return result[0];
}
}  // namespace operations::experimental::ccl
}  // namespace ttnn
