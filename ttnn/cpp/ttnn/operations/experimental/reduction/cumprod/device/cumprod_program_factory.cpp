// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cumprod_device_operation.hpp"
#include "tt-metalium/base_types.hpp"
#include "tt-metalium/circular_buffer_types.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/util.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/work_split.hpp>

#include <algorithm>

namespace ttnn::operations::experimental::reduction {

uint32_t CumprodDeviceOperation::SingleCoreCumprodProgramFactory::mul_lower_ranks(
    const Shape& input_shape, const int32_t& dim) {
    uint32_t PLow{1};
    for (int32_t i{dim - 1}; i >= 0; --i) {
        PLow *= input_shape[i];
    }

    return PLow;
}

uint32_t CumprodDeviceOperation::SingleCoreCumprodProgramFactory::mul_higher_ranks(
    const Shape& input_shape, const int32_t& dim) {
    uint32_t PHigh{1};
    for (int32_t i{dim + 1}; i < input_shape.rank() - 2; ++i) {
        PHigh *= input_shape[i];
    }

    return PHigh;
}

uint32_t CumprodDeviceOperation::SingleCoreCumprodProgramFactory::calc_htwt(const Shape& input_shape) {
    switch (input_shape.rank()) {
        case 0: return 0;
        case 1: return input_shape[0] / tt::constants::TILE_WIDTH;
        default:
            return (input_shape[input_shape.rank() - 1] / tt::constants::TILE_WIDTH) *
                   (input_shape[input_shape.rank() - 2] / tt::constants::TILE_HEIGHT);
    }
}

CumprodDeviceOperation::SingleCoreCumprodProgramFactory::cached_program_t
CumprodDeviceOperation::SingleCoreCumprodProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor{tensor_args.input_tensor};
    auto& output_tensor{tensor_return_value};
    const auto& input_shape{input_tensor.get_padded_shape()};
    const uint32_t input_rank{input_shape.rank()};
    const int32_t dim{
        (operation_attributes.dim >= 0) ? operation_attributes.dim : (input_rank + operation_attributes.dim)};

    const uint32_t tiles_per_row{input_shape[dim]};
    const uint32_t num_rows{tensor_args.input_tensor.volume() / 1024 / tiles_per_row};
    const uint32_t PHi{mul_higher_ranks(input_shape, dim)};
    const uint32_t PLo{mul_lower_ranks(input_shape, dim)};
    const uint32_t HtWt{calc_htwt(input_shape)};

    Program program{};

    IDevice* device{input_tensor.device()};
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t total_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    const uint32_t all_work_units{PLo * PHi * HtWt};
    CoreRangeSet core_range_set{};
    uint32_t cores_x{compute_with_storage_grid_size.x};
    uint32_t cores_y{compute_with_storage_grid_size.y};
    if (all_work_units < total_number_of_cores) {
        cores_x = all_work_units % compute_with_storage_grid_size.x;
        cores_y = all_work_units / compute_with_storage_grid_size.x;
        if (cores_y == 0 && cores_x == 0) {
            core_range_set = CoreRangeSet{CoreCoord{0, 0}};
        } else if (cores_y == 0) {
            core_range_set = CoreRangeSet{CoreRange{{0, 0}, {cores_x - 1, 0}}};
        } else {
            core_range_set = CoreRangeSet{CoreRange{{0, 0}, {compute_with_storage_grid_size.x - 1, cores_y - 1}}};
            if (cores_x > 0) {
                core_range_set = core_range_set.merge(CoreRangeSet{CoreRange{{0, cores_y}, {cores_x - 1, cores_y}}});
            }
        }
    } else {
        core_range_set = CoreRangeSet{
            CoreRange{{0, 0}, {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1}}};
    }

    auto src_buffer{input_tensor.buffer()};
    auto dst_buffer{output_tensor.buffer()};

    auto cb_src{create_cb(program, input_tensor.get_dtype(), CumprodCB::SRC, core_range_set, 1)};
    auto cb_acc{create_cb(program, input_tensor.get_dtype(), CumprodCB::ACC, core_range_set, 1)};
    auto cb_one{create_cb(program, input_tensor.get_dtype(), CumprodCB::ONE, core_range_set, 1)};
    auto cb_dst{create_cb(program, input_tensor.get_dtype(), CumprodCB::DST, core_range_set, 1)};

    const uint32_t src_is_dram{src_buffer->buffer_type() == BufferType::DRAM ? 1 : 0};
    const uint32_t dst_is_dram{dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0};

    const auto dst_cb_data_format{datatype_to_dataformat_converter(output_tensor.get_dtype())};
    const bool fp32_dest_acc_en{
        (dst_cb_data_format == DataFormat::Float32) || (dst_cb_data_format == DataFormat::Int32) ||
        (dst_cb_data_format == DataFormat::UInt32)};
    const uint32_t height_tiles{input_shape[2] / constants::TILE_HEIGHT};
    const uint32_t width_tiles{input_shape[3] / constants::TILE_WIDTH};

    const ReaderDataMovementConfig reader_config{};
    const ComputeConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false, .compile_args = {}};
    const WriterDataMovementConfig writer_config{};

    const uint32_t core_utilization_count{std::min(total_number_of_cores, all_work_units ? all_work_units : 1)};
    auto cumprod_reader_kernel_id{create_kernel(
        program,
        KERNEL_PATHS[0],
        core_range_set,
        reader_config,
        {src_buffer->address(),
         num_rows,
         tiles_per_row,
         PHi,
         PLo,
         HtWt,
         core_utilization_count,
         compute_with_storage_grid_size.x})};
    auto cumprod_compute_sc_kernel_id{create_kernel(
        program,
        KERNEL_PATHS[1],
        core_range_set,
        compute_config,
        {num_rows, tiles_per_row, PHi, PLo, HtWt, core_utilization_count, compute_with_storage_grid_size.x})};
    auto cumprod_writer_kernel_id{create_kernel(
        program,
        KERNEL_PATHS[2],
        core_range_set,
        writer_config,
        {dst_buffer->address(),
         num_rows,
         tiles_per_row,
         PHi,
         PLo,
         HtWt,
         core_utilization_count,
         compute_with_storage_grid_size.x})};

    // // TODO(jbbieniekTT): the following algorithm is to be explained.
    // for (uint32_t core_id{0}; core_id < core_utilization_count; ++core_id) {
    //     const CoreCoord core{core_id % compute_with_storage_grid_size.x, core_id / compute_with_storage_grid_size.x};
    //     SetRuntimeArgs(program, cumprod_reader_kernel_id, core, {
    //         src_buffer->address(),
    //         num_rows,
    //         tiles_per_row,
    //         PHi,
    //         PLo,
    //         HtWt,
    //         core_utilization_count,
    //         compute_with_storage_grid_size.x
    //     });

    //     SetRuntimeArgs(program, cumprod_compute_sc_kernel_id, core, {
    //         num_rows,
    //         tiles_per_row,
    //         PHi,
    //         PLo,
    //         HtWt,
    //         core_utilization_count,
    //         compute_with_storage_grid_size.x
    //     });

    //     SetRuntimeArgs(program, cumprod_writer_kernel_id, core, {
    //         dst_buffer->address(),
    //         num_rows,
    //         tiles_per_row,
    //         PHi,
    //         PLo,
    //         HtWt,
    //         core_utilization_count,
    //         compute_with_storage_grid_size.x
    //     });
    // }

    return {
        std::move(program),
        {.cumprod_reader_kernel_id = cumprod_reader_kernel_id,
         .cumprod_compute_kernel_id = cumprod_compute_sc_kernel_id,
         .cumprod_writer_kernel_id = cumprod_writer_kernel_id}};
}

void CumprodDeviceOperation::SingleCoreCumprodProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {}

CBHandle CumprodDeviceOperation::SingleCoreCumprodProgramFactory::create_cb(
    Program& program,
    const DataType& dtype,
    const CumprodCB& cumprod_cb,
    const CoreRangeSet& core_range_set,
    const uint32_t& num_tiles) {
    using tt::tt_metal::detail::TileSize;
    const uint32_t cb_id{static_cast<uint32_t>(cumprod_cb)};
    const auto cb_data_format{datatype_to_dataformat_converter(dtype)};
    const uint32_t single_tile_size{TileSize(cb_data_format)};
    const auto cb_config{CircularBufferConfig{num_tiles * single_tile_size, {{cb_id, cb_data_format}}}.set_page_size(
        cb_id, single_tile_size)};
    return CreateCircularBuffer(program, core_range_set, cb_config);
}

KernelHandle CumprodDeviceOperation::SingleCoreCumprodProgramFactory::create_kernel(
    Program& program,
    const char* kernel_path,
    const CoreRangeSet& core_range_set,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config,
    const std::vector<uint32_t>& runtime_args) {
    auto kernel_id{CreateKernel(program, kernel_path, core_range_set, config)};

    SetRuntimeArgs(program, kernel_id, core_range_set, runtime_args);

    return kernel_id;
}

}  // namespace ttnn::operations::experimental::reduction
