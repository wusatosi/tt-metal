#include "mul_add_op.hpp"

#include <sys/types.h>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

namespace ttnn::operations::mul_add {

MulAddDeviceOperation::program_factory_t MulAddDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t&) {
    return MulAddProgramFactoryMultiCore();
}

void MulAddDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {}
void MulAddDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {}

TensorSpec MulAddDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return {tensor_args.input_tensor_a.get_tensor_spec()};
}

Tensor MulAddDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return {create_device_tensor(tensor_args.input_tensor_a.get_tensor_spec(), tensor_args.input_tensor_a.device())};
}

std::tuple<MulAddDeviceOperation::operation_attributes_t, MulAddDeviceOperation::tensor_args_t>
MulAddDeviceOperation::invoke(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, const Tensor& input_tensor_c) {
    return {operation_attributes_t{true}, tensor_args_t{input_tensor_a, input_tensor_b, input_tensor_c}};
}

MulAddDeviceOperation::MulAddProgramFactoryMultiCore::cached_program_t
MulAddDeviceOperation::MulAddProgramFactoryMultiCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::Program program{};

    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& input_tensor_c = tensor_args.input_tensor_c;
    auto& output = tensor_return_value;
    auto src_buffer_a = input_tensor_a.buffer();
    auto src_buffer_b = input_tensor_b.buffer();
    auto src_buffer_c = input_tensor_c.buffer();
    auto dst_buffer = output.buffer();

    // Prepare compute kernels
    tt::tt_metal::Device* device = input_tensor_a.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/mul_add/device/kernels/compute/muladd_compute.cpp",
        all_device_cores,
        tt::tt_metal::ComputeConfig{.fp32_dest_acc_en = false});

    // Prepare circular buffers for input/intermediate/output
    tt::DataFormat src0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_a.get_dtype());
    uint32_t src0_single_tile_size = tt::tt_metal::detail::TileSize(src0_cb_data_format);
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, src0_single_tile_size);

    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src0_config);

    tt::DataFormat src1_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_b.get_dtype());
    uint32_t src1_single_tile_size = tt::tt_metal::detail::TileSize(src1_cb_data_format);
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_tiles * src1_single_tile_size, {{src1_cb_index, src1_cb_data_format}})
            .set_page_size(src1_cb_index, src1_single_tile_size);

    auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src1_config);

    tt::DataFormat src2_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_c.get_dtype());
    uint32_t src2_single_tile_size = tt::tt_metal::detail::TileSize(src2_cb_data_format);
    uint32_t src2_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_src2_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_tiles * src2_single_tile_size, {{src2_cb_index, src2_cb_data_format}})
            .set_page_size(src2_cb_index, src2_single_tile_size);

    auto cb_src2 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src2_config);

    uint32_t output_cb_index = tt::CBIndex::c_3;
    uint32_t num_output_tiles = 2;
    tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt::tt_metal::detail::TileSize(dst_cb_data_format);

    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, dst_single_tile_size);

    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_output_config);

    uint32_t intermediate_cb_index = tt::CBIndex::c_4;
    tt::DataFormat intermediate_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t intermediate_single_tile_size = tt::tt_metal::detail::TileSize(dst_cb_data_format);

    tt::tt_metal::CircularBufferConfig cb_intermediate_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * intermediate_single_tile_size, {{intermediate_cb_index, intermediate_cb_data_format}})
            .set_page_size(intermediate_cb_index, intermediate_single_tile_size);

    auto cb_intermediate = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_intermediate_config);

    // Prepare reader kernels
    KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/mul_add/device/kernels/dataflow/reader.cpp",
        all_device_cores,
        tt::tt_metal::ReaderDataMovementConfig());

    KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/mul_add/device/kernels/dataflow/writer.cpp",
        all_device_cores,
        tt::tt_metal::WriterDataMovementConfig());

    // Prepare runtime arguments
    std::vector<std::vector<uint32_t>> reader_args;
    std::vector<std::vector<uint32_t>> compute_args;
    std::vector<std::vector<uint32_t>> writer_args;

    uint32_t num_cores_total = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;
    std::vector<CoreCoord> cores = grid_to_cores(
        num_cores_total,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y,
        /*row_major*/ true);

    reader_args = {cores.size(), std::vector<uint32_t>(4)};
    compute_args = {cores.size(), std::vector<uint32_t>(2)};
    writer_args = {cores.size(), std::vector<uint32_t>(3)};

    uint32_t num_tiles = input_tensor_a.volume() / tt::constants::TILE_HW;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_cores, num_tiles_per_core_group_1, num_tiles_per_core_group_2;
    std::tie(num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2) =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles, /*row_major*/ true);

    uint32_t block_size_per_core_group_1 = 1, block_size_per_core_group_2 = 1, max_block_size = 1;
    uint32_t block_cnt_per_core_group_1 = num_tiles_per_core_group_1;
    uint32_t block_cnt_per_core_group_2 = num_tiles_per_core_group_2;

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; ++i) {
        uint32_t start_id = num_tiles_read;
        uint32_t num_tiles_per_core;
        uint32_t block_cnt_per_core;
        uint32_t block_size_per_core;

        if (i < g1_numcores) {
            num_tiles_per_core = num_tiles_per_core_group_1;
            block_cnt_per_core = block_cnt_per_core_group_1;
            block_size_per_core = block_size_per_core_group_1;
        } else {
            num_tiles_per_core = num_tiles_per_core_group_2;
            block_cnt_per_core = block_cnt_per_core_group_2;
            block_size_per_core = block_size_per_core_group_2;
        }

        reader_args[i] = {
            src_buffer_a->address(), src_buffer_b->address(), src_buffer_c->address(), num_tiles_per_core, start_id};
        compute_args[i] = {block_cnt_per_core, block_size_per_core};
        writer_args[i] = {dst_buffer->address(), num_tiles_per_core, num_tiles_read};

        num_tiles_read += num_tiles_per_core;
    }

    SetRuntimeArgs(program, reader_kernel_id, cores, reader_args);
    SetRuntimeArgs(program, compute_kernel_id, cores, compute_args);
    SetRuntimeArgs(program, writer_kernel_id, cores, writer_args);

    return {std::move(program), {.reader_kernel_id = reader_kernel_id, .writer_kernel_id = writer_kernel_id}};
}

void MulAddDeviceOperation::MulAddProgramFactoryMultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;

    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& input_tensor_c = tensor_args.input_tensor_c;
    auto& output = tensor_return_value;
    auto src_buffer_a = input_tensor_a.buffer();
    auto src_buffer_b = input_tensor_b.buffer();
    auto src_buffer_c = input_tensor_c.buffer();
    auto dst_buffer = output.buffer();

    auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, CoreCoord{0, 0});
    reader_args[0] = src_buffer_a->address();
    reader_args[1] = src_buffer_b->address();
    reader_args[2] = src_buffer_c->address();

    auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, CoreCoord{0, 0});
    writer_args[0] = dst_buffer->address();
}

}  // namespace ttnn::operations::mul_add
