// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include <cmath>

inline uint32_t get_block_size(uint32_t num_tiles, uint32_t max_block_size) {
    uint32_t block_size{1};
    for (uint32_t current_block_size = max_block_size; current_block_size >= 1; current_block_size >>= 1) {
        if (num_tiles % current_block_size == 0) {
            block_size = current_block_size;
            break;
        }
    }
    return block_size;
}

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_shape_dims(const Tensor& x) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return {shape[-4], shape[-3], shape[-2] / tile.get_height(), shape[-1] / tile.get_width()};
}

template <typename F>
void set_or_update_runtime_arguments(
    Program& program,
    KernelHandle reader_kernel_id,
    KernelHandle writer_kernel_id,
    KernelHandle compute_kernel_id,
    CoreCoord compute_with_storage_grid_size,
    const ttnn::operations::normalization::BatchNormOperation::operation_attributes_t& operation_attributes,
    const ttnn::operations::normalization::BatchNormOperation::tensor_args_t& tensor_args,
    ttnn::operations::normalization::BatchNormOperation::tensor_return_value_t& c,
    F handle_args) {
    const auto& a = tensor_args.input;
    const auto& b = tensor_args.running_mean;

    const auto ashape = a.get_padded_shape();
    const auto bshape = b.has_value() ? b->get_padded_shape() : SimpleShape{1, 1};
    const auto cshape = c[0]->get_padded_shape();

    const auto [aN, aC, aHt, aWt] = extract_shape_dims(a);
    const auto [bN, bC, bHt, bWt] = b.has_value() ? extract_shape_dims(*b) : std::tuple{1u, 1u, 1u, 1u};
    const auto [cN, cC, cHt, cWt] = extract_shape_dims(c[0].value());

    uint32_t num_output_tiles = c[0].value().volume() / c[0].value().tensor_spec().tile().get_tile_hw();

    constexpr bool row_major = true;
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);
    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            handle_args(program, reader_kernel_id, core, std::array<uint32_t, 10>{0});
            handle_args(program, writer_kernel_id, core, std::array<uint32_t, 11>{0});
            handle_args(program, compute_kernel_id, core, std::array<uint32_t, 3>{0});
            continue;
        }

        uint32_t cHtWt = cHt * cWt;
        std::array reader_runtime_args = {
            a.buffer()->address(),
            start_tile_id,
            num_tiles_per_core,
            cHtWt,
            aHt * aWt * aC * (aN > 1),
            aHt * aWt * (aC > 1),
            cN,
            cC,
            cHt,
            cWt};
        handle_args(program, reader_kernel_id, core, reader_runtime_args);

        if (b.has_value()) {
            std::array writer_runtime_args = {
                b->buffer()->address(),
                c[0].value().buffer()->address(),
                start_tile_id,
                num_tiles_per_core,
                cHtWt,
                bHt * bWt * bC * (bN > 1),
                bHt * bWt * (bC > 1),
                cN,
                cC,
                cHt,
                cWt};
            handle_args(program, writer_kernel_id, core, writer_runtime_args);

            auto [freq, counter] = calculate_compute_kernel_args(
                ttnn::operations::binary_ng::SubtileBroadcastType::SCALAR_B, start_tile_id, cHtWt, cWt);
            std::array compute_runtime_args = {num_tiles_per_core, freq, counter};
            handle_args(program, compute_kernel_id, core, compute_runtime_args);
        } else {
            class bfloat16 bfloat_scalar(*operation_attributes.scalar);
            uint32_t packed_scalar = pack_two_bfloat16_into_uint32({bfloat_scalar, bfloat_scalar});
            std::array writer_runtime_args = {
                packed_scalar,
                c[0].value().buffer()->address(),
                start_tile_id,
                num_tiles_per_core,
                cHtWt,
                cN,
                cC,
                cHt,
                cWt,
                0u,
                0u};
            handle_args(program, writer_kernel_id, core, writer_runtime_args);

            std::array compute_runtime_args = {num_tiles_per_core, 0u, 0u};
            handle_args(program, compute_kernel_id, core, compute_runtime_args);
        }

        start_tile_id += num_tiles_per_core;
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::normalization {
BatchNormOperation::BatchNormFactory::cached_program_t BatchNormOperation::BatchNormFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    using namespace tt;
    using namespace tt::constants;

    const auto& input = tensor_args.input;
    auto gamma = tensor_args.gamma;
    auto beta = tensor_args.beta;
    auto mean = outputs[1];
    auto rstd = outputs[2];

    auto& output = outputs[0].value();

    auto eps = operation_attributes.eps;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = input.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.get_shape();

    const auto n = input_shape.value[0];
    const auto c = input_shape.value[1];
    const auto h = input_shape.value[2];
    const auto w = input_shape.value[3];

    const auto origin_input_shape = input_shape.value.without_padding();

    const auto origin_h = origin_input_shape[2];
    const auto origin_w = origin_input_shape[3];

    const bool is_lastdim_layernorm = false;
    const bool is_group_norm = true;

    const bool do_mask_h = (origin_h % TILE_HEIGHT) != 0;
    const auto mask_h = do_mask_h ? origin_h % TILE_HEIGHT : TILE_HEIGHT;

    const bool do_mask_w = (origin_w % TILE_WIDTH) != 0;
    const auto mask_w = do_mask_w ? origin_w % TILE_WIDTH : TILE_WIDTH;

    const auto Ht = h / TILE_HEIGHT;
    const auto Wt = w / TILE_WIDTH;

    const auto num_channels = c;
    const auto num_rows = n;
    const auto num_inner_tiles = (num_channels)*Ht * Wt;

    const auto f_c = static_cast<float>(num_channels);
    const auto f_ht = static_cast<float>(origin_h) / static_cast<float>(TILE_HEIGHT);
    const auto f_wt = static_cast<float>(origin_w) / static_cast<float>(TILE_WIDTH);
    auto scaler = 1.0f / (static_cast<float>(TILE_WIDTH) * std::sqrt(f_c * f_ht * f_wt));

    const bool gamma_has_value = gamma.has_value();
    const bool beta_has_value = beta.has_value();
    const bool mean_has_value = mean.has_value();
    const bool rstd_has_value = rstd.has_value();

    constexpr uint32_t MAX_BLOCK_SIZE = 8;
    const uint32_t block_size = get_block_size(num_inner_tiles, MAX_BLOCK_SIZE);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_rows_per_core_group_1,
         num_rows_per_core_group_2] = tt_metal::split_work_to_cores(grid, num_rows);

    log_debug(LogTest, "num_cores_to_be_used: {}", num_cores_to_be_used);
    log_debug(LogTest, "num_rows_per_core_group_1: {}", num_rows_per_core_group_1);
    log_debug(LogTest, "num_rows_per_core_group_2: {}", num_rows_per_core_group_2);
    log_debug(LogTest, "block_size: {}", block_size);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_t = num_inner_tiles;                         // input
    const uint32_t in1_t = 1;                                 // scaler
    const uint32_t in2_t = 1;                                 // epsilon
    const uint32_t in3_t = gamma_has_value ? block_size : 0;  // gamma
    const uint32_t in4_t = beta_has_value ? block_size : 0;   // beta
    const uint32_t in5_t = do_mask_h ? 1 : 0;                 // mask_h
    const uint32_t in6_t = do_mask_w ? 1 : 0;                 // mask_w

    const uint32_t out0_t = block_size;              // output
    const uint32_t out1_t = mean_has_value ? 1 : 0;  // mean
    const uint32_t out2_t = rstd_has_value ? 1 : 0;  // rstd

    const uint32_t im0_t = 1;                                                         // E[x]
    uint32_t im1_t = num_inner_tiles;                                                 // x - E[x]
    uint32_t im2_t = 1;                                                               // (x - E[x])^2
    const uint32_t im3_t = 1;                                                         // Sum[(x - E[x])^2]
    const uint32_t im4_t = 1;                                                         // E[(x - E[x])^2] = Var[x]
    const uint32_t im5_t = 1;                                                         // 1.0/(sqrt(Var[x] + eps))
    const uint32_t im6_t = (gamma_has_value || beta_has_value) ? 2 * block_size : 0;  // x * gamm + beta
    const uint32_t im7_t = 2;                                                         // Sum[x]

    const auto cb_data_format = datatype_to_dataformat_converter(input.get_dtype());
    const auto single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    const auto cb_usage = (in0_t + in1_t + in2_t + in3_t + in4_t + in5_t + in6_t + out0_t + out1_t + out2_t + im0_t +
                           im1_t + im2_t + im3_t + im4_t + im5_t + im6_t + im7_t) *
                          single_tile_size;
    const auto available_L1 = device->l1_size_per_core() - device->get_base_allocator_addr(HalMemType::L1);
    const bool use_large_algorithm = cb_usage >= available_L1;

    if (use_large_algorithm) {
        log_info(LogTest, "Large batch_norm algorithm is selected.");
        in0_t = block_size;
        im1_t = 2 * block_size;
        im2_t = 2 * block_size;
    } else {
        log_info(LogTest, "Small batch_norm algorithm is selected.");
    }

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CBIndex::c_0, in0_t},    // input
            {CBIndex::c_1, in1_t},    // scaler
            {CBIndex::c_2, in2_t},    // eps
            {CBIndex::c_3, in3_t},    // gamma
            {CBIndex::c_4, in4_t},    // beta
            {CBIndex::c_5, in5_t},    // mask_h
            {CBIndex::c_6, in6_t},    // mask_w
            {CBIndex::c_16, out0_t},  // output
            {CBIndex::c_17, out1_t},  // mean
            {CBIndex::c_18, out2_t},  // rstd
            {CBIndex::c_24, im0_t},   // E[x]
            {CBIndex::c_25, im1_t},   // x - E[x]
            {CBIndex::c_26, im2_t},   // (x - E[x])^2
            {CBIndex::c_27, im3_t},   // Sum[(x - E[x])^2]
            {CBIndex::c_28, im4_t},   // E[(x - E[x])^2] = Var[x]
            {CBIndex::c_29, im5_t},   // 1.0/(sqrt(Var[x] + eps))
            {CBIndex::c_30, im6_t},   // y * gamm + beta
            {CBIndex::c_31, im7_t},   // Sum[x]
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file =
        use_large_algorithm
            ? "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/reader_batch_norm_large.cpp"
            : "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/reader_batch_norm_small.cpp";

    const std::string writer_kernel_file(
        "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_batch_norm.cpp");

    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, all_cores);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, all_cores);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines{};
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_SCALAR";

    const auto compute_kernel_file =
        use_large_algorithm
            ? "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_large_kernel.cpp"
            : "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_small_kernel.cpp";

    const std::vector<uint32_t> compute_args_group_1{
        num_rows_per_core_group_1,
        origin_h,
        origin_w,
        num_inner_tiles,
        block_size,
        static_cast<uint32_t>(gamma_has_value),
        static_cast<uint32_t>(beta_has_value),
        static_cast<uint32_t>(mean_has_value),
        static_cast<uint32_t>(rstd_has_value),
        static_cast<uint32_t>(is_lastdim_layernorm),
        static_cast<uint32_t>(is_group_norm)};

    CreateComputeKernel(
        program, compute_kernel_file, {core_group_1, num_rows_per_core_group_1, compute_args_group_1}, compute_defines);

    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{
            num_rows_per_core_group_2,
            origin_h,
            origin_w,
            num_inner_tiles,
            block_size,
            static_cast<uint32_t>(gamma_has_value),
            static_cast<uint32_t>(beta_has_value),
            static_cast<uint32_t>(mean_has_value),
            static_cast<uint32_t>(rstd_has_value),
            static_cast<uint32_t>(is_lastdim_layernorm),
            static_cast<uint32_t>(is_group_norm)};

        CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_rows_per_core_group_2, compute_args_group_2},
            compute_defines);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto input_addr = input.buffer()->address();

    const auto output_addr = output.buffer()->address();
    const auto mean_addr = mean_has_value ? mean.value().buffer()->address() : 0;
    const auto rstd_addr = rstd_has_value ? rstd.value().buffer()->address() : 0;

    const auto gamma_addr = gamma_has_value ? gamma.value().buffer()->address() : 0;
    const auto beta_addr = beta_has_value ? beta.value().buffer()->address() : 0;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_rows_per_core;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // reader
        const std::vector<uint32_t> reader_runtime_args{
            input_addr,
            static_cast<uint32_t>(is_dram(input)),
            gamma_addr,
            static_cast<uint32_t>(is_dram(gamma)),
            static_cast<uint32_t>(gamma_has_value),
            beta_addr,
            static_cast<uint32_t>(is_dram(beta)),
            static_cast<uint32_t>(beta_has_value),
            *reinterpret_cast<uint32_t*>(&scaler),
            *reinterpret_cast<uint32_t*>(&eps),
            tile_offset,
            num_rows_per_core,
            num_inner_tiles,
            num_channels,
            origin_h,
            origin_w,
            block_size,
        };
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        // writer
        const std::vector<uint32_t> writer_runtime_args{
            output_addr,
            static_cast<uint32_t>(is_dram(output)),
            mean_addr,
            static_cast<uint32_t>(mean_has_value ? is_dram(mean.value()) : 1),
            static_cast<uint32_t>(mean_has_value),
            rstd_addr,
            static_cast<uint32_t>(rstd_has_value ? is_dram(rstd.value()) : 1),
            static_cast<uint32_t>(rstd_has_value),
            tile_offset,
            num_rows_per_core,
            num_inner_tiles,
            block_size,
        };
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        tile_offset += num_rows_per_core * num_inner_tiles;
    }

    return {std::move(program), {reader_kernels_id, writer_kernels_id, num_cores_to_be_used, num_cores_y}};
}

void BatchNormOperation::BatchNormFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto input_buffer = tensor_args.input.buffer();
    auto gamma_buffer = tensor_args.gamma.has_value() ? tensor_args.gamma.value().buffer() : nullptr;
    auto beta_buffer = tensor_args.beta.has_value() ? tensor_args.beta.value().buffer() : nullptr;

    auto ouput_buffer = tensor_return_value[0]->buffer();
    auto mean_buffer = tensor_return_value[1]->buffer();
    auto rstd_buffer = tensor_return_value[2]->buffer();

    auto reader_kernels_id = cached_program.shared_variables.reader_kernels_id;
    auto writer_kernels_id = cached_program.shared_variables.writer_kernels_id;
    auto num_cores_to_be_used = cached_program.shared_variables.num_cores_to_be_used;
    auto num_cores_y = cached_program.shared_variables.num_cores_y;

    for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(cached_program.program, reader_kernels_id, core);
            runtime_args[0] = input_buffer->address();
            if (gamma_buffer != nullptr) {
                runtime_args[2] = gamma_buffer->address();
            }
            if (beta_buffer != nullptr) {
                runtime_args[5] = beta_buffer->address();
            }
        }

        {
            auto& runtime_args = GetRuntimeArgs(cached_program.program, writer_kernels_id, core);
            runtime_args[0] = ouput_buffer->address();
            if (mean_buffer != nullptr) {
                runtime_args[2] = mean_buffer->address();
            }
            if (rstd_buffer != nullptr) {
                runtime_args[5] = rstd_buffer->address();
            }
        }
    }
}

// inference mode

BatchNormOperation::BatchNormFactory_Inference::cached_program_t BatchNormOperation::BatchNormFactory_Inference::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, tensor_return_value_t& c) {
    // replicate bcast only
    using namespace tt;
    using namespace tt::constants;

    const auto& input = tensor_args.input;  // a
    // auto gamma = tensor_args.gamma;
    // auto beta = tensor_args.beta;
    const auto& running_mean = tensor_args.running_mean;  // b

    auto& output = c[0].value();

    // auto eps = operation_attributes.eps;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = input.device();
    auto program = CreateProgram();
    auto a_data_format = datatype_to_dataformat_converter(input.get_dtype());
    auto b_data_format =
        running_mean.has_value() ? datatype_to_dataformat_converter(running_mean->get_dtype()) : DataFormat::Float16_b;
    auto c_data_format = datatype_to_dataformat_converter(output.get_dtype());

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////

    uint32_t a_single_tile_size = tt_metal::detail::TileSize(a_data_format);
    uint32_t b_single_tile_size = tt_metal::detail::TileSize(b_data_format);
    uint32_t c_single_tile_size = tt_metal::detail::TileSize(c_data_format);

    uint32_t num_output_tiles = output.volume() / output.tensor_spec().tile().get_tile_hw();

    // we parallelize the computation across the output tiles
    constexpr bool row_major = true;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    Buffer* a_buffer = input.buffer();
    Buffer* b_buffer = nullptr;
    Buffer* c_buffer = output.buffer();

    // How many tiles to store per input CB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;
    auto [a_cb, a_cb_handle] =
        create_cb(tt::CB::c_in0, program, all_device_cores, a_single_tile_size, num_tiles_per_cb, a_data_format);
    auto [c_cb, c_cb_handle] =
        create_cb(tt::CB::c_out0, program, all_device_cores, c_single_tile_size, num_tiles_per_cb, c_data_format);

    // If b is a scalar, we only need one tile in the CB
    uint32_t b_num_tiles_per_cb = b_buffer != nullptr ? num_tiles_per_cb : 1;
    auto [b_cb, b_cb_handle] =
        create_cb(tt::CB::c_in1, program, all_device_cores, b_single_tile_size, b_num_tiles_per_cb, b_data_format);

    auto a_is_dram = static_cast<uint32_t>(a_buffer->buffer_type() == tt_metal::BufferType::DRAM);
    bool b_is_dram = false;
    auto c_is_dram = static_cast<uint32_t>(c_buffer->buffer_type() == tt_metal::BufferType::DRAM);

    auto kernel_config =
        CMAKE_UNIQUE_NAMESPACE::BinaryNgKernelConfig(ttnn::operations::binary_ng::SubtileBroadcastType::NONE);

    const std::string reader_kernel_file =
        "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast.cpp";
    const std::string writer_kernel_file =
        "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar_bcast.cpp";

    auto reader_kernel_id = tt_metal::CreateKernel(
        program, reader_kernel_file, all_device_cores, tt_metal::ReaderDataMovementConfig({a_is_dram}));

    b_buffer = running_mean->buffer();
    b_is_dram = static_cast<uint32_t>(b_buffer->buffer_type() == tt_metal::BufferType::DRAM);

    auto writer_kernel_id = tt_metal::CreateKernel(
        program, writer_kernel_file, all_device_cores, tt_metal::WriterDataMovementConfig({b_is_dram, c_is_dram}));

    // COMPUTE KERNEL
    bool fp32_dest_acc_en = c_data_format == tt::DataFormat::UInt32 || c_data_format == tt::DataFormat::Int32 ||
                            c_data_format == tt::DataFormat::Float32;

    auto kernel_config = ttnn::operations::binary_ng::SubtileBroadcastType::SCALAR_B;

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_inference_calculation.cpp",
        all_device_cores,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en, .defines = {{"BCAST_INPUT", kernel_config.bcast_input_str()}}});

    auto set_runtime_args = [](Program& program, KernelHandle kernel_id, CoreCoord core, auto&& args) {
        tt_metal::SetRuntimeArgs(program, kernel_id, core, args);
    };

    CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
        program,
        reader_kernel_id,
        writer_kernel_id,
        compute_kernel_id,
        compute_with_storage_grid_size,
        operation_attributes,
        tensor_args,
        output,
        set_runtime_args);

    return {
        std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id, compute_with_storage_grid_size}};
}

void BatchNormOperation::BatchNormFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto input_buffer = tensor_args.input.buffer();
    auto gamma_buffer = tensor_args.gamma.has_value() ? tensor_args.gamma.value().buffer() : nullptr;
    auto beta_buffer = tensor_args.beta.has_value() ? tensor_args.beta.value().buffer() : nullptr;

    auto ouput_buffer = tensor_return_value[0]->buffer();
    auto mean_buffer = tensor_return_value[1]->buffer();
    auto rstd_buffer = tensor_return_value[2]->buffer();

    auto reader_kernels_id = cached_program.shared_variables.reader_kernels_id;
    auto writer_kernels_id = cached_program.shared_variables.writer_kernels_id;
    auto num_cores_to_be_used = cached_program.shared_variables.num_cores_to_be_used;
    auto num_cores_y = cached_program.shared_variables.num_cores_y;

    for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(cached_program.program, reader_kernels_id, core);
            runtime_args[0] = input_buffer->address();
            if (gamma_buffer != nullptr) {
                runtime_args[2] = gamma_buffer->address();
            }
            if (beta_buffer != nullptr) {
                runtime_args[5] = beta_buffer->address();
            }
        }

        {
            auto& runtime_args = GetRuntimeArgs(cached_program.program, writer_kernels_id, core);
            runtime_args[0] = ouput_buffer->address();
            if (mean_buffer != nullptr) {
                runtime_args[2] = mean_buffer->address();
            }
            if (rstd_buffer != nullptr) {
                runtime_args[5] = rstd_buffer->address();
            }
        }
    }
}
}  // namespace ttnn::operations::normalization
