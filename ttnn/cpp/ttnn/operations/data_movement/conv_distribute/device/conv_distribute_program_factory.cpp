// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hostdevcommon/kernel_structs.h"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/buffer_constants.hpp"
#include "tt-metalium/circular_buffer_types.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/logger.hpp"
#include "tt-metalium/small_vector.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/types.hpp"
#include <cstdint>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tuple>
#include <vector>
#include "conv_distribute_program_factory.hpp"
#include "ttnn/operations/data_movement/utils/split_knit_utils.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks conv_distribute_multi_core(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::CoreRangeSet& cores,
    int divisor,
    uint32_t num_blocks_per_core,
    uint32_t num_cores_with_extra_block) {
    tt::tt_metal::Program program{};
    auto device = input_tensor.device();
    auto core_grid = cores.bounding_box().grid_size();

    // process input and output tensor formats
    auto input_shard_spec = input_tensor.shard_spec().value();
    uint32_t input_sticks_per_core = input_shard_spec.shape[0];
    uint32_t input_stick_size = input_shard_spec.shape[1];

    auto output_shard_spec = output_tensor.shard_spec().value();
    uint32_t output_sticks_per_core_max = output_shard_spec.shape[0];
    uint32_t output_stick_size = output_shard_spec.shape[1];

    auto all_core_mappings =
        calculate_core_mapping(cores, divisor, input_sticks_per_core, num_blocks_per_core, num_cores_with_extra_block);

    // set up circular buffers
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());

    uint32_t out_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::CircularBufferConfig out_cb_config =
        tt::tt_metal::CircularBufferConfig(
            output_tensor.volume() * output_tensor.element_size(), {{out_cb_index, output_cb_data_format}})
            .set_page_size(out_cb_index, output_stick_size * output_tensor.element_size())
            .set_globally_allocated_address(*output_tensor.buffer());
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, cores, out_cb_config);

    // create kernels
    const std::string kernel_name =
        "ttnn/cpp/ttnn/operations/data_movement/conv_distribute/device/kernels/conv_distribute_reader.cpp";
    tt::tt_metal::KernelHandle kernel_id_0 =
        tt::tt_metal::CreateKernel(program, kernel_name, cores, tt::tt_metal::ReaderDataMovementConfig({out_cb_index}));
    tt::tt_metal::KernelHandle kernel_id_1 =
        tt::tt_metal::CreateKernel(program, kernel_name, cores, tt::tt_metal::WriterDataMovementConfig({out_cb_index}));

    uint32_t src_address = input_tensor.buffer()->address();
    // Process CoreMapping to assign runtime arguments to cores
    for (CoreMapping m : all_core_mappings) {
        auto core = m.first;
        auto data_chunks = m.second;

        // we split each chunk in two so num_reads is the number of chunks
        uint32_t num_reads = data_chunks.size();

        std::vector<uint32_t> runtime_args_kernel_0;
        std::vector<uint32_t> runtime_args_kernel_1;

        // Runtime arg format:
        // src_address
        // num_reads
        // For each chunk: bank_id, read_offset, write_offset, read_size
        runtime_args_kernel_0.push_back(src_address);
        runtime_args_kernel_0.push_back(num_reads);

        runtime_args_kernel_1.push_back(src_address);
        runtime_args_kernel_1.push_back(num_reads);

        uint32_t write_offset = 0;  // Shared cumulative write offset for both kernels

        for (auto chunk : data_chunks) {
            auto input_core = std::get<0>(chunk);
            auto start_index = std::get<1>(chunk);
            auto end_index = std::get<2>(chunk);

            auto bank_id = device->allocator()->get_bank_ids_from_logical_core(BufferType::L1, input_core)[0];

            uint32_t element_size = input_tensor.element_size();
            uint32_t read_offset = start_index * input_stick_size * element_size;
            uint32_t read_size = (end_index - start_index) * input_stick_size * element_size;

            uint32_t chunk_half_size = read_size / 2;

            // Chunkwise kernel arguments
            runtime_args_kernel_0.push_back(bank_id);
            runtime_args_kernel_0.push_back(read_offset);
            runtime_args_kernel_0.push_back(write_offset);
            runtime_args_kernel_0.push_back(chunk_half_size);

            runtime_args_kernel_1.push_back(bank_id);
            runtime_args_kernel_1.push_back(read_offset + chunk_half_size);
            runtime_args_kernel_1.push_back(write_offset + chunk_half_size);
            runtime_args_kernel_1.push_back(read_size - chunk_half_size);

            write_offset += read_size;
        }

        SetRuntimeArgs(program, kernel_id_0, core, runtime_args_kernel_0);
        SetRuntimeArgs(program, kernel_id_1, core, runtime_args_kernel_1);
    }

    auto override_runtime_arguments_callback = [kernel_id_0, kernel_id_1, cb_output, cores](
                                                   const void* operation,
                                                   tt::tt_metal::Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& input = input_tensors.at(0);
        const auto& output = output_tensors.at(0);

        uint32_t input_address = input.buffer()->address();

        auto& runtime_args_0_by_core = GetRuntimeArgs(program, kernel_id_0);
        auto& runtime_args_1_by_core = GetRuntimeArgs(program, kernel_id_1);

        for (auto core : cores.bounding_box()) {
            auto& runtime_args_0 = runtime_args_0_by_core[core.x][core.y];
            auto& runtime_args_1 = runtime_args_1_by_core[core.x][core.y];
            runtime_args_0[0] = input_address;
            runtime_args_1[0] = input_address;
        }

        UpdateDynamicCircularBufferAddress(program, cb_output, *output.buffer());
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = nullptr};
}

}  // namespace ttnn::operations::data_movement::detail
