#include "muladd_program_factory.hpp"
#include <sys/types.h>
#include <cstddef>
#include <cstdint>
#include "tt-metalium/base_types.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/buffer_constants.hpp"
#include "tt-metalium/core_coord.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <cstdio>
#include <memory>
#include <tt-metalium/work_split.hpp>
#include <vector>

namespace ttnn::operations::muladd {

using namespace tt::constants;
using namespace tt::tt_metal;

operation::ProgramWithCallbacks single_core_muladd(
    const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, Tensor& output, MathFidelity math_fidelity) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    uint32_t total_elements = 1;
    const auto shape = a.get_logical_shape();
    for (int i = 0; i < shape.size(); i++) {
        total_elements *= shape[i];
    }
    const uint32_t num_output_tiles = total_elements / 1024;

    const tt::DataFormat input_format = tt::DataFormat::Float16_b;
    const tt::DataFormat output_format = input_format;
    const uint32_t input_tile_size = tt::tt_metal::detail::TileSize(input_format);
    const uint32_t output_tile_size = input_tile_size;

    const auto create_circular_buffer = [&program](
                                            uint32_t index,
                                            uint32_t num_tiles,
                                            uint32_t tile_size,
                                            const tt::DataFormat& format) -> tt::tt_metal::CBHandle {
        const tt::tt_metal::CircularBufferConfig config =
            tt::tt_metal::CircularBufferConfig(num_tiles * tile_size, {{index, format}})
                .set_page_size(index, tile_size);
        return tt::tt_metal::CreateCircularBuffer(program, CoreCoord{0, 0}, config);
    };

    const auto input_cb_0 = create_circular_buffer(tt::CBIndex::c_0, 1, input_tile_size, input_format);
    const auto input_cb_1 = create_circular_buffer(tt::CBIndex::c_1, 1, input_tile_size, input_format);
    const auto input_cb_2 = create_circular_buffer(tt::CBIndex::c_2, 1, input_tile_size, input_format);
    const auto input_cb_3 = create_circular_buffer(tt::CBIndex::c_3, 1, input_tile_size, input_format);

    const auto output_cb_0 = create_circular_buffer(tt::CBIndex::c_13, 1, output_tile_size, output_format);
    const auto output_cb_1 = create_circular_buffer(tt::CBIndex::c_14, 1, output_tile_size, output_format);
    const auto output_cb_2 = create_circular_buffer(tt::CBIndex::c_15, 1, output_tile_size, output_format);
    const auto output_cb_3 = create_circular_buffer(tt::CBIndex::c_16, 1, output_tile_size, output_format);

    std::vector<uint32_t> reader_compile_time_args = {};
    std::vector<uint32_t> writer_compile_time_args = {tt::CBIndex::c_16};
    std::vector<uint32_t> compute_compile_time_args = {};

    const CoreCoord core = {0, 0};

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/muladd/device/kernels/dataflow/reader_muladd.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/muladd/device/kernels/dataflow/writer_muladd.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/muladd/device/kernels/compute/muladd.cpp",
        core,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    auto set_runtime_args =
        [reader_kernel_id, writer_kernel_id, compute_kernel_id, core, num_output_tiles](
            Program& program, const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, Tensor& output) {
            const auto* input_buffer_a = a.buffer();
            const auto* input_buffer_b = b.buffer();
            const auto* input_buffer_c = c.buffer();
            const auto* input_buffer_d = d.buffer();

            const auto* output_buffer = output.buffer();

            std::vector<uint32_t> reader_runtime_args = {
                input_buffer_a->address(),
                input_buffer_b->address(),
                input_buffer_c->address(),
                input_buffer_d->address(),
                num_output_tiles,
                0};

            std::vector<uint32_t> writer_runtime_args = {output_buffer->address(), num_output_tiles, 0};

            std::vector<uint32_t> compute_runtime_args = {num_output_tiles};

            SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
            SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);
        };
    set_runtime_args(program, a, b, c, d, output);

    return {.program = std::move(program)};
}

operation::ProgramWithCallbacks multi_core_muladd(
    const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, Tensor& output, MathFidelity math_fidelity) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    IDevice* device = output.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t total_elements = 1;
    const auto shape = a.get_logical_shape();
    for (int i = 0; i < shape.size(); i++) {
        total_elements *= shape[i];
    }
    const uint32_t num_output_tiles = total_elements / 1024;

    auto input_buffer_a_address = a.buffer()->address();
    auto input_buffer_b_address = b.buffer()->address();
    auto input_buffer_c_address = c.buffer()->address();
    auto input_buffer_d_address = d.buffer()->address();
    const auto& output_buffer_address = output.buffer()->address();

    const bool& ina_sharded = is_sharded(a.buffer()->buffer_layout());
    const bool& inb_sharded = is_sharded(b.buffer()->buffer_layout());
    const bool& inc_sharded = is_sharded(c.buffer()->buffer_layout());
    const bool& ind_sharded = is_sharded(d.buffer()->buffer_layout());
    const bool& out_sharded = is_sharded(output.buffer()->buffer_layout());

    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_tiles_per_core_group_1,
         num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_output_tiles);

    uint32_t shard_tiles_w = 0, shard_tiles_h = 0;
    uint32_t stride = 0;

    CoreRangeSet cores;
    TensorMemoryLayout memory_layout;
    if (ina_sharded) {
        shard_tiles_w = a.buffer()->shard_spec().shape()[1] / 32;
        shard_tiles_h = a.buffer()->shard_spec().shape()[0] / 32;
        num_cores = a.buffer()->shard_spec().grid().num_cores();
        num_cores_x = a.buffer()->shard_spec().grid().bounding_box().grid_size().x;
        cores = a.buffer()->shard_spec().grid();
        memory_layout = a.memory_config().memory_layout;
    }
    if (inb_sharded) {
        shard_tiles_w = b.buffer()->shard_spec().shape()[1] / 32;
        shard_tiles_h = b.buffer()->shard_spec().shape()[0] / 32;
        num_cores = b.buffer()->shard_spec().grid().num_cores();
        num_cores_x = b.buffer()->shard_spec().grid().bounding_box().grid_size().x;
        num_cores_y = b.buffer()->shard_spec().grid().bounding_box().grid_size().y;
        cores = b.buffer()->shard_spec().grid();
        memory_layout = b.memory_config().memory_layout;
    }
    if (inc_sharded) {
        shard_tiles_w = c.buffer()->shard_spec().shape()[1] / 32;
        shard_tiles_h = c.buffer()->shard_spec().shape()[0] / 32;
        num_cores = c.buffer()->shard_spec().grid().num_cores();
        num_cores_x = c.buffer()->shard_spec().grid().bounding_box().grid_size().x;
        num_cores_y = c.buffer()->shard_spec().grid().bounding_box().grid_size().y;
        cores = c.buffer()->shard_spec().grid();
        memory_layout = c.memory_config().memory_layout;
    }
    if (ind_sharded) {
        shard_tiles_w = d.buffer()->shard_spec().shape()[1] / 32;
        shard_tiles_h = d.buffer()->shard_spec().shape()[0] / 32;
        num_cores = d.buffer()->shard_spec().grid().num_cores();
        num_cores_x = d.buffer()->shard_spec().grid().bounding_box().grid_size().x;
        num_cores_y = d.buffer()->shard_spec().grid().bounding_box().grid_size().y;
        cores = d.buffer()->shard_spec().grid();
        memory_layout = d.memory_config().memory_layout;
    }
    if (out_sharded) {
        shard_tiles_w = output.buffer()->shard_spec().shape()[1] / 32;
        shard_tiles_h = output.buffer()->shard_spec().shape()[0] / 32;
        num_cores = output.buffer()->shard_spec().grid().num_cores();
        num_cores_x = output.buffer()->shard_spec().grid().bounding_box().grid_size().x;
        num_cores_y = output.buffer()->shard_spec().grid().bounding_box().grid_size().y;
        cores = output.buffer()->shard_spec().grid();
        memory_layout = output.memory_config().memory_layout;
    }

    const tt::DataFormat input_format = tt::DataFormat::Float16_b;
    const tt::DataFormat output_format = input_format;
    const uint32_t input_tile_size = tt::tt_metal::detail::TileSize(input_format);
    const uint32_t output_tile_size = input_tile_size;

    const auto create_circular_buffer = [&program, all_cores, shard_tiles_w](
                                            uint32_t index,
                                            uint32_t num_tiles,
                                            uint32_t tile_size,
                                            const tt::DataFormat& format,
                                            bool use_global_address = false,
                                            Buffer* buf = nullptr) -> tt::tt_metal::CBHandle {
        tt::tt_metal::CircularBufferConfig config =
            tt::tt_metal::CircularBufferConfig(num_tiles * tile_size, {{index, format}})
                .set_page_size(index, tile_size);
        if (use_global_address) {
            config.set_globally_allocated_address(*buf);
        }
        return tt::tt_metal::CreateCircularBuffer(program, all_cores, config);
    };

    auto ina_cb_size = ina_sharded ? shard_tiles_w * shard_tiles_h : 1;
    auto inb_cb_size = inb_sharded ? shard_tiles_w * shard_tiles_h : 1;
    auto inc_cb_size = inc_sharded ? shard_tiles_w * shard_tiles_h : 1;
    auto ind_cb_size = ind_sharded ? shard_tiles_w * shard_tiles_h : 1;
    auto out_cb_size = out_sharded ? shard_tiles_w * shard_tiles_h : 1;

    const auto input_cb_0 =
        create_circular_buffer(tt::CBIndex::c_0, ina_cb_size, input_tile_size, input_format, ina_sharded, a.buffer());
    const auto input_cb_1 =
        create_circular_buffer(tt::CBIndex::c_1, inb_cb_size, input_tile_size, input_format, inb_sharded, b.buffer());
    const auto input_cb_2 =
        create_circular_buffer(tt::CBIndex::c_2, inc_cb_size, input_tile_size, input_format, inc_sharded, c.buffer());
    const auto input_cb_3 =
        create_circular_buffer(tt::CBIndex::c_3, ind_cb_size, input_tile_size, input_format, ind_sharded, d.buffer());

    const auto intermed_cb_0 = create_circular_buffer(tt::CBIndex::c_13, 1, output_tile_size, output_format);
    const auto intermed_cb_1 = create_circular_buffer(tt::CBIndex::c_14, 1, output_tile_size, output_format);
    const auto intermed_cb_2 = create_circular_buffer(tt::CBIndex::c_15, 1, output_tile_size, output_format);
    const auto output_cb = create_circular_buffer(
        tt::CBIndex::c_16, out_cb_size, output_tile_size, output_format, out_sharded, output.buffer());

    std::vector<uint32_t> reader_compile_time_args = {
        a.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0,
        b.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0,
        c.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0,
        d.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0,
        memory_layout == TensorMemoryLayout::BLOCK_SHARDED ? (num_cores_x - 1) * shard_tiles_w : 0,
        shard_tiles_w != 0 ? shard_tiles_w : 1,
    };
    std::vector<uint32_t> writer_compile_time_args = {
        tt::CBIndex::c_16,
        output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0,
        memory_layout == TensorMemoryLayout::BLOCK_SHARDED ? (num_cores_x - 1) * shard_tiles_w : 0,
        shard_tiles_w != 0 ? shard_tiles_w : 1,
    };
    std::vector<uint32_t> compute_compile_time_args = {};

    std::map<string, string> defines;
    if (ina_sharded) {
        defines["IN0_SHARDED"] = "1";
    }
    if (inb_sharded) {
        defines["IN1_SHARDED"] = "1";
    }
    if (inc_sharded) {
        defines["IN2_SHARDED"] = "1";
    }
    if (ind_sharded) {
        defines["IN3_SHARDED"] = "1";
    }
    if (out_sharded) {
        defines["OUT_SHARDED"] = "1";
    }
    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/muladd/device/kernels/dataflow/reader_muladd.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, defines));

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/muladd/device/kernels/dataflow/writer_muladd.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, defines = defines));

    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/muladd/device/kernels/compute/muladd.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args,
            .defines = defines});

    printf("Num cores: %d\n", num_cores);
    printf("Shard tiles w: %d, h: %d\n", shard_tiles_w, shard_tiles_h);

    for (uint32_t i = 0, output_tile_start_id = 0; i < num_cores; i++) {
        CoreCoord core(i % num_cores_x, i / num_cores_x);
        uint32_t num_output_tiles_per_core = 0;
        if (ina_sharded || inb_sharded || inc_sharded || ind_sharded) {
            num_output_tiles_per_core = shard_tiles_w * shard_tiles_h;
        } else {
            if (core_group_1.contains(core)) {
                num_output_tiles_per_core = num_output_tiles_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_output_tiles_per_core = num_output_tiles_per_core_group_2;
            } else {
                TT_ASSERT(false, "Core not in specified core ranges");
            }
        }
        std::vector<uint32_t> reader_runtime_args = {
            input_buffer_a_address,
            input_buffer_b_address,
            input_buffer_c_address,
            input_buffer_d_address,
            num_output_tiles_per_core,
            output_tile_start_id,
        };

        std::vector<uint32_t> writer_runtime_args = {
            output_buffer_address,
            num_output_tiles_per_core,
            output_tile_start_id,
        };

        std::vector<uint32_t> compute_runtime_args = {num_output_tiles_per_core};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);
        if (!ina_sharded && !inb_sharded && !inc_sharded && !ind_sharded && !out_sharded) {
            output_tile_start_id += num_output_tiles_per_core;
        } else if (memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            output_tile_start_id = ((i + 1) / num_cores_x * num_cores_x) * shard_tiles_h * shard_tiles_w +
                                   ((i + 1) % num_cores_x) * shard_tiles_w;
        } else {  // HEIGHT_SHARDED
            output_tile_start_id += shard_tiles_h * shard_tiles_w;
        }
    }
    return {.program = std::move(program)};
}
}  // namespace ttnn::operations::muladd
