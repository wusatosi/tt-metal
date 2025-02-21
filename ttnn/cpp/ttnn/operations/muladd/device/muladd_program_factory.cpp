#include "muladd_program_factory.hpp"
#include "tt-metalium/base_types.hpp"
#include "ttnn/common/queue_id.hpp"

namespace ttnn::operations::muladd {

using namespace tt::constants;
using namespace tt::tt_metal;

operation::ProgramWithCallbacks single_core_muladd(
    const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, Tensor& output, MathFidelity math_fidelity) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

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
        [reader_kernel_id, writer_kernel_id, compute_kernel_id, core](
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
            };

            std::vector<uint32_t> writer_runtime_args = {output_buffer->address()};

            std::vector<uint32_t> compute_runtime_args = {};

            SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
            SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);
        };
    set_runtime_args(program, a, b, c, d, output);

    // auto override_runtime_arguments_callback = [set_runtime_args](
    //     const void* operation,
    //     Program& program,
    //     const std::vector<Tensor>& input_tensors,
    //     const std::vector<std::optional<const Tensor>>&,
    //     std::vector<Tensor>& output_tensors) {
    //         set_runtime_args(program,
    //         input_tensors.at(0),input_tensors.at(1),input_tensors.at(2),input_tensors.at(3), output_tensors.at(0));
    //     };

    return {.program = std::move(program)};
}
}  // namespace ttnn::operations::muladd
