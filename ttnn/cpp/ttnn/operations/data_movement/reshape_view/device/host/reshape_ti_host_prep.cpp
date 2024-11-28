#include <math.h>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape_common.hpp"

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"

#define MASK_64      0xFFFFFFFFFFFFFFC0
#define OFFSET_64    0x000000000000003F
#define MASK_16      0xFFFFFFFFFFFFFFF0
#define OFFSET_16    0x000000000000000F

namespace ttnn::operations::data_movement::tile_reshape{

operation::ProgramWithCallbacks tile_reshape_preparer(const Tensor& input, const Tensor& output, uint32_t pad_value)
{
    printf("tile_reshape_preparer\n");
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    //TODO fix padding stuff
    //uint64_t casted_pad_value = (uint64_t)(std::get(pad_value));
    uint64_t casted_pad_value = pad_value;
    //get datum size
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    const uint32_t data_size = input.element_size();
    //Correct pad val, break into upper and lower
    uint32_t lower_pad_val = 0;
    uint32_t upper_pad_val = 0;
    CoreRange core({0, 0}, {0, 0});

    tt::tt_metal::Device *device = input.device();
    switch(data_size)
    {
        case 1:
            lower_pad_val = (uint32_t) (casted_pad_value&0xFF);
            lower_pad_val = lower_pad_val | (lower_pad_val << 8) | (lower_pad_val << 16) | (lower_pad_val << 24);
            upper_pad_val = lower_pad_val;
            break;
        case 2:
            lower_pad_val = (uint32_t) (casted_pad_value&0xFFFF);
            lower_pad_val = lower_pad_val | (lower_pad_val << 16);
            upper_pad_val = lower_pad_val;
            break;
        case 4:
            lower_pad_val = (uint32_t) (casted_pad_value&0xFFFFFFFF);
            upper_pad_val = lower_pad_val;
            break;
        case 8:
            lower_pad_val = (uint32_t) (casted_pad_value&0xFFFFFFFF);
            upper_pad_val = (uint32_t) ((casted_pad_value>>32)&0xFFFFFFFF);
            break;
        default:
            TT_THROW("Unsupported data size");
    }
    Tile tiling = input.get_tile();
    //I am assuming 4 faces traversed bottom left, bottom right, top left, top right
    ttnn::Shape input_log_shape = ttnn::Shape(input.get_logical_shape().view());
    ttnn::Shape output_log_shape = ttnn::Shape(output.get_logical_shape().view());
    const uint32_t tile_width = tiling.get_width();
    const uint32_t tile_height = tiling.get_height();
    const uint32_t tile_hw = tile_width * tile_height;
    const uint32_t tile_volume = tile_hw * data_size;
    tt::log_debug("tiled reshape");
    tt::log_debug("input shape: {}", input_log_shape);
    tt::log_debug("output shape: {}", output_log_shape);
    tt::log_debug("data size: {}", data_size);
    tt::log_debug("tile width: {}", tile_width);
    tt::log_debug("tile height: {}", tile_height);
    tt::log_debug("tile volume: {}", tile_volume);

    tt::tt_metal::Buffer *src_buffer = input.buffer();
    tt::tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    TT_ASSERT(data_size * tile_width%2 == 0);
    TT_ASSERT(data_size * tile_height%2 == 0);
    uint32_t src_w_tiles = (input_log_shape[-1] + tile_width - 1) / tile_width;
    uint32_t dst_w_tiles = (output_log_shape[-1] + tile_width - 1) / tile_width;

    uint32_t face_width_bytes = data_size * tile_width / 2;//By assumption this is an integer
    uint32_t face_height = tile_height / 2; //by assumption this is an integer
    bool     face_alligned_16 = (face_width_bytes%16==0);
    bool     face_alligned_64 = (face_width_bytes%64==0);
    uint32_t face_size_bytes = face_height * face_width_bytes;
    uint32_t tile_size_bytes  = tile_width * tile_height * data_size; //By assumption this is a multiple of 64
    TT_ASSERT(tile_size_bytes%64==0);
    const uint32_t min_line_size = (((src_w_tiles) * (face_width_bytes) * 2 +face_width_bytes)&MASK_64) + 64;
    const uint32_t cb_size = ((dst_w_tiles*tile_size_bytes)&MASK_64) + 2*min_line_size + 128;


    uint32_t src0_cb_index = 0;


    tt::tt_metal::CircularBufferConfig cb_src0_config = tt::tt_metal::CircularBufferConfig(cb_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, cb_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);
    //set the runtime args
    //set the compile time args
    bool src0_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> compile_time_args = {
        (std::uint32_t) src0_is_dram,
        data_size,
        tile_width,
        tile_height,
        src_w_tiles
    };

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/tile_reshape_interleaved.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(compile_time_args));
    std::vector<uint32_t> reader_runtime_args = {
        src_buffer->address(),
        dst_buffer->address(),
        src0_cb_index,
        input_log_shape[-1] * data_size,
        input_log_shape[-2],
        input_log_shape[-3],
        output_log_shape[-1] * data_size,
        output_log_shape[-2],
        output_log_shape[-3],
        (input_log_shape[-2] + tile_width - 1) / tile_width,
        (output_log_shape[-1] + tile_width - 1) / tile_width,
        (output_log_shape[-2] + tile_width - 1) / tile_width,
        output_log_shape[-2]%tile_height == 0 ? 0 : tile_height - output_log_shape[-2]%tile_height,
        upper_pad_val,
        lower_pad_val
    };
    tt::tt_metal::SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        reader_runtime_args
    );
    return {.program=std::move(program)};
}
}; // namespace ttnn::operations::data_movement::rm_reshape
