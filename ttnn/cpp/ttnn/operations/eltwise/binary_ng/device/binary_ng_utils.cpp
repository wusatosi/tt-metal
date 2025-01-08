// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "tt_metal/common/assert.hpp"

#include <fmt/core.h>
#include <fmt/format.h>
#include <magic_enum/magic_enum.hpp>

template <>
struct fmt::formatter<ttnn::operations::binary_ng::Lowercase> : fmt::formatter<std::string_view> {
    auto format(const ttnn::operations::binary_ng::Lowercase& value, fmt::format_context& ctx) const {
        auto out = ctx.out();
        for (char c : value.view) {
            *out++ = std::tolower(static_cast<unsigned char>(c));
        }
        return out;
    }
};

namespace ttnn::operations::binary_ng {

BinaryNgKernelConfig::BinaryNgKernelConfig(SubtileBroadcastType subtile_broadcast_type) {
    switch (subtile_broadcast_type) {
        case SubtileBroadcastType::NONE:
            reader_kernel = KernelName::ReaderNoBcast;
            compute_kernel = KernelName::ComputeNoBcast;
            writer_kernel = KernelName::WriterNoBcast;
            bcast_input = std::nullopt;
            break;

        case SubtileBroadcastType::SCALAR_A:
            reader_kernel = KernelName::ReaderScalarBcast;
            compute_kernel = KernelName::ComputeBcast;
            writer_kernel = KernelName::WriterNoBcast;
            bcast_input = 0;
            break;

        case SubtileBroadcastType::SCALAR_B:
            reader_kernel = KernelName::ReaderNoBcast;
            compute_kernel = KernelName::ComputeBcast;
            writer_kernel = KernelName::WriterScalarBcast;
            bcast_input = 1;
            break;

        case SubtileBroadcastType::ROW_A:
            reader_kernel = KernelName::ReaderRowBcast;
            compute_kernel = KernelName::ComputeNoBcast;
            writer_kernel = KernelName::WriterNoBcast;
            bcast_input = std::nullopt;
            break;

        case SubtileBroadcastType::ROW_B:
            reader_kernel = KernelName::ReaderNoBcast;
            compute_kernel = KernelName::ComputeNoBcast;
            writer_kernel = KernelName::WriterRowBcast;
            bcast_input = std::nullopt;
            break;

        case SubtileBroadcastType::COL_A:
            reader_kernel = KernelName::ReaderColBcast;
            compute_kernel = KernelName::ComputeBcast;
            writer_kernel = KernelName::WriterNoBcast;
            bcast_input = 0;
            break;

        case SubtileBroadcastType::COL_B:
            reader_kernel = KernelName::ReaderNoBcast;
            compute_kernel = KernelName::ComputeBcast;
            writer_kernel = KernelName::WriterColBcast;
            bcast_input = 1;
            break;

        case SubtileBroadcastType::ROW_A_COL_B:
            reader_kernel = KernelName::ReaderRowBcast;
            compute_kernel = KernelName::ComputeBcast;
            writer_kernel = KernelName::WriterColBcast;
            bcast_input = 1;
            break;

        case SubtileBroadcastType::ROW_B_COL_A:
            reader_kernel = KernelName::ReaderColBcast;
            compute_kernel = KernelName::ComputeBcast;
            writer_kernel = KernelName::WriterRowBcast;
            bcast_input = 0;
            break;
    }
}

std::string BinaryNgKernelConfig::bcast_input_str() const {
    if (bcast_input.has_value()) {
        return std::to_string(*bcast_input);
    }
    return "";
}

std::string get_kernel_file_path(KernelName kernel_name, bool is_sfpu) {
    constexpr std::string_view root = "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels";
    constexpr std::string_view dataflow = "{}/dataflow/{}";
    constexpr std::string_view compute = "{}/compute/{}";

    switch (kernel_name) {
        case KernelName::ReaderNoBcast: return fmt::format(dataflow, root, "reader_interleaved_no_bcast.cpp");
        case KernelName::ReaderRowBcast: return fmt::format(dataflow, root, "reader_interleaved_row_bcast.cpp");
        case KernelName::ReaderColBcast: return fmt::format(dataflow, root, "reader_interleaved_col_bcast.cpp");
        case KernelName::ReaderScalarBcast: return fmt::format(dataflow, root, "reader_interleaved_scalar_bcast.cpp");
        case KernelName::WriterNoBcast: return fmt::format(dataflow, root, "writer_interleaved_no_bcast.cpp");
        case KernelName::WriterRowBcast: return fmt::format(dataflow, root, "writer_interleaved_row_bcast.cpp");
        case KernelName::WriterColBcast: return fmt::format(dataflow, root, "writer_interleaved_col_bcast.cpp");
        case KernelName::WriterScalarBcast: return fmt::format(dataflow, root, "writer_interleaved_scalar_bcast.cpp");
        case KernelName::WriterScalar: return fmt::format(dataflow, root, "writer_interleaved_scalar.cpp");
        case KernelName::ComputeNoBcast:
            if (is_sfpu) {
                return fmt::format(compute, root, "eltwise_binary_sfpu_no_bcast.cpp");
            } else {
                return fmt::format(compute, root, "eltwise_binary_no_bcast.cpp");
            }
        case KernelName::ComputeBcast:
            if (is_sfpu) {
                return fmt::format(compute, root, "eltwise_binary_sfpu.cpp");
            } else {
                return fmt::format(compute, root, "eltwise_binary.cpp");
            }
        case KernelName::ComputeScalar:
            if (is_sfpu) {
                return fmt::format(compute, root, "eltwise_binary_sfpu_scalar.cpp");
            } else {
                return fmt::format(compute, root, "eltwise_binary_scalar.cpp");
            }
        default: __builtin_unreachable();  // GCC 12 doesn't compile even though we exhaustively match
    }
}

OpConfig::OpConfig(BinaryOpType binary_op_type, bool is_sfpu_op) {
    if (!is_sfpu_op) {
        fpu_binary_op = FpuBinaryOp::SUB;
        switch (binary_op_type) {
            case BinaryOpType::ADD: fpu_binary_op = FpuBinaryOp::ADD; break;
            case BinaryOpType::SUB: break;
            case BinaryOpType::MUL: fpu_binary_op = FpuBinaryOp::MUL; break;
            case BinaryOpType::DIV:
                process_rhs = unary::UnaryOpType::RECIP;
                fpu_binary_op = FpuBinaryOp::MUL;
                break;
            case BinaryOpType::RSUB:
                process_rhs = unary::UnaryOpType::NEG;
                fpu_binary_op = FpuBinaryOp::ADD;
                break;
            case BinaryOpType::GT: postprocess = unary::UnaryOpType::GTZ; break;
            case BinaryOpType::LT: postprocess = unary::UnaryOpType::LTZ; break;
            case BinaryOpType::GTE: postprocess = unary::UnaryOpType::GEZ; break;
            case BinaryOpType::LTE: postprocess = unary::UnaryOpType::LEZ; break;
            case BinaryOpType::EQ: postprocess = unary::UnaryOpType::EQZ; break;
            case BinaryOpType::NE: postprocess = unary::UnaryOpType::NEZ; break;
            case BinaryOpType::SQUARED_DIFFERENCE: postprocess = unary::UnaryOpType::SQUARE; break;
            case BinaryOpType::BIAS_GELU:
                fpu_binary_op = FpuBinaryOp::ADD;
                process_lhs = unary::UnaryOpType::GELU;
                break;
            case BinaryOpType::LOGICAL_AND:
                fpu_binary_op = FpuBinaryOp::MUL;
                postprocess = unary::UnaryOpType::NEZ;
                break;
            case BinaryOpType::LOGICAL_OR:
                fpu_binary_op = FpuBinaryOp::ADD;
                process_lhs = unary::UnaryOpType::NEZ;
                process_rhs = unary::UnaryOpType::NEZ;
                postprocess = unary::UnaryOpType::GTZ;
                break;
            case BinaryOpType::LOGICAL_XOR:
                process_lhs = unary::UnaryOpType::NEZ;
                process_rhs = unary::UnaryOpType::NEZ;
                postprocess = unary::UnaryOpType::NEZ;
                break;
            case BinaryOpType::LDEXP:
                fpu_binary_op = FpuBinaryOp::MUL;
                process_rhs = unary::UnaryOpType::EXP2;
                break;
            case BinaryOpType::LOGADDEXP:
                fpu_binary_op = FpuBinaryOp::ADD;
                process_lhs = unary::UnaryOpType::EXP;
                process_rhs = unary::UnaryOpType::EXP;
                postprocess = unary::UnaryOpType::LOG;
                break;
            case BinaryOpType::LOGADDEXP2:
                fpu_binary_op = FpuBinaryOp::ADD;
                process_lhs = unary::UnaryOpType::EXP2;
                process_rhs = unary::UnaryOpType::EXP2;
                postprocess = unary::UnaryOpType::LOG2;
                break;
            default: TT_THROW("Unsupported binary op {}", binary_op_type);
        }
    } else {
        sfpu_binary_op = SfpuBinaryOp::SUB;
        switch (binary_op_type) {
            case BinaryOpType::ADD: sfpu_binary_op = SfpuBinaryOp::ADD; break;
            case BinaryOpType::SUB: break;
            case BinaryOpType::MUL: sfpu_binary_op = SfpuBinaryOp::MUL; break;
            case BinaryOpType::DIV: sfpu_binary_op = SfpuBinaryOp::DIV; break;
            case BinaryOpType::RSUB: sfpu_binary_op = SfpuBinaryOp::RSUB; break;
            case BinaryOpType::GT: postprocess = unary::UnaryOpType::GTZ; break;
            case BinaryOpType::LT: postprocess = unary::UnaryOpType::LTZ; break;
            case BinaryOpType::GTE: postprocess = unary::UnaryOpType::GEZ; break;
            case BinaryOpType::LTE: postprocess = unary::UnaryOpType::LEZ; break;
            case BinaryOpType::EQ: postprocess = unary::UnaryOpType::EQZ; break;
            case BinaryOpType::NE: postprocess = unary::UnaryOpType::NEZ; break;
            case BinaryOpType::SQUARED_DIFFERENCE: postprocess = unary::UnaryOpType::SQUARE; break;
            case BinaryOpType::BIAS_GELU:
                sfpu_binary_op = SfpuBinaryOp::ADD;
                process_lhs = unary::UnaryOpType::GELU;
                break;
            case BinaryOpType::LOGICAL_AND:
                sfpu_binary_op = SfpuBinaryOp::MUL;
                postprocess = unary::UnaryOpType::NEZ;
                break;
            case BinaryOpType::LOGICAL_OR:
                sfpu_binary_op = SfpuBinaryOp::ADD;
                process_lhs = unary::UnaryOpType::NEZ;
                process_rhs = unary::UnaryOpType::NEZ;
                postprocess = unary::UnaryOpType::GTZ;
                break;
            case BinaryOpType::LOGICAL_XOR:
                process_lhs = unary::UnaryOpType::NEZ;
                process_rhs = unary::UnaryOpType::NEZ;
                postprocess = unary::UnaryOpType::NEZ;
                break;
            case BinaryOpType::LDEXP:
                sfpu_binary_op = SfpuBinaryOp::MUL;
                process_rhs = unary::UnaryOpType::EXP2;
                break;
            case BinaryOpType::LOGADDEXP:
                sfpu_binary_op = SfpuBinaryOp::ADD;
                process_lhs = unary::UnaryOpType::EXP;
                process_rhs = unary::UnaryOpType::EXP;
                postprocess = unary::UnaryOpType::LOG;
                break;
            case BinaryOpType::LOGADDEXP2:
                sfpu_binary_op = SfpuBinaryOp::ADD;
                process_lhs = unary::UnaryOpType::EXP2;
                process_rhs = unary::UnaryOpType::EXP2;
                postprocess = unary::UnaryOpType::LOG2;
                break;
            case BinaryOpType::BITWISE_AND: sfpu_binary_op = SfpuBinaryOp::BITWISE_AND; break;
            case BinaryOpType::BITWISE_OR: sfpu_binary_op = SfpuBinaryOp::BITWISE_OR; break;
            case BinaryOpType::BITWISE_XOR: sfpu_binary_op = SfpuBinaryOp::BITWISE_XOR; break;
            case BinaryOpType::LEFT_SHIFT: sfpu_binary_op = SfpuBinaryOp::LEFT_SHIFT; break;
            case BinaryOpType::RIGHT_SHIFT: sfpu_binary_op = SfpuBinaryOp::RIGHT_SHIFT; break;
            case BinaryOpType::POWER: sfpu_binary_op = SfpuBinaryOp::POWER; break;
            default: TT_THROW("Unsupported binary op {}", binary_op_type);
        }
    }
}

std::pair<std::string, std::string> OpConfig::get_sfpu_init_fn(DataType dtype) const {
    switch (sfpu_binary_op) {
        case SfpuBinaryOp::ADD:
            if (dtype == DataType::INT32) {
                return {"add_int32_tile_init();", "add_int32_tile"};
            } else {
                return {"add_binary_tile_init();", "add_binary_tile"};
            }
        case SfpuBinaryOp::SUB: return {"sub_binary_tile_init();", "sub_binary_tile"};
        case SfpuBinaryOp::MUL: return {"mul_binary_tile_init();", "mul_binary_tile"};
        case SfpuBinaryOp::DIV: return {"div_binary_tile_init();", "div_binary_tile"};
        case SfpuBinaryOp::POWER: return {"power_binary_tile_init();", "power_binary_tile"};
        case SfpuBinaryOp::RSUB: return {"rsub_binary_tile_init();", "rsub_binary_tile"};
        case SfpuBinaryOp::LEFT_SHIFT: return {"binary_shift_tile_init();", "binary_left_shift_tile"};
        case SfpuBinaryOp::RIGHT_SHIFT: return {"binary_shift_tile_init();", "binary_right_shift_tile"};
        case SfpuBinaryOp::BITWISE_AND: return {"binary_bitwise_tile_init();", "and_binary_tile"};
        case SfpuBinaryOp::BITWISE_OR: return {"binary_bitwise_tile_init();", "or_binary_tile"};
        case SfpuBinaryOp::BITWISE_XOR: return {"binary_bitwise_tile_init();", "xor_binary_tile"};
        default: TT_THROW("Unsupported sfpu binary op {}", sfpu_binary_op); return {"", ""};
    }
}

std::map<std::string, std::string> OpConfig::as_sfpu_defines(DataType dtype) const {
    std::map<std::string, std::string> defines;
    auto [tile_init, tile_fn] = get_sfpu_init_fn(dtype);
    defines["BINARY_SFPU_INIT"] = fmt::format("{}", tile_init);
    defines["BINARY_SFPU_OP"] = fmt::format("{}", tile_fn);

    return defines;
}

std::map<std::string, std::string> OpConfig::as_defines() const {
    std::map<std::string, std::string> defines;

    auto binary_op_str = magic_enum::enum_name(fpu_binary_op);
    defines["BINARY_OP"] = fmt::format("{}_tiles", Lowercase{binary_op_str});
    defines["BINARY_OP_TYPE"] = fmt::format("EltwiseBinaryType::ELW{}", binary_op_str);

    return defines;
}

void add_activation_defines(
    std::map<std::string, std::string>& defines,
    tt::stl::Span<const unary::UnaryOpType> activations,
    std::string_view operand) {
    auto prepend_separator = false;
    std::string process = "";

    for (auto& a : activations) {
        if (prepend_separator) {
            process += ';';
        }
        prepend_separator = true;
        process += fmt::format("PROCESS_ACTIVATION({}, i)", magic_enum::enum_name(a));
        unary::utils::update_macro_defines(a, defines);
    }

    defines[fmt::format("PROCESS_{}_ACTIVATIONS(i)", operand)] = process;
}

}  // namespace ttnn::operations::binary_ng
