// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <array>
#include <memory>
#include <optional>

#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

#include "ttnn/core.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp"
#include "ttnn/operations/copy.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/quantization/quantization.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/embedding/embedding.hpp"
#include "ttnn/operations/embedding_backward/embedding_backward.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/moreh/moreh_cumsum/moreh_cumsum.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/pool/generic/generic_pools.hpp"
#include "ttnn/operations/pool/upsample/upsample.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/reduction/prod/prod.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn {
namespace operations {
namespace binary {
namespace test {

std::tuple<::ttnn::Tensor, ::ttnn::Tensor> forward(
    ::ttnn::Tensor v1,
    ::ttnn::Tensor v2,
    ::ttnn::Tensor v3,
    ::ttnn::Tensor v4,
    ::ttnn::Tensor v5,
    ::ttnn::Tensor v6,
    ::ttnn::Tensor v7,
    ::ttnn::Tensor v8,
    ::ttnn::Tensor v9,
    ::ttnn::Tensor v10,
    ::ttnn::Tensor v11,
    ::ttnn::Tensor v12,
    ::ttnn::Tensor v13,
    ttnn::IDevice* v14) {
    // ttnn::IDevice* v14 = ttnn::DeviceGetter::getInstance();
    ::ttnn::Tensor v15 = ttnn::transpose(
        v2,
        1,
        2,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v2, false);
    ::ttnn::Tensor v16 = ttnn::transpose(
        v15,
        2,
        3,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v15, false);
    ::ttnn::Tensor v17 = ttnn::transpose(
        v1,
        1,
        2,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v1, false);
    ::ttnn::Tensor v18 = ttnn::clamp(
        v17,
        0.000000f,
        6.000000f,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v17, false);
    ::ttnn::Tensor v19 = ttnn::transpose(
        v18,
        2,
        3,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v18, false);
    ::ttnn::Tensor v20 = ttnn::reshape(v19, ::std::vector<int32_t>{1, 1, 196, 384}, ::std::nullopt);
    ttnn::deallocate(v19, false);
    ::ttnn::Tensor v21 = ttnn::from_device(v20);
    ttnn::deallocate(v20, false);
    ::ttnn::Tensor v22 = ttnn::to_layout(
        v21,
        ::ttnn::Layout::ROW_MAJOR,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::SYSTEM_MEMORY},
        static_cast<::ttnn::IDevice*>(nullptr));
    ttnn::deallocate(v21, false);
    ::ttnn::Tensor v23 = ttnn::to_device(
        v22,
        v14,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v22, false);
    ::std::tuple<::ttnn::Tensor, uint32_t, uint32_t, ::ttnn::Tensor, ::std::optional<::ttnn::Tensor>> v24 =
        ttnn::conv2d(
            v23,
            v10,
            v14,
            384,
            384,
            1,
            14,
            14,
            ::std::array<uint32_t, 2>{3, 3},
            ::std::array<uint32_t, 2>{1, 1},
            ::std::array<uint32_t, 2>{1, 1},
            ::std::array<uint32_t, 2>{1, 1},
            384,
            ::std::nullopt,
            ::ttnn::operations::conv::conv2d::Conv2dConfig{
                .dtype = ::ttnn::DataType::FLOAT32, .weights_dtype = ::ttnn::DataType::FLOAT32},
            ::std::nullopt,
            ::ttnn::MemoryConfig{
                .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v25 = ::std::get<0>(v24);
    ttnn::deallocate(v23, false);
    ttnn::deallocate(v10, false);
    ::ttnn::Tensor v26 = ttnn::reshape(v25, ::std::vector<int32_t>{1, 14, 14, 384}, ::std::nullopt);
    ttnn::deallocate(v25, false);
    ::ttnn::Tensor v27 = ttnn::multiply(
        v26,
        v3,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v26, false);
    ttnn::deallocate(v3, false);
    ::ttnn::Tensor v28 = ttnn::add(
        v27,
        v4,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v27, false);
    ttnn::deallocate(v4, false);
    ::ttnn::Tensor v29 = ttnn::clamp(
        v28,
        0.000000f,
        6.000000f,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v28, false);
    ::ttnn::Tensor v30 = ttnn::reshape(v29, ::std::vector<int32_t>{1, 1, 196, 384}, ::std::nullopt);
    ttnn::deallocate(v29, false);
    ::ttnn::Tensor v31 = ttnn::from_device(v30);
    ttnn::deallocate(v30, false);
    ::ttnn::Tensor v32 = ttnn::to_layout(
        v31,
        ::ttnn::Layout::ROW_MAJOR,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::SYSTEM_MEMORY},
        static_cast<::ttnn::IDevice*>(nullptr));
    ttnn::deallocate(v31, false);
    ::ttnn::Tensor v33 = ttnn::to_device(
        v32,
        v14,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v32, false);
    ::std::tuple<::ttnn::Tensor, uint32_t, uint32_t, ::ttnn::Tensor, ::std::optional<::ttnn::Tensor>> v34 =
        ttnn::conv2d(
            v33,
            v11,
            v14,
            384,
            64,
            1,
            14,
            14,
            ::std::array<uint32_t, 2>{1, 1},
            ::std::array<uint32_t, 2>{1, 1},
            ::std::array<uint32_t, 2>{0, 0},
            ::std::array<uint32_t, 2>{1, 1},
            1,
            ::std::nullopt,
            ::ttnn::operations::conv::conv2d::Conv2dConfig{
                .dtype = ::ttnn::DataType::FLOAT32, .weights_dtype = ::ttnn::DataType::FLOAT32},
            ::std::nullopt,
            ::ttnn::MemoryConfig{
                .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v35 = ::std::get<0>(v34);
    ttnn::deallocate(v33, false);
    ttnn::deallocate(v11, false);
    ::ttnn::Tensor v36 = ttnn::reshape(v35, ::std::vector<int32_t>{1, 14, 14, 64}, ::std::nullopt);
    ttnn::deallocate(v35, false);
    ::ttnn::Tensor v37 = ttnn::multiply(
        v36,
        v5,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v36, false);
    ttnn::deallocate(v5, false);
    ::ttnn::Tensor v38 = ttnn::add(
        v37,
        v6,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v37, false);
    ttnn::deallocate(v6, false);
    ::ttnn::Tensor v39 = ttnn::reshape(v16, ::std::vector<int32_t>{1, 1, 196, 64}, ::std::nullopt);
    ttnn::deallocate(v16, false);
    ::ttnn::Tensor v40 = ttnn::reshape(v38, ::std::vector<int32_t>{1, 1, 196, 64}, ::std::nullopt);
    ttnn::deallocate(v38, false);
    ::ttnn::Tensor v41 = ttnn::add(
        v39,
        v40,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v40, false);
    ttnn::deallocate(v39, false);
    ::ttnn::Tensor v42 = ttnn::from_device(v41);
    ttnn::deallocate(v41, false);
    ::ttnn::Tensor v43 = ttnn::to_layout(
        v42,
        ::ttnn::Layout::ROW_MAJOR,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::SYSTEM_MEMORY},
        static_cast<::ttnn::IDevice*>(nullptr));
    ttnn::deallocate(v42, false);
    ::ttnn::Tensor v44 = ttnn::to_device(
        v43,
        v14,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v43, false);
    ::std::tuple<::ttnn::Tensor, uint32_t, uint32_t, ::ttnn::Tensor, ::std::optional<::ttnn::Tensor>> v45 =
        ttnn::conv2d(
            v44,
            v12,
            v14,
            64,
            384,
            1,
            14,
            14,
            ::std::array<uint32_t, 2>{1, 1},
            ::std::array<uint32_t, 2>{1, 1},
            ::std::array<uint32_t, 2>{0, 0},
            ::std::array<uint32_t, 2>{1, 1},
            1,
            ::std::nullopt,
            ::ttnn::operations::conv::conv2d::Conv2dConfig{
                .dtype = ::ttnn::DataType::FLOAT32, .weights_dtype = ::ttnn::DataType::FLOAT32},
            ::std::nullopt,
            ::ttnn::MemoryConfig{
                .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v46 = ::std::get<0>(v45);
    ttnn::deallocate(v44, false);
    ttnn::deallocate(v12, false);
    ::ttnn::Tensor v47 = ttnn::reshape(v46, ::std::vector<int32_t>{1, 14, 14, 384}, ::std::nullopt);
    ttnn::deallocate(v46, false);
    ::ttnn::Tensor v48 = ttnn::multiply(
        v47,
        v7,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v47, false);
    ttnn::deallocate(v7, false);
    ::ttnn::Tensor v49 = ttnn::add(
        v48,
        v8,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v48, false);
    ttnn::deallocate(v8, false);
    ::ttnn::Tensor v50 = ttnn::clamp(
        v49,
        0.000000f,
        6.000000f,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v49, false);
    ::ttnn::Tensor v51 = ttnn::reshape(v50, ::std::vector<int32_t>{1, 1, 196, 384}, ::std::nullopt);
    ttnn::deallocate(v50, false);
    ::ttnn::Tensor v52 = ttnn::from_device(v51);
    ttnn::deallocate(v51, false);
    ::ttnn::Tensor v53 = ttnn::to_layout(
        v52,
        ::ttnn::Layout::ROW_MAJOR,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::SYSTEM_MEMORY},
        static_cast<::ttnn::IDevice*>(nullptr));
    ttnn::deallocate(v52, false);
    ::ttnn::Tensor v54 = ttnn::to_device(
        v53,
        v14,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v53, false);
    ::std::tuple<::ttnn::Tensor, uint32_t, uint32_t, ::ttnn::Tensor, ::std::optional<::ttnn::Tensor>> v55 =
        ttnn::conv2d(
            v54,
            v13,
            v14,
            384,
            384,
            1,
            14,
            14,
            ::std::array<uint32_t, 2>{3, 3},
            ::std::array<uint32_t, 2>{1, 1},
            ::std::array<uint32_t, 2>{1, 1},
            ::std::array<uint32_t, 2>{1, 1},
            384,
            ::std::nullopt,
            ::ttnn::operations::conv::conv2d::Conv2dConfig{
                .dtype = ::ttnn::DataType::FLOAT32, .weights_dtype = ::ttnn::DataType::FLOAT32},
            ::std::nullopt,
            ::ttnn::MemoryConfig{
                .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v56 = ::std::get<0>(v55);
    ttnn::deallocate(v54, false);
    ttnn::deallocate(v13, false);
    ::ttnn::Tensor v57 = ttnn::reshape(v56, ::std::vector<int32_t>{1, 14, 14, 384}, ::std::nullopt);
    ttnn::deallocate(v56, false);
    ::ttnn::Tensor v58 = ttnn::transpose(
        v57,
        2,
        3,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v57, false);
    ::ttnn::Tensor v59 = ttnn::transpose(
        v58,
        1,
        2,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v58, false);
    return std::make_tuple(v59, v9);
}

std::tuple<
    ::ttnn::Tensor,
    ::ttnn::Tensor,
    ::ttnn::Tensor,
    ::ttnn::Tensor,
    ::ttnn::Tensor,
    ::ttnn::Tensor,
    ::ttnn::Tensor,
    ::ttnn::Tensor,
    ::ttnn::Tensor,
    ::ttnn::Tensor,
    ::ttnn::Tensor,
    ::ttnn::Tensor,
    ::ttnn::Tensor>
createInputsFor_forward(ttnn::IDevice* v1) {
    // ttnn::IDevice* v1 = ttnn::DeviceGetter::getInstance();
    ::ttnn::Tensor v2 = ttnn::ones(
        ::ttnn::Shape({1, 384, 14, 14}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::TILE,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v3 = ttnn::to_device(
        v2,
        v1,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v4 = ttnn::ones(
        ::ttnn::Shape({1, 64, 14, 14}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::TILE,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v5 = ttnn::to_device(
        v4,
        v1,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v6 = ttnn::ones(
        ::ttnn::Shape({1, 1, 1, 384}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::TILE,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v7 = ttnn::to_device(
        v6,
        v1,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v8 = ttnn::ones(
        ::ttnn::Shape({1, 1, 1, 384}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::TILE,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v9 = ttnn::to_device(
        v8,
        v1,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v10 = ttnn::ones(
        ::ttnn::Shape({1, 1, 1, 64}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::TILE,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v11 = ttnn::to_device(
        v10,
        v1,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v12 = ttnn::ones(
        ::ttnn::Shape({1, 1, 1, 64}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::TILE,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v13 = ttnn::to_device(
        v12,
        v1,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v14 = ttnn::ones(
        ::ttnn::Shape({1, 1, 1, 384}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::TILE,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v15 = ttnn::to_device(
        v14,
        v1,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v16 = ttnn::ones(
        ::ttnn::Shape({1, 1, 1, 384}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::TILE,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v17 = ttnn::to_device(
        v16,
        v1,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v18 = ttnn::ones(
        ::ttnn::Shape({384}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::TILE,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v19 = ttnn::to_device(
        v18,
        v1,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v20 = ttnn::ones(
        ::ttnn::Shape({384, 1, 3, 3}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::ROW_MAJOR,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED,
            .buffer_type = ::ttnn::BufferType::SYSTEM_MEMORY});
    ::ttnn::Tensor v21 = ttnn::ones(
        ::ttnn::Shape({64, 384, 1, 1}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::ROW_MAJOR,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED,
            .buffer_type = ::ttnn::BufferType::SYSTEM_MEMORY});
    ::ttnn::Tensor v22 = ttnn::ones(
        ::ttnn::Shape({384, 64, 1, 1}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::ROW_MAJOR,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED,
            .buffer_type = ::ttnn::BufferType::SYSTEM_MEMORY});
    ::ttnn::Tensor v23 = ttnn::ones(
        ::ttnn::Shape({384, 1, 3, 3}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::ROW_MAJOR,
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            .memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED,
            .buffer_type = ::ttnn::BufferType::SYSTEM_MEMORY});
    return std::make_tuple(v3, v5, v7, v9, v11, v13, v15, v17, v19, v20, v21, v22, v23);
}

TEST(TTNNGraphRepro, TTNNGraphReproERROR) {
    // auto param = GetParam();
    // auto& device = *device_;
    const chip_id_t device_id = 0;

    // Sets the size for L1 small on the device - 16KB
    // The halo op which is contained in the Conv2D op uses L1 small memory
    // Without this, the convolution operation will fail due to L1_SMALL Out of Memory error
    const size_t L1_small_size = 32768;

    IDevice* device = CreateDevice(device_id, 1, L1_small_size);
    enable_program_cache(*device);

    {
        ::ttnn::Tensor v1;
        ::ttnn::Tensor v2;
        ::ttnn::Tensor v3;
        ::ttnn::Tensor v4;
        ::ttnn::Tensor v5;
        ::ttnn::Tensor v6;
        ::ttnn::Tensor v7;
        ::ttnn::Tensor v8;
        ::ttnn::Tensor v9;
        ::ttnn::Tensor v10;
        ::ttnn::Tensor v11;
        ::ttnn::Tensor v12;
        ::ttnn::Tensor v13;
        std::tie(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13) = createInputsFor_forward(device);
        ::ttnn::Tensor v14;
        ::ttnn::Tensor v15;
        std::tie(v14, v15) = forward(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, device);
        // ::ttnn::Tensor v1;
        // ::ttnn::Tensor v2;
        // ::ttnn::Tensor v3;
        // ::ttnn::Tensor v4;
        // ::ttnn::Tensor v5;
        // ::ttnn::Tensor v6;
        // ::ttnn::Tensor v7;
        // ::ttnn::Tensor v8;
        // ::ttnn::Tensor v9;
        // ::ttnn::Tensor v10;
        // ::ttnn::Tensor v11;
        // ::ttnn::Tensor v12;
        // ::ttnn::Tensor v13;
        // std::tie(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13) = param;
        // ::ttnn::Tensor v14;
        // ::ttnn::Tensor v15;
        // std::tie(v14, v15) = forward(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13);
    }

    // ttnn::close_device(device);
    CloseDevice(device);
}

// INSTANTIATE_TEST_SUITE_P(

//     ::ttnn::Tensor v1;
//     ::ttnn::Tensor v2;
//     ::ttnn::Tensor v3;
//     ::ttnn::Tensor v4;
//     ::ttnn::Tensor v5;
//     ::ttnn::Tensor v6;
//     ::ttnn::Tensor v7;
//     ::ttnn::Tensor v8;
//     ::ttnn::Tensor v9;
//     ::ttnn::Tensor v10;
//     ::ttnn::Tensor v11;
//     ::ttnn::Tensor v12;
//     ::ttnn::Tensor v13;
//     std::tie(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13) = createInputsFor_forward();

//     TTNNGraphReproTests,
//     TTNNGraphRepro,
//     ::testing::Values(
//         std::tie(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13)
//     )

// );

}  // namespace test
}  // namespace binary
}  // namespace operations
}  // namespace ttnn
