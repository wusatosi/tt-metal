// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "emitc.hpp"

namespace ttnn {
namespace test {

// ******************* Conv2dF32 *******************
::ttnn::Tensor conv2d_f32(::ttnn::Tensor v1, ::ttnn::Tensor v2, ::ttnn::Tensor v3) {
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v1, ::std::vector<int32_t>{1, 1, 16384, 64}, ::std::nullopt);
  ttnn::deallocate(v1, false);
  ::ttnn::Tensor v6 = ttnn::from_device(v5);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::to_device(v7, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ttnn::deallocate(v7, false);
  ::std::variant<::ttnn::Tensor, ::std::tuple<::ttnn::Tensor, ::std::tuple<uint32_t, uint32_t>>, ::std::tuple<::ttnn::Tensor, ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>, ::std::tuple<::ttnn::Tensor, ::std::tuple<uint32_t, uint32_t>, ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>> v9 = ttnn::conv2d(v8, v2, v4, 64, 64, 16, 32, 32, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::std::nullopt, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v10 = ::std::get<0>(v9);
  ttnn::deallocate(v8, false);
  ttnn::deallocate(v3, false);
  ttnn::deallocate(v2, false);
  ::ttnn::Tensor v11 = ttnn::reshape(v10, ::std::vector<int32_t>{16, 30, 30, 64}, ::std::nullopt);
  ttnn::deallocate(v10, false);
  return v11;
}

std::tuple<::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor> create_inputs_for_conv2d_f32() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({16, 32, 32, 64}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({64, 64, 3, 3}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY});
  ::ttnn::Tensor v5 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v6 = ttnn::to_device(v5, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  return std::make_tuple(v3, v4, v6);
}

TEST(EmitC, Conv2df32) {
  ::ttnn::Tensor v1;
  ::ttnn::Tensor v2;
  ::ttnn::Tensor v3;
  std::tie(v1, v2, v3) = create_inputs_for_conv2d_f32();  
  ::ttnn::Tensor v4 = conv2d_f32(v1, v2, v3);
}

// ******************* PointwiseConv2dF32 *******************
::ttnn::Tensor pointwise_conv2d_1x1_f32(::ttnn::Tensor v1, ::ttnn::Tensor v2, ::ttnn::Tensor v3) {
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v1, ::std::vector<int32_t>{1, 1, 16384, 64}, ::std::nullopt);
  ttnn::deallocate(v1, false);
  ::ttnn::Tensor v6 = ttnn::from_device(v5);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::to_device(v7, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ttnn::deallocate(v7, false);
  ::std::variant<::ttnn::Tensor, ::std::tuple<::ttnn::Tensor, ::std::tuple<uint32_t, uint32_t>>, ::std::tuple<::ttnn::Tensor, ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>, ::std::tuple<::ttnn::Tensor, ::std::tuple<uint32_t, uint32_t>, ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>> v9 = ttnn::conv2d(v8, v2, v4, 64, 32, 16, 32, 32, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::std::nullopt, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v10 = ::std::get<0>(v9);
  ttnn::deallocate(v8, false);
  ttnn::deallocate(v3, false);
  ttnn::deallocate(v2, false);
  ::ttnn::Tensor v11 = ttnn::reshape(v10, ::std::vector<int32_t>{16, 32, 32, 32}, ::std::nullopt);
  ttnn::deallocate(v10, false);
  return v11;
}

std::tuple<::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor> create_inputs_for_pointwise_conv2d_1x1_f32() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({16, 32, 32, 64}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({32, 64, 1, 1}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY});
  ::ttnn::Tensor v5 = ttnn::ones(::ttnn::Shape({1, 1, 1, 32}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v6 = ttnn::to_device(v5, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  return std::make_tuple(v3, v4, v6);
}

TEST(EmitC, PointwiseConv2dF32) {
  ::ttnn::Tensor v1;
  ::ttnn::Tensor v2;
  ::ttnn::Tensor v3;
  std::tie(v1, v2, v3) = create_inputs_for_pointwise_conv2d_1x1_f32();  
  ::ttnn::Tensor v4 = pointwise_conv2d_1x1_f32(v1, v2, v3);
}

// ******************* DepthwiseConv2dF32 *******************
::ttnn::Tensor depthwise_conv2d_f32(::ttnn::Tensor v1, ::ttnn::Tensor v2, ::ttnn::Tensor v3) {
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v1, ::std::vector<int32_t>{1, 1, 16384, 64}, ::std::nullopt);
  ttnn::deallocate(v1, false);
  ::ttnn::Tensor v6 = ttnn::from_device(v5);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::to_device(v7, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ttnn::deallocate(v7, false);
  ::std::variant<::ttnn::Tensor, ::std::tuple<::ttnn::Tensor, ::std::tuple<uint32_t, uint32_t>>, ::std::tuple<::ttnn::Tensor, ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>, ::std::tuple<::ttnn::Tensor, ::std::tuple<uint32_t, uint32_t>, ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>> v9 = ttnn::conv2d(v8, v2, v4, 64, 64, 16, 32, 32, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 64, v3, ::std::nullopt, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v10 = ::std::get<0>(v9);
  ttnn::deallocate(v8, false);
  ttnn::deallocate(v3, false);
  ttnn::deallocate(v2, false);
  ::ttnn::Tensor v11 = ttnn::reshape(v10, ::std::vector<int32_t>{16, 30, 30, 64}, ::std::nullopt);
  ttnn::deallocate(v10, false);
  return v11;
}

std::tuple<::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor> create_inputs_for_depthwise_conv2d_f32() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({16, 32, 32, 64}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({64, 1, 3, 3}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY});
  ::ttnn::Tensor v5 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v6 = ttnn::to_device(v5, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  return std::make_tuple(v3, v4, v6);
}

TEST(EmitC, DepthwiseConv2dF32) {
  ::ttnn::Tensor v1;
  ::ttnn::Tensor v2;
  ::ttnn::Tensor v3;
  std::tie(v1, v2, v3) = create_inputs_for_depthwise_conv2d_f32();
  ::ttnn::Tensor v4 = depthwise_conv2d_f32(v1, v2, v3);
}

// ******************* DepthwiseSeparableConv2dF32 *******************
::ttnn::Tensor depthwise_separable_conv2d_f32(::ttnn::Tensor v1, ::ttnn::Tensor v2, ::ttnn::Tensor v3, ::ttnn::Tensor v4, ::ttnn::Tensor v5) {
  ttnn::distributed::MeshDevice* v6 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v7 = ttnn::reshape(v1, ::std::vector<int32_t>{1, 1, 32768, 64}, ::std::nullopt);
  ttnn::deallocate(v1, false);
  ::ttnn::Tensor v8 = ttnn::from_device(v7);
  ttnn::deallocate(v7, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::to_device(v9, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ttnn::deallocate(v9, false);
  ::std::variant<::ttnn::Tensor, ::std::tuple<::ttnn::Tensor, ::std::tuple<uint32_t, uint32_t>>, ::std::tuple<::ttnn::Tensor, ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>, ::std::tuple<::ttnn::Tensor, ::std::tuple<uint32_t, uint32_t>, ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>> v11 = ttnn::conv2d(v10, v2, v6, 64, 64, 32, 32, 32, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 64, v3, ::std::nullopt, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v12 = ::std::get<0>(v11);
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v3, false);
  ttnn::deallocate(v2, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::to_layout(v13, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v13, false);
  ::ttnn::Tensor v15 = ttnn::to_device(v14, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ttnn::deallocate(v14, false);
  ::std::variant<::ttnn::Tensor, ::std::tuple<::ttnn::Tensor, ::std::tuple<uint32_t, uint32_t>>, ::std::tuple<::ttnn::Tensor, ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>, ::std::tuple<::ttnn::Tensor, ::std::tuple<uint32_t, uint32_t>, ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>> v16 = ttnn::conv2d(v15, v4, v6, 64, 256, 32, 30, 30, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v5, ::std::nullopt, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v17 = ::std::get<0>(v16);
  ttnn::deallocate(v15, false);
  ttnn::deallocate(v5, false);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v18 = ttnn::reshape(v17, ::std::vector<int32_t>{32, 30, 30, 256}, ::std::nullopt);
  ttnn::deallocate(v17, false);
  return v18;
}

std::tuple<::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor> create_inputs_for_depthwise_separable_conv2d_f32() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({32, 32, 32, 64}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({64, 1, 3, 3}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY});
  ::ttnn::Tensor v5 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v6 = ttnn::to_device(v5, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v7 = ttnn::ones(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY});
  ::ttnn::Tensor v8 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v9 = ttnn::to_device(v8, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  return std::make_tuple(v3, v4, v6, v7, v9);
}

TEST(EmitC, DepthwiseSeparableConv2dF32) {
  ::ttnn::Tensor v1;
  ::ttnn::Tensor v2;
  ::ttnn::Tensor v3;
  ::ttnn::Tensor v4;
  ::ttnn::Tensor v5;
  std::tie(v1, v2, v3, v4, v5) = create_inputs_for_depthwise_separable_conv2d_f32();  
  ::ttnn::Tensor v6 = depthwise_separable_conv2d_f32(v1, v2, v3, v4, v5);
}

// ******************* GroupedConv2dF32 *******************
::ttnn::Tensor grouped_conv2d_f32(::ttnn::Tensor v1, ::ttnn::Tensor v2, ::ttnn::Tensor v3) {
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v1, ::std::vector<int32_t>{1, 1, 16384, 64}, ::std::nullopt);
  ttnn::deallocate(v1, false);
  ::ttnn::Tensor v6 = ttnn::from_device(v5);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::to_device(v7, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ttnn::deallocate(v7, false);
  ::std::variant<::ttnn::Tensor, ::std::tuple<::ttnn::Tensor, ::std::tuple<uint32_t, uint32_t>>, ::std::tuple<::ttnn::Tensor, ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>, ::std::tuple<::ttnn::Tensor, ::std::tuple<uint32_t, uint32_t>, ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>> v9 = ttnn::conv2d(v8, v2, v4, 64, 64, 16, 32, 32, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 4, v3, ::std::nullopt, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v10 = ::std::get<0>(v9);
  ttnn::deallocate(v8, false);
  ttnn::deallocate(v3, false);
  ttnn::deallocate(v2, false);
  ::ttnn::Tensor v11 = ttnn::reshape(v10, ::std::vector<int32_t>{16, 30, 30, 64}, ::std::nullopt);
  ttnn::deallocate(v10, false);
  return v11;
}

std::tuple<::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor> create_inputs_for_grouped_conv2d_f32() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({16, 32, 32, 64}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({64, 16, 3, 3}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY});
  ::ttnn::Tensor v5 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v6 = ttnn::to_device(v5, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  return std::make_tuple(v3, v4, v6);
}

TEST(EmitC, GroupedConv2dF32) {
  ::ttnn::Tensor v1;
  ::ttnn::Tensor v2;
  ::ttnn::Tensor v3;
  std::tie(v1, v2, v3) = create_inputs_for_grouped_conv2d_f32();
  ::ttnn::Tensor v4 = grouped_conv2d_f32(v1, v2, v3);
}

// ******************* DilatedConv2dB *******************
::ttnn::Tensor dilated_even_conv2d_f32(::ttnn::Tensor v1, ::ttnn::Tensor v2, ::ttnn::Tensor v3) {
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v1, ::std::vector<int32_t>{1, 1, 16384, 64}, ::std::nullopt);
  ttnn::deallocate(v1, false);
  ::ttnn::Tensor v6 = ttnn::from_device(v5);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::to_device(v7, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ttnn::deallocate(v7, false);
  ::std::variant<::ttnn::Tensor, ::std::tuple<::ttnn::Tensor, ::std::tuple<uint32_t, uint32_t>>, ::std::tuple<::ttnn::Tensor, ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>, ::std::tuple<::ttnn::Tensor, ::std::tuple<uint32_t, uint32_t>, ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>> v9 = ttnn::conv2d(v8, v2, v4, 64, 64, 16, 32, 32, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{2, 2}, 1, v3, ::std::nullopt, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v10 = ::std::get<0>(v9);
  ttnn::deallocate(v8, false);
  ttnn::deallocate(v3, false);
  ttnn::deallocate(v2, false);
  ::ttnn::Tensor v11 = ttnn::reshape(v10, ::std::vector<int32_t>{16, 28, 28, 64}, ::std::nullopt);
  ttnn::deallocate(v10, false);
  return v11;
}

::ttnn::Tensor dilated_uneven_conv2d_f32(::ttnn::Tensor v1, ::ttnn::Tensor v2, ::ttnn::Tensor v3) {
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v1, ::std::vector<int32_t>{1, 1, 16384, 64}, ::std::nullopt);
  ttnn::deallocate(v1, false);
  ::ttnn::Tensor v6 = ttnn::to_layout(v5, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM}, v4);
  ttnn::deallocate(v5, false);
  ::std::variant<::ttnn::Tensor, ::std::tuple<::ttnn::Tensor, ::std::tuple<uint32_t, uint32_t>>, ::std::tuple<::ttnn::Tensor, ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>, ::std::tuple<::ttnn::Tensor, ::std::tuple<uint32_t, uint32_t>, ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>> v7 = ttnn::conv2d(v6, v2, v4, 64, 64, 16, 32, 32, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{3, 2}, 1, v3, ::std::nullopt, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v8 = ::std::get<0>(v7);
  ttnn::deallocate(v6, false);
  ttnn::deallocate(v3, false);
  ttnn::deallocate(v2, false);
  ::ttnn::Tensor v9 = ttnn::reshape(v8, ::std::vector<int32_t>{16, 26, 28, 64}, ::std::nullopt);
  ttnn::deallocate(v8, false);
  return v9;
}

std::tuple<::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor> create_inputs_for_dilated_even_conv2d_f32() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({16, 32, 32, 64}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({64, 64, 3, 3}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY});
  ::ttnn::Tensor v5 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v6 = ttnn::to_device(v5, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  return std::make_tuple(v3, v4, v6);
}

std::tuple<::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor> create_inputs_for_dilated_uneven_conv2d_f32() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({16, 32, 32, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({64, 64, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY});
  ::ttnn::Tensor v5 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v6 = ttnn::to_device(v5, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  return std::make_tuple(v3, v4, v6);
}

TEST(EmitC, DilatedConv2dF32) {
  ::ttnn::Tensor v1;
  ::ttnn::Tensor v2;
  ::ttnn::Tensor v3;
  std::tie(v1, v2, v3) = create_inputs_for_dilated_even_conv2d_f32();
  ::ttnn::Tensor v4 = dilated_even_conv2d_f32(v1, v2, v3);
  ::ttnn::Tensor v5;
  ::ttnn::Tensor v6;
  ::ttnn::Tensor v7;
  std::tie(v5, v6, v7) = create_inputs_for_dilated_uneven_conv2d_f32();
  ::ttnn::Tensor v8 = dilated_uneven_conv2d_f32(v5, v6, v7);
}


}  // namespace test
}  // namespace ttnn
