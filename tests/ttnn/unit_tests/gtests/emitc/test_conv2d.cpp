// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "emitc.hpp"

namespace ttnn {
namespace test {

::ttnn::Tensor conv2d_f32(::ttnn::Tensor v1, ::ttnn::Tensor v2, ::ttnn::Tensor v3) {
    ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
    ::ttnn::Tensor v5 = ttnn::reshape(v1, ::std::vector<int32_t>{1, 1, 16384, 64}, ::std::nullopt);
    ttnn::deallocate(v1, false);
    ::ttnn::Tensor v6 = ttnn::from_device(v5);
    ttnn::deallocate(v5, false);
    ::ttnn::Tensor v7 = ttnn::to_layout(
        v6,
        ::ttnn::Layout::ROW_MAJOR,
        ::std::nullopt,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY},
        static_cast<::ttnn::distributed::MeshDevice*>(nullptr));
    ttnn::deallocate(v6, false);
    ::ttnn::Tensor v8 = ttnn::to_device(
        v7, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v7, false);
    ::std::variant<
        ::ttnn::Tensor,
        ::std::tuple<::ttnn::Tensor, ::std::tuple<uint32_t, uint32_t>>,
        ::std::tuple<::ttnn::Tensor, ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>,
        ::std::tuple<
            ::ttnn::Tensor,
            ::std::tuple<uint32_t, uint32_t>,
            ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>>
        v9 = ttnn::conv2d(
            v8,
            v2,
            v4,
            64,
            64,
            16,
            32,
            32,
            ::std::array<uint32_t, 2>{3, 3},
            ::std::array<uint32_t, 2>{1, 1},
            ::std::array<uint32_t, 2>{0, 0},
            ::std::array<uint32_t, 2>{1, 1},
            1,
            v3,
            ::std::nullopt,
            ::std::nullopt,
            ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
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
    ::ttnn::Tensor v2 = ttnn::ones(
        ::ttnn::Shape({16, 32, 32, 64}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::TILE,
        ::std::nullopt,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v3 = ttnn::to_device(
        v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v4 = ttnn::ones(
        ::ttnn::Shape({64, 64, 3, 3}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::ROW_MAJOR,
        ::std::nullopt,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY});
    ::ttnn::Tensor v5 = ttnn::ones(
        ::ttnn::Shape({1, 1, 1, 64}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::TILE,
        ::std::nullopt,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v6 = ttnn::to_device(
        v5, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
    return std::make_tuple(v3, v4, v6);
}

TEST(EmitC, Conv2d) {
    ::ttnn::Tensor v1;
    ::ttnn::Tensor v2;
    ::ttnn::Tensor v3;
    std::tie(v1, v2, v3) = create_inputs_for_conv2d_f32();
    ::ttnn::Tensor v4 = conv2d_f32(v1, v2, v3);

    ASSERT_EQ(v4.get_dtype(), ::ttnn::DataType::FLOAT32);
}

}  // namespace test
}  // namespace ttnn
