// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "emitc.hpp"

namespace ttnn {
namespace test {

::ttnn::Tensor alexNet(::ttnn::Tensor v1, ::ttnn::Tensor v2) {
    ttnn::IDevice* v3 = ttnn::DeviceGetter::getInstance();
    ::ttnn::Tensor v4 = ttnn::reshape(v1, ::std::vector<int32_t>{1, 1, 1605632, 3}, ::std::nullopt);
    ttnn::deallocate(v1, false);
    ::ttnn::Tensor v5 = ttnn::to_layout(
        v4,
        ::ttnn::Layout::ROW_MAJOR,
        ::std::nullopt,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM},
        v3);
    ttnn::deallocate(v4, false);
    ::std::variant<
        ::ttnn::Tensor,
        ::std::tuple<::ttnn::Tensor, ::std::tuple<uint32_t, uint32_t>>,
        ::std::tuple<::ttnn::Tensor, ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>,
        ::std::tuple<
            ::ttnn::Tensor,
            ::std::tuple<uint32_t, uint32_t>,
            ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>>
        v6 = ttnn::conv2d(
            v5,
            v2,
            v3,
            3,
            64,
            32,
            224,
            224,
            ::std::array<uint32_t, 2>{11, 11},
            ::std::array<uint32_t, 2>{4, 4},
            ::std::array<uint32_t, 2>{0, 0},
            ::std::array<uint32_t, 2>{1, 1},
            1,
            ::std::nullopt,
            ::ttnn::operations::conv::conv2d::Conv2dConfig{
                .dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16},
            ::std::nullopt,
            ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v7 = ::std::get<0>(v6);
    ttnn::deallocate(v5, false);
    ttnn::deallocate(v2, false);
    ::ttnn::Tensor v8 = ttnn::reshape(v7, ::std::vector<int32_t>{32, 54, 54, 64}, ::std::nullopt);
    ttnn::deallocate(v7, false);
    return v8;
}

std::vector<float> load_tensor_from_bin_and_json(const std::string& bin_path, const std::string& json_path) {
    // Load JSON metadata
    std::ifstream json_file(json_path);
    if (!json_file) {
        throw std::runtime_error("Cannot open JSON file: " + json_path);
    }
    json meta;
    json_file >> meta;

    // Extract dtype and shape
    std::string dtype = meta["dtype"];
    if (dtype != "float32") {
        throw std::runtime_error("Only float32 is supported (got dtype: " + dtype + ")");
    }

    std::vector<int> out_shape = meta["shape"].get<std::vector<int>>();

    // Compute total number of elements
    size_t num_elements = 1;
    for (int dim : out_shape) {
        num_elements *= dim;
    }

    // Load binary file
    std::ifstream bin_file(bin_path, std::ios::binary);
    if (!bin_file) {
        throw std::runtime_error("Cannot open binary file: " + bin_path);
    }

    std::vector<float> tensor(num_elements);
    bin_file.read(reinterpret_cast<char*>(tensor.data()), num_elements * sizeof(float));
    if (!bin_file) {
        throw std::runtime_error("Failed to read full tensor data from " + bin_path);
    }

    return tensor;
}

std::tuple<::ttnn::Tensor, ::ttnn::Tensor> create_inputs_for_alexNet() {
    ttnn::IDevice* v1 = ttnn::DeviceGetter::getInstance();
    std::vector<float> v2_data = load_tensor_from_bin_and_json(
        "tests/ttnn/unit_tests/gtests/emitc/I_original.bin", "tests/ttnn/unit_tests/gtests/emitc/I_original.json");
    ::ttnn::TensorSpec tensorSpec2(
        ::ttnn::Shape({32, 224, 224, 3}),
        tt::tt_metal::TensorLayout(
            ::ttnn::DataType::BFLOAT16,
            ::ttnn::Layout::TILE,
            ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM}));
    ::ttnn::Tensor v2 = ::ttnn::Tensor::from_vector(v2_data, tensorSpec2);
    ::ttnn::Tensor v3 = ttnn::to_device(
        v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});

    std::vector<float> v4_data = load_tensor_from_bin_and_json(
        "tests/ttnn/unit_tests/gtests/emitc/weight_original.bin",
        "tests/ttnn/unit_tests/gtests/emitc/weight_original.json");
    ::ttnn::TensorSpec tensorSpec4(
        ::ttnn::Shape({64, 3, 11, 11}),
        tt::tt_metal::TensorLayout(
            ::ttnn::DataType::BFLOAT16,
            ::ttnn::Layout::ROW_MAJOR,
            ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY}));
    ::ttnn::Tensor v4 = ::ttnn::Tensor::from_vector(v4_data, tensorSpec4);

    return std::make_tuple(v3, v4);
}

TEST(EmitC, Sanity) {
    ::ttnn::Tensor v1;
    ::ttnn::Tensor v2;
    std::tie(v1, v2) = create_inputs_for_alexNet();
    ::ttnn::Tensor v3 = alexNet(v1, v2);
    ::ttnn::core::set_printoptions("FULL");
    const std::string tensor_string = v3.write_to_string();
    std::ofstream log_file("tests/ttnn/unit_tests/gtests/emitc/output.txt");
    if (log_file.is_open()) {
        log_file << tensor_string;
        log_file.close();
    }
}

}  // namespace test
}  // namespace ttnn
