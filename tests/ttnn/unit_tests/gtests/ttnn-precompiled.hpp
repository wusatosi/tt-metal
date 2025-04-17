// // SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
// //
// // SPDX-License-Identifier: Apache-2.0

// #ifndef TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP
// #define TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP

// // ANCHOR: standalone_includes
// // #include "core.hpp"
// // #include "device.hpp"
// // #include "operations/ccl/all_gather/all_gather.hpp"
// // #include "operations/ccl/ccl_host_types.hpp"
// // #include "operations/ccl/mesh_shard_impl.h"
// // #include "operations/ccl/reduce_scatter/reduce_scatter.hpp"
// // #include "operations/conv/conv2d/conv2d.hpp"
// // #include "operations/conv/conv2d/prepare_conv2d_weights.cpp"
// // #include "operations/copy.hpp"
// // #include "operations/core/core.hpp"
// // #include "operations/creation.hpp"
// // #include "operations/data_movement/concat/concat.hpp"
// // #include "operations/data_movement/permute/permute.hpp"
// // #include "operations/data_movement/repeat/repeat.hpp"
// // #include "operations/data_movement/repeat_interleave/repeat_interleave.hpp"
// // #include "operations/data_movement/slice/slice.hpp"
// #include "ttnn/core.hpp"
// #include "ttnn/device.hpp"
// #include "ttnn/operations/ccl/all_gather/all_gather.hpp"
// #include "ttnn/operations/ccl/ccl_host_types.hpp"
// // #include "ttnn/operations/ccl/mesh_shard_impl.h"
// #include "ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp"
// #include "ttnn/operations/conv/conv2d/conv2d.hpp"
// #include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp"
// #include "ttnn/operations/copy.hpp"
// #include "ttnn/operations/core/core.hpp"
// #include "ttnn/operations/creation.hpp"
// #include "ttnn/operations/data_movement/concat/concat.hpp"
// #include "ttnn/operations/data_movement/permute/permute.hpp"
// #include "ttnn/operations/data_movement/repeat/repeat.hpp"
// #include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp"
// #include "ttnn/operations/data_movement/slice/slice.hpp"
// #include "ttnn/operations/data_movement/transpose/transpose.hpp"
// // #include "operations/eltwise/binary/binary.hpp"
// // #include "operations/eltwise/binary/binary_composite.hpp"
// // #include "operations/eltwise/quantization/quantization.hpp"
// // #include "operations/eltwise/unary/unary_composite.hpp"
// // #include "operations/embedding/embedding.hpp"
// // #include "operations/embedding_backward/embedding_backward.hpp"
// // #include "operations/matmul/matmul.hpp"
// // #include "operations/moreh/moreh_cumsum/moreh_cumsum.hpp"
// // #include "operations/normalization/softmax/softmax.hpp"
// // #include "operations/pool/generic/generic_pools.hpp"
// // #include "operations/pool/upsample/upsample.hpp"
// // #include "operations/reduction/argmax/argmax.hpp"
// // #include "operations/reduction/generic/generic_reductions.hpp"
// // #include "operations/reduction/prod/prod.hpp"
// // #include "tensor/tensor.hpp"
// // #include "tensor/types.hpp"
// #include "ttnn/operations/eltwise/binary/binary.hpp"
// #include "ttnn/operations/eltwise/binary/binary_composite.hpp"
// #include "ttnn/operations/eltwise/quantization/quantization.hpp"
// #include "ttnn/operations/eltwise/unary/unary_composite.hpp"
// #include "ttnn/operations/embedding/embedding.hpp"
// #include "ttnn/operations/embedding_backward/embedding_backward.hpp"
// #include "ttnn/operations/matmul/matmul.hpp"
// #include "ttnn/operations/moreh/moreh_cumsum/moreh_cumsum.hpp"
// #include "ttnn/operations/normalization/softmax/softmax.hpp"
// #include "ttnn/operations/pool/generic/generic_pools.hpp"
// #include "ttnn/operations/pool/upsample/upsample.hpp"
// #include "ttnn/operations/reduction/argmax/argmax.hpp"
// #include "ttnn/operations/reduction/generic/generic_reductions.hpp"
// #include "ttnn/operations/reduction/prod/prod.hpp"
// #include "ttnn/tensor/tensor.hpp"
// #include "ttnn/tensor/types.hpp"
// // #include "types.hpp"
// // ANCHOR_END: standalone_includes

// #include <cassert>
// #include <cstddef>
// #include <iostream>
// #include <vector>

// namespace ttnn {

// // DeviceGetter class
// //
// // Singleton implementation for Device
// //
// class DeviceGetter {
// public:
//   static ttnn::IDevice *getInstance() {
//     static ttnn::IDevice *instance = &ttnn::open_device(0, 1 << 15);
//     // instance->enable_program_cache();

//     return instance;
//   }

// private:
//   ~DeviceGetter() { ttnn::close_device(*device); }

// public:
//   DeviceGetter(DeviceGetter const &) = delete;
//   void operator=(DeviceGetter const &) = delete;

//   ttnn::IDevice *device;
// };

// } // namespace ttnn

// #endif // TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP
