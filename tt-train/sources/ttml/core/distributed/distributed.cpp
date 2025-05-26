// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core/distributed/distributed.hpp"

#include <chrono>
#include <core/ttnn_all_includes.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::core::distributed {

ttnn::Tensor synchronize_tensor(const ttnn::Tensor& tensor) {
    auto* device = &autograd::ctx().get_device();
    auto devices_count = device->get_devices().size();
    assert(devices_count >= 1U);
    // no need to synchronize if there is only one device
    if (devices_count == 1U) {
        return tensor;
    }

    // all_reduce Mean is not supported, use sum and divide by #devices
    auto result = ttnn_fixed::distributed::all_reduce(tensor);
    result = ttnn::multiply(result, 1.0F / static_cast<float>(devices_count));
    return result;
}

void synchronize_parameters(const serialization::NamedParameters& parameters) {
    for (auto& [name, tensor] : parameters) {
        if (tensor->is_grad_initialized()) {
            tensor->set_grad(synchronize_tensor(tensor->get_grad()));
        }
    }
}

void send_all_tensors(const autograd::DistributedContext& ctx, std::vector<ttnn::Tensor>& tensors, Rank dest) {
    std::vector<ttnn::Tensor> cpu_tensors;
    cpu_tensors.reserve(tensors.size());

    std::vector<tt::tt_metal::distributed::multihost::RequestPtr> requests;
    requests.reserve(tensors.size());
    int tag_counter = 0;
    for (auto& tensor : tensors) {
        cpu_tensors.push_back(tensor.cpu());

        auto buffers = ttml::core::get_bytes_from_cpu_tensor(cpu_tensors.back());
        for (auto buffer : buffers) {
            requests.push_back(ctx.isend(buffer, dest, Tag{tag_counter++}));
        }
    }

    for (auto& request : requests) {
        // what should i do with the status?
        [[maybe_unused]] auto status = request->wait();
    }
}

void receive_all_tensors(const autograd::DistributedContext& ctx, std::vector<ttnn::Tensor>& tensors, Rank source) {
    std::vector<ttnn::Tensor> cpu_tensors;
    cpu_tensors.reserve(tensors.size());

    std::vector<tt::tt_metal::distributed::multihost::RequestPtr> requests;
    requests.reserve(tensors.size());
    int tag_counter = 0;
    for (auto& tensor : tensors) {
        cpu_tensors.push_back(tensor.cpu());

        auto buffers = ttml::core::get_bytes_from_cpu_tensor(cpu_tensors.back());
        for (auto buffer : buffers) {
            requests.push_back(ctx.irecv(buffer, source, Tag{tag_counter++}));
        }
    }

    for (auto& request : requests) {
        // what should i do with the status?
        [[maybe_unused]] auto status = request->wait();
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        ttnn::assign(cpu_tensors[i].to_device(tensors[i].device()), tensors[i]);
    }
}

class Timer {
public:
    Timer(const std::string& name) : m_name(name) {
    }

    void start() {
        m_start_time = std::chrono::high_resolution_clock::now();
    }

    void end() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - m_start_time);
        auto duration_ms = static_cast<float>(duration.count());
        total_time += duration_ms;
        num_measurements++;

        fmt::println("[{}] Time: {:.2f} ms, Average: {:.2f} ms", m_name, duration_ms, total_time / num_measurements);
    }

private:
    float total_time{};
    uint32_t num_measurements{};
    string m_name;

    std::chrono::high_resolution_clock::time_point m_start_time;
};

void send_tensor(const autograd::DistributedContext& ctx, const ttnn::Tensor& tensor, Rank dest, Tag tag) {
    // static Timer to_cpu_timer(fmt::format("Rank {}, Send to CPU", *ctx.rank()));
    // static Timer send_buffer_timer(fmt::format("Rank {}, Send buffer", *ctx.rank()));

    // to_cpu_timer.start();
    auto cpu_tensor = tensor.cpu();
    auto buffers = ttml::core::get_bytes_from_cpu_tensor(cpu_tensor);
    // to_cpu_timer.end();

    // send_buffer_timer.start();
    for (auto buffer : buffers) {
        ctx.send(buffer, dest, tag);
    }
    // send_buffer_timer.end();
}

void recv_tensor(const autograd::DistributedContext& ctx, ttnn::Tensor& tensor, Rank source, Tag tag) {
    // static Timer to_cpu_timer(fmt::format("Rank {}, Recv to CPU", *ctx.rank()));
    // static Timer recv_buffer_timer(fmt::format("Rank {}, Recv buffer", *ctx.rank()));
    // static Timer assign_timer(fmt::format("Rank {}, To device", *ctx.rank()));

    // to_cpu_timer.start();
    auto cpu_tensor = tensor.cpu();
    auto buffers = ttml::core::get_bytes_from_cpu_tensor(cpu_tensor);
    // to_cpu_timer.end();

    // recv_buffer_timer.start();
    for (auto buffer : buffers) {
        ctx.recv(buffer, source, tag);
    }
    // recv_buffer_timer.end();

    // assign_timer.start();
    ttnn::assign(cpu_tensor.to_device(tensor.device()), tensor);
    // assign_timer.end();
}

void broadcast_tensor(const autograd::DistributedContext& ctx, ttnn::Tensor& tensor, Rank root) {
    auto cpu_tensor = tensor.cpu();

    auto buffers = ttml::core::get_bytes_from_cpu_tensor(cpu_tensor);

    for (auto buffer : buffers) {
        ctx.broadcast(buffer, root);
    }
    if (ctx.rank() != root) {
        ttnn::assign(cpu_tensor.to_device(tensor.device()), tensor);
    }
}

}  // namespace ttml::core::distributed
