// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/asio.hpp>
#include <future>
#include <iostream>
#include <numa.h>
#include <semaphore>

#include "tt_metal/common/thread_pool.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal {

std::unordered_map<int, std::vector<uint32_t>> get_cpu_cores_per_numa_node() {
    std::unordered_map<int, std::vector<uint32_t>> cpu_cores_per_numa_node = {};
    for (int cpu = 0; cpu < numa_num_configured_cpus(); ++cpu) {
        int node = numa_node_of_cpu(cpu);
        cpu_cores_per_numa_node[node].push_back(cpu);
    }
    return cpu_cores_per_numa_node;
}

namespace thread_pool_impls {
// Boost backed thread-pool.
class BoostThreadPool : public ThreadPool {
public:
    BoostThreadPool(size_t thread_count) : pool_(thread_count) {
        // Given the current use case, we don't expect to
        // enqueue more tasks than the number of threads.
        // Add a factor of safety and modify as needed.
        futures_.reserve(thread_count * 4);
    }

    ~BoostThreadPool() noexcept override = default;

    void enqueue(std::function<void()>&& f, uint32_t thread_idx = 0) override {
        std::packaged_task<void()> task(std::move(f));
        futures_.push_back(task.get_future());
        boost::asio::post(pool_, [executor = std::move(task)]() mutable { executor(); });
    }

    void wait() override {
        for (auto& future : futures_) {
            future.get();
        }
        futures_.clear();
    }

private:
    boost::asio::thread_pool pool_;
    std::vector<std::future<void>> futures_;
};

// Custom Thread-Pool using the a vector of boost threads.
class DeviceBoundThreadPool : public ThreadPool {
public:
    DeviceBoundThreadPool(uint32_t thread_count, uint32_t logical_cpu_offset) {
        workers_.reserve(thread_count);
        for (uint32_t i = 0; i < thread_count; i++) {
            workers_.emplace_back(std::make_unique<BoostThreadPool>(1));
        }
    }

    void enqueue(std::function<void()>&& f, uint32_t thread_idx = 0) override {
        workers_[thread_idx]->enqueue(std::move(f));
    }

    void wait() override {
        for (auto& worker : workers_) {
            worker->wait();
        }
    }

private:
    std::vector<std::unique_ptr<BoostThreadPool>> workers_;
};

}  // namespace thread_pool_impls

std::shared_ptr<ThreadPool> create_boost_thread_pool(int num_threads) {
    return std::make_shared<thread_pool_impls::BoostThreadPool>(num_threads);
}

std::shared_ptr<ThreadPool> create_device_bound_thread_pool(int num_threads, uint32_t logical_cpu_offset) {
    return std::make_shared<thread_pool_impls::DeviceBoundThreadPool>(num_threads, logical_cpu_offset);
}

}  // namespace tt::tt_metal
