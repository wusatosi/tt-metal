// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/asio.hpp>
#include <future>
#include <iostream>

#include "tt_metal/common/thread_pool.hpp"

namespace tt::tt_metal {

namespace detail {

class BoostThreadPool : public ThreadPool {
public:
    BoostThreadPool(size_t thread_count) : pool_(thread_count) {}

    ~BoostThreadPool() noexcept override = default;

    void enqueue(std::function<void()>&& f) override { boost::asio::post(pool_, f); }

    void wait() override { pool_.wait(); }

private:
    boost::asio::thread_pool pool_;
};

}  // namespace detail

std::shared_ptr<ThreadPool> create_boost_thread_pool(int num_threads) {
    return std::make_shared<detail::BoostThreadPool>(num_threads);
}

}  // namespace tt::tt_metal
