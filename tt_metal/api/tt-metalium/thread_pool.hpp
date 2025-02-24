// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <thread>
#include <mutex>
#include <semaphore>
#include <functional>
#include <memory>

#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>

class WorkerQueue {
private:
    struct Node {
        std::packaged_task<void()> data;
        Node* next = nullptr;
    };

    std::atomic<Node*> head;
    std::atomic<Node*> tail;

    Node* pop_head() {
        Node* oldHead = head.load();
        if (oldHead == tail.load()) {
            return nullptr;  // Queue is empty
        }
        head.store(oldHead->next);
        return oldHead;
    }
    // Statically allocated ring buffer containing
    // node objects, which contain handles to data
    // and another node object to traverse ring buffer.
    const static uint32_t ring_buffer_size = 32768;
    Node ring_buffer[ring_buffer_size];

public:
    WorkerQueue() {
        // Initialize ring buffer for traversal. Each node points to the subsequent node, except for the last one, which
        // points to the head.
        for (int node_idx = 0; node_idx < ring_buffer_size; node_idx++) {
            (node_idx < ring_buffer_size - 1) ? ring_buffer[node_idx].next = (&ring_buffer[node_idx + 1])
                                              : ring_buffer[node_idx].next = &(ring_buffer[0]);
        }
        // Initialize head and tail ptrs to start of ring buffer.
        this->head = ring_buffer;
        this->tail = ring_buffer;
    }
    void push(std::packaged_task<void()>&& task) {
        // Stall condition: this push will update the tail (wptr)
        // to match the location of head (rptr). The current push can
        // thus overwrite data that's being read. Stall until head
        // has progressed (data has been read).
        while (tail.load()->next == head.load());
        tail.load()->data = std::move(task);
        tail.store(tail.load()->next);
    }

    std::packaged_task<void()>&& pop() {
        Node* oldHead = pop_head();
        return std::move(oldHead->data);
    }

    void clear() {
        while (!empty()) {
            void(pop());
        }
    }

    bool empty() const { return head.load() == tail.load(); }
};

class ThreadPool {
public:
    explicit ThreadPool(size_t thread_count = std::thread::hardware_concurrency()) : shutdown_(false) {
        workers_.reserve(thread_count);

        for (size_t i = 0; i < thread_count; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::packaged_task<void()> task;  // Changed type
                    {
                        task_semaphore.acquire();
                        if (shutdown_) {
                            return;
                        }
                        std::unique_lock<std::mutex> lock(mutex_);
                        task = std::move(tasks_.pop());  // Move the packaged_task
                    }
                    task();  // Execute the packaged_task
                    counter--;
                }
            });
            // cpu_set_t cpuset;
            // CPU_ZERO(&cpuset);
            // CPU_SET(i + 8, &cpuset);
            // int rc = pthread_setaffinity_np(workers_.back().native_handle(), sizeof(cpu_set_t), &cpuset);
            // if (rc != 0) {
            //     std::cerr << "Error setting affinity for thread " << i << ": " << strerror(rc) << std::endl;
            // }
        }
    }

    template <class F>
    void enqueue(F&& f) {
        auto task = std::packaged_task<void()>(std::forward<F>(f));
        tasks_.push(std::move(task));  // Move the task directly into queue
        task_semaphore.release();
        counter++;
    }

    void barrier() const noexcept { while (counter); }

    std::size_t num_threads() const noexcept { return workers_.size(); }

    ~ThreadPool() {
        shutdown_ = true;
        for (size_t i = 0; i < workers_.size(); ++i) {
            task_semaphore.release();
        }
        for (std::thread& worker : workers_) {
            worker.join();
        }
    }

private:
    std::vector<std::thread> workers_;
    WorkerQueue tasks_;
    std::mutex mutex_;
    std::counting_semaphore<> task_semaphore{0};
    std::atomic<int> counter = 0;
    bool shutdown_;
};
