#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>

#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>

class ThreadPool {
public:
    explicit ThreadPool(size_t thread_count = std::thread::hardware_concurrency()) : shutdown_(false) {
        workers_.reserve(thread_count);

        for (size_t i = 0; i < thread_count; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this] { return shutdown_ || !tasks_.empty(); });

                        if (shutdown_ && tasks_.empty()) {
                            return;
                        }

                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });

            // cpu_set_t cpuset;
            // CPU_ZERO(&cpuset);
            // CPU_SET(2 * i + 8, &cpuset);

            // int rc = pthread_setaffinity_np(workers_.back().native_handle(), sizeof(cpu_set_t), &cpuset);
            // if (rc != 0) {
            //     std::cerr << "Error setting affinity for thread " << i << ": " << strerror(rc) << std::endl;
            // }
        }
    }

    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            [fn = std::forward<F>(f), ... capturedArgs = std::forward<Args>(args)]() mutable {
                fn(std::move(capturedArgs)...);
            });

        std::future<return_type> result = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            tasks_.emplace([task]() { (*task)(); });
        }
        condition_.notify_one();
        return result;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            shutdown_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_) {
            worker.join();
        }
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool shutdown_;
};
