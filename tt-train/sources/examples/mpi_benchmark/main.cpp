// mpi_send_recv_benchmark.cpp
// Improved MPI send/receive micro‑benchmark using Google Benchmark
// Compile with: mpicxx -O3 -std=c++20 -lbenchmark -lpthread -o mpi_bench mpi_send_recv_benchmark.cpp

#include <benchmark/benchmark.h>
#include <fmt/core.h>
#include <mpi.h>

#include <chrono>
#include <vector>

#include "autograd/auto_context.hpp"

namespace {
using Rank = tt::tt_metal::distributed::multihost::Rank;
using Tag = tt::tt_metal::distributed::multihost::Tag;

// User‑defined literals for convenient byte sizes
constexpr std::size_t operator"" _KiB(unsigned long long k) {
    return k * 1024ULL;
}
constexpr std::size_t operator"" _MiB(unsigned long long m) {
    return m * 1024_KiB;
}

/// Silent reporter used by non‑root ranks to suppress redundant console output
class SilentReporter final : public benchmark::BenchmarkReporter {
public:
    bool ReportContext(const Context &) noexcept override {
        return true;
    }
    void ReportRuns(const std::vector<Run> &) noexcept override {
    }
    void Finalize() noexcept override {
    }
};

//-----------------------------------------------------------------------------//
// Core benchmark: round‑trip ping–pong between rank 0 and rank 1
//-----------------------------------------------------------------------------//

template <std::size_t Bytes>
static void SendRecv(benchmark::State &state) {
    auto &dist_ctx = ttml::autograd::ctx().get_distributed_context();
    const int world_rank = *dist_ctx.rank();

    // Synchronise ranks before measurement starts
    dist_ctx.barrier();

    std::vector<std::byte> buffer(Bytes);

    for (auto _ : state) {
        const auto t0 = std::chrono::steady_clock::now();

        if (world_rank == 0) {
            dist_ctx.send(buffer, Rank{1}, Tag{0});
        } else if (world_rank == 1) {
            dist_ctx.recv(buffer, Rank{0}, Tag{0});
        }

        const double elapsed_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        // Gather the slowest time to ensure consistent statistics
        double max_sec{};
        MPI_Allreduce(&elapsed_sec, &max_sec, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_sec);

        // Add a bandwidth counter (one‑way)
        const double gbytes = static_cast<double>(Bytes) / (1024.0 * 1024.0 * 1024.0);
        state.counters["GB/s"] = benchmark::Counter(
            gbytes / max_sec, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
    }

    state.SetBytesProcessed(static_cast<int64_t>(Bytes) * state.iterations());
}

// Register benchmark instances
BENCHMARK_TEMPLATE(SendRecv, 1_MiB)->UseManualTime();
BENCHMARK_TEMPLATE(SendRecv, 25_MiB)->UseManualTime();

}  // namespace

int main(int argc, char *argv[]) {
    auto &ctx = ttml::autograd::ctx();
    ctx.initialize_distributed_context(argc, argv);
    auto &dist_ctx = ctx.get_distributed_context();

    // Abort early if the world is too small
    if (*dist_ctx.size() < 2) {
        if (*dist_ctx.rank() == 0)
            fmt::print(stderr, "[warning] Need at least 2 MPI ranks to run benchmark.\n");
        return 0;
    }

    benchmark::Initialize(&argc, argv);

    const bool is_root = (*dist_ctx.rank() == 0);
    if (is_root) {
        benchmark::RunSpecifiedBenchmarks();
    } else {
        SilentReporter silent;
        benchmark::RunSpecifiedBenchmarks(&silent);
    }

    return 0;
}
