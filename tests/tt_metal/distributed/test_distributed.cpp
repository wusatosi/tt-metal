// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <random>

#include "tests/tt_metal/tt_metal/dispatch/dispatch_test_utils.hpp"
#include "tt_metal/distributed/mesh_device.hpp"
#include "tt_metal/distributed/mesh_device_view.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/distributed/mesh_workload.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/bfloat16.hpp"
namespace tt::tt_metal::distributed::test {

static inline void skip_test_if_not_t3000() {
    auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    const auto arch = tt::Cluster::instance().arch();
    const size_t num_devices = tt::Cluster::instance().number_of_devices();

    if (slow_dispatch) {
        GTEST_SKIP() << "Skipping Multi-Device test suite, since it can only be run in Fast Dispatch Mode.";
    }
    if (num_devices < 8 or arch != tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping T3K Multi-Device test suite on non T3K machine.";
    }
}
class MeshDevice_T3000 : public ::testing::Test {
protected:
    void SetUp() override {
        skip_test_if_not_t3000();
        this->mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));
    }

    void TearDown() override {
        mesh_device_->close_devices();
        mesh_device_.reset();
    }
    std::shared_ptr<MeshDevice> mesh_device_;
};

class MeshDevice_N300 : public ::testing::Test {
protected:
    void SetUp() override { this->mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 2))); }

    void TearDown() override {
        mesh_device_->close_devices();
        mesh_device_.reset();
    }
    std::shared_ptr<MeshDevice> mesh_device_;
};

struct CBConfig {
    uint32_t cb_id;
    uint32_t num_pages;
    uint32_t page_size;
    tt::DataFormat data_format;
};

std::vector<std::shared_ptr<Program>> create_random_programs(
    uint32_t num_programs,
    CoreCoord worker_grid_size,
    uint32_t seed,
    const std::unordered_set<CoreCoord>& active_eth_cores = {}) {
    uint32_t MAX_LOOP = 100;
    uint32_t page_size = 1024;
    uint32_t max_eth_cores = 3;

    uint32_t BRISC_OUTER_LOOP, BRISC_MIDDLE_LOOP, BRISC_INNER_LOOP, NUM_CBS, NUM_SEMS;
    uint32_t NCRISC_OUTER_LOOP, NCRISC_MIDDLE_LOOP, NCRISC_INNER_LOOP;
    uint32_t TRISC_OUTER_LOOP, TRISC_MIDDLE_LOOP, TRISC_INNER_LOOP;
    uint32_t ERISC_OUTER_LOOP, ERISC_MIDDLE_LOOP, ERISC_INNER_LOOP;
    bool USE_MAX_RT_ARGS;

    CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    CoreRangeSet cr_set(cr);

    log_info(tt::LogTest, "Creating {} programs now.", num_programs);
    std::vector<std::shared_ptr<Program>> programs;

    std::map<string, string> data_movement_defines = {{"DATA_MOVEMENT", "1"}};
    std::map<string, string> compute_defines = {{"COMPUTE", "1"}};
    std::map<string, string> erisc_defines = {{"ERISC", "1"}};

    for (uint32_t i = 0; i < num_programs; i++) {
        programs.push_back(std::make_shared<Program>());
        Program& program = *(programs.back());
        // ========== Set configs for BRISC ==========
        if (i == 0) {
            // Ensures that we get at least one compilation with the max amount to
            // ensure it compiles and runs
            BRISC_OUTER_LOOP = MAX_LOOP;
            BRISC_MIDDLE_LOOP = MAX_LOOP;
            BRISC_INNER_LOOP = MAX_LOOP;
            NUM_CBS = NUM_CIRCULAR_BUFFERS;
            NUM_SEMS = NUM_SEMAPHORES;
            USE_MAX_RT_ARGS = true;
        } else {
            BRISC_OUTER_LOOP = rand() % (MAX_LOOP) + 1;
            BRISC_MIDDLE_LOOP = rand() % (MAX_LOOP) + 1;
            BRISC_INNER_LOOP = rand() % (MAX_LOOP) + 1;
            NUM_CBS = rand() % (NUM_CIRCULAR_BUFFERS) + 1;
            NUM_SEMS = rand() % (NUM_SEMAPHORES) + 1;
            USE_MAX_RT_ARGS = false;
        }
        // Create CBs
        for (uint32_t j = 0; j < NUM_CBS; j++) {
            CircularBufferConfig cb_config = CircularBufferConfig(page_size * (j + 1), {{j, tt::DataFormat::Float16_b}})
                                                 .set_page_size(j, page_size * (j + 1));
            auto cb = CreateCircularBuffer(program, cr_set, cb_config);
        }

        // Create Semaphores
        for (uint32_t j = 0; j < NUM_SEMS; j++) {
            CreateSemaphore(program, cr_set, j + 1);
            uint32_t curr_idx = 0;
            if (active_eth_cores.size()) {
                auto active_eth_core = active_eth_cores.begin();
                for (int k = 0; k < max_eth_cores && active_eth_core != active_eth_cores.end();
                     ++i, ++active_eth_core) {
                    CreateSemaphore(program, *active_eth_core, j + 1, CoreType::ETH);
                }
            }
        }

        // Create RTAs
        auto [brisc_unique_rtargs, brisc_common_rtargs] = create_runtime_args(USE_MAX_RT_ARGS);
        uint32_t num_brisc_unique_rtargs = brisc_unique_rtargs.size();
        uint32_t num_brisc_common_rtargs = brisc_common_rtargs.size();
        std::vector<uint32_t> brisc_compile_args = {
            BRISC_OUTER_LOOP,
            BRISC_MIDDLE_LOOP,
            BRISC_INNER_LOOP,
            NUM_CBS,
            NUM_SEMS,
            num_brisc_unique_rtargs,
            num_brisc_common_rtargs,
            page_size};

        // ========== Set configs for NCRISC ==========
        if (i == 0) {
            NCRISC_OUTER_LOOP = MAX_LOOP;
            NCRISC_MIDDLE_LOOP = MAX_LOOP;
            NCRISC_INNER_LOOP = MAX_LOOP;
        } else {
            NCRISC_OUTER_LOOP = rand() % (MAX_LOOP) + 1;
            NCRISC_MIDDLE_LOOP = rand() % (MAX_LOOP) + 1;
            NCRISC_INNER_LOOP = rand() % (MAX_LOOP) + 1;
        }

        auto [ncrisc_unique_rtargs, ncrisc_common_rtargs] = create_runtime_args(USE_MAX_RT_ARGS);
        uint32_t num_ncrisc_unique_rtargs = ncrisc_unique_rtargs.size();
        uint32_t num_ncrisc_common_rtargs = ncrisc_common_rtargs.size();
        std::vector<uint32_t> ncrisc_compile_args = {
            NCRISC_OUTER_LOOP,
            NCRISC_MIDDLE_LOOP,
            NCRISC_INNER_LOOP,
            NUM_CBS,
            NUM_SEMS,
            num_ncrisc_unique_rtargs,
            num_ncrisc_common_rtargs,
            page_size};

        // ========== Set configs for TRISC ==========
        if (i == 0) {
            TRISC_OUTER_LOOP = MAX_LOOP;
            TRISC_MIDDLE_LOOP = MAX_LOOP;
            TRISC_INNER_LOOP = MAX_LOOP;
        } else {
            TRISC_OUTER_LOOP = rand() % (MAX_LOOP) + 1;
            TRISC_MIDDLE_LOOP = rand() % (MAX_LOOP) + 1;
            TRISC_INNER_LOOP = rand() % (MAX_LOOP) + 1;
        }

        auto [trisc_unique_rtargs, trisc_common_rtargs] = create_runtime_args(USE_MAX_RT_ARGS);
        uint32_t num_trisc_unique_rtargs = trisc_unique_rtargs.size();
        uint32_t num_trisc_common_rtargs = trisc_common_rtargs.size();
        std::vector<uint32_t> trisc_compile_args = {
            TRISC_OUTER_LOOP,
            TRISC_MIDDLE_LOOP,
            TRISC_INNER_LOOP,
            NUM_CBS,
            NUM_SEMS,
            num_trisc_unique_rtargs,
            num_trisc_common_rtargs,
            page_size};

        if (i == 0) {
            ERISC_OUTER_LOOP = MAX_LOOP;
            ERISC_MIDDLE_LOOP = MAX_LOOP;
            ERISC_INNER_LOOP = MAX_LOOP;
        } else {
            ERISC_OUTER_LOOP = rand() % (MAX_LOOP) + 1;
            ERISC_MIDDLE_LOOP = rand() % (MAX_LOOP) + 1;
            ERISC_INNER_LOOP = rand() % (MAX_LOOP) + 1;
        }
        // Only setup RTAs on ERISC. No Common RTAs.
        uint32_t max_erisc_rtas = 64;
        uint32_t num_erisc_rtas = rand() % (max_erisc_rtas + 1);
        auto [erisc_unique_rtargs, erisc_common_rtargs] = create_runtime_args(num_erisc_rtas, 0, 0, 0);
        uint32_t num_erisc_unique_rtargs = erisc_unique_rtargs.size();
        uint32_t num_erisc_common_rt_args = erisc_common_rtargs.size();

        std::vector<uint32_t> erisc_compile_time_args = {
            ERISC_OUTER_LOOP,
            ERISC_MIDDLE_LOOP,
            ERISC_INNER_LOOP,
            0, /* CBs are not supported on ERISC cores */
            NUM_SEMS,
            num_erisc_unique_rtargs,
            num_erisc_common_rt_args,
            page_size};

        // Create Kernels
        bool at_least_one_kernel = false;
        if (i == 0 or ((rand() % 2) == 0)) {
            auto dummy_brisc_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                cr_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = brisc_compile_args,
                    .defines = data_movement_defines});
            SetRuntimeArgs(program, dummy_brisc_kernel, cr_set, brisc_unique_rtargs);
            SetCommonRuntimeArgs(program, dummy_brisc_kernel, brisc_common_rtargs);
            at_least_one_kernel = true;
        }

        if (i == 0 or ((rand() % 2) == 0)) {
            auto dummy_ncrisc_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                cr_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = NOC::RISCV_1_default,
                    .compile_args = ncrisc_compile_args,
                    .defines = data_movement_defines});
            SetRuntimeArgs(program, dummy_ncrisc_kernel, cr_set, ncrisc_unique_rtargs);
            SetCommonRuntimeArgs(program, dummy_ncrisc_kernel, ncrisc_common_rtargs);
            at_least_one_kernel = true;
        }

        if (i == 0 or ((rand() % 2) == 0)) {
            auto dummy_trisc_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                cr_set,
                ComputeConfig{
                    .math_approx_mode = false, .compile_args = trisc_compile_args, .defines = compute_defines});
            SetRuntimeArgs(program, dummy_trisc_kernel, cr_set, trisc_unique_rtargs);
            SetCommonRuntimeArgs(program, dummy_trisc_kernel, trisc_common_rtargs);
            at_least_one_kernel = true;
        }

        if (not at_least_one_kernel) {
            uint32_t random_risc = rand() % 3 + 1;
            if (random_risc == 1) {
                auto dummy_brisc_kernel = CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                    cr_set,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                        .compile_args = brisc_compile_args,
                        .defines = data_movement_defines});
                SetRuntimeArgs(program, dummy_brisc_kernel, cr_set, brisc_unique_rtargs);
                SetCommonRuntimeArgs(program, dummy_brisc_kernel, brisc_common_rtargs);
            } else if (random_risc == 2) {
                auto dummy_ncrisc_kernel = CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                    cr_set,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                        .compile_args = ncrisc_compile_args,
                        .defines = data_movement_defines});
                SetRuntimeArgs(program, dummy_ncrisc_kernel, cr_set, ncrisc_unique_rtargs);
                SetCommonRuntimeArgs(program, dummy_ncrisc_kernel, ncrisc_common_rtargs);
            } else if (random_risc == 3) {
                auto dummy_trisc_kernel = CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                    cr_set,
                    ComputeConfig{
                        .math_approx_mode = false, .compile_args = trisc_compile_args, .defines = compute_defines});
                SetRuntimeArgs(program, dummy_trisc_kernel, cr_set, trisc_unique_rtargs);
                SetCommonRuntimeArgs(program, dummy_trisc_kernel, trisc_common_rtargs);
            } else {
                TT_THROW("Invalid");
            }
        }
        if (active_eth_cores.size()) {
            auto active_eth_core = active_eth_cores.begin();
            for (int k = 0; k < max_eth_cores && active_eth_core != active_eth_cores.end(); ++i, ++active_eth_core) {
                auto dummy_erisc_kernel = CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                    *active_eth_core,
                    EthernetConfig{
                        .noc = NOC::NOC_0, .compile_args = erisc_compile_time_args, .defines = erisc_defines});
                SetRuntimeArgs(program, dummy_erisc_kernel, *active_eth_core, erisc_unique_rtargs);
            }
        }
    }
    return programs;
}

TEST_F(MeshDevice_T3000, TestMeshWorkloadOnActiveEth) {
    uint32_t num_workloads = 10;
    auto random_seed = 0;
    uint32_t num_iters = 500;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    std::vector<std::shared_ptr<MeshWorkload>> workloads = {};
    std::vector<std::shared_ptr<Program>> programs = {};

    for (int i = 0; i < num_workloads; i++) {
        std::shared_ptr<MeshWorkload> workload = std::make_shared<MeshWorkload>();
        for (std::size_t logical_x = 0; logical_x < mesh_device_->num_cols(); logical_x++) {
            for (std::size_t logical_y = 0; logical_y < mesh_device_->num_rows(); logical_y++) {
                Device* device = mesh_device_->get_device(logical_y, logical_x);
                auto programs_ = create_random_programs(
                    1, mesh_device_->compute_with_storage_grid_size(), seed, device->get_active_ethernet_cores(true));
                LogicalDeviceRange devices = {{logical_x, logical_y}, {logical_x + 1, logical_y + 1}};
                workload->add_program(devices, *programs_[0]);
            }
        }
        workload->enqueue(mesh_device_, 0, false);
        workloads.push_back(workload);
    }
    for (int i = 0; i < num_iters; i++) {
        if (i % 100 == 0) {
            log_info(tt::LogTest, "Run MeshWorkloads for iteration {}", i);
        }
        for (auto& workload : workloads) {
            workload->enqueue(mesh_device_, 0, false);
        }
    }
    for (auto device : mesh_device_->get_devices()) {
        Finish(device->command_queue(0));
    }
}

TEST_F(MeshDevice_T3000, TestMeshWorkloadOnActiveEthRandomGridSize) {
    uint32_t num_workloads = 30;
    auto random_seed = 0;
    uint32_t num_iters = 500;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    std::vector<std::shared_ptr<MeshWorkload>> workloads = {};
    std::vector<std::shared_ptr<Program>> programs = {};

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> gen_x(1, 4);
    std::uniform_int_distribution<int> gen_y(1, 2);

    for (int i = 0; i < num_workloads; i++) {
        std::shared_ptr<MeshWorkload> workload = std::make_shared<MeshWorkload>();
        uint32_t x_end = gen_x(rng);
        uint32_t y_end = gen_y(rng);
        for (std::size_t logical_x = 0; logical_x < x_end; logical_x++) {
            for (std::size_t logical_y = 0; logical_y < y_end; logical_y++) {
                Device* device = mesh_device_->get_device(logical_y, logical_x);
                auto programs_ = create_random_programs(
                    1, mesh_device_->compute_with_storage_grid_size(), seed, device->get_active_ethernet_cores(true));
                LogicalDeviceRange devices = {{logical_x, logical_y}, {logical_x + 1, logical_y + 1}};
                workload->add_program(devices, *programs_[0]);
            }
        }
        workload->enqueue(mesh_device_, 0, false);
        workloads.push_back(workload);
    }
    for (int i = 0; i < num_iters; i++) {
        if (i % 100 == 0) {
            log_info(tt::LogTest, "Run MeshWorkloads for iteration {}", i);
        }
        for (auto& workload : workloads) {
            workload->enqueue(mesh_device_, 0, false);
        }
    }
    for (auto device : mesh_device_->get_devices()) {
        Finish(device->command_queue(0));
    }
}

TEST_F(MeshDevice_T3000, TestSimultaneousMeshWorkloads) {
    uint32_t num_programs = 100;
    uint32_t num_heterogeneous_programs = 64;
    uint32_t num_iterations = 1000;
    auto random_seed = 0;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    log_info(tt::LogTest, "Using Test Seed: {}", seed);
    srand(seed);

    auto programs = create_random_programs(num_programs, mesh_device_->compute_with_storage_grid_size(), seed);
    std::vector<std::shared_ptr<MeshWorkload>> mesh_workloads = {};

    log_info(tt::LogTest, "Compile and load {} MeshWorkloads", num_programs);
    for (int i = 0; i < num_programs; i += 2) {
        std::shared_ptr<MeshWorkload> random_workload = std::make_shared<MeshWorkload>();
        if (i % 2) {
            LogicalDeviceRange devices_0 = LogicalDeviceRange({0, 0}, {4, 1});
            LogicalDeviceRange devices_1 = LogicalDeviceRange({0, 1}, {4, 2});
            random_workload->add_program(devices_0, *programs[i]);
            random_workload->add_program(devices_1, *programs[i + 1]);
        } else {
            LogicalDeviceRange devices_0 = LogicalDeviceRange({0, 0}, {2, 2});
            LogicalDeviceRange devices_1 = LogicalDeviceRange({2, 0}, {4, 2});
            random_workload->add_program(devices_0, *programs[i]);
            random_workload->add_program(devices_1, *programs[i + 1]);
        }
        random_workload->enqueue(mesh_device_, 0, false);
        mesh_workloads.push_back(random_workload);
    }
    programs = create_random_programs(num_programs, mesh_device_->compute_with_storage_grid_size(), seed);
    for (int i = 0; i < num_programs; i += 4) {
        std::shared_ptr<MeshWorkload> random_workload = std::make_shared<MeshWorkload>();
        LogicalDeviceRange devices_0 = LogicalDeviceRange({0, 0}, {1, 2});
        LogicalDeviceRange devices_1 = LogicalDeviceRange({1, 0}, {2, 2});
        LogicalDeviceRange devices_2 = LogicalDeviceRange({2, 0}, {3, 2});
        LogicalDeviceRange devices_3 = LogicalDeviceRange({3, 0}, {4, 2});

        random_workload->add_program(devices_0, *programs[i]);
        random_workload->add_program(devices_1, *programs[i + 1]);
        random_workload->add_program(devices_2, *programs[i + 2]);
        random_workload->add_program(devices_3, *programs[i + 3]);

        random_workload->enqueue(mesh_device_, 0, false);
        mesh_workloads.push_back(random_workload);
    }
    programs = create_random_programs(num_heterogeneous_programs, mesh_device_->compute_with_storage_grid_size(), seed);
    for (int i = 0; i < num_heterogeneous_programs; i += 8) {
        std::shared_ptr<MeshWorkload> random_workload = std::make_shared<MeshWorkload>();
        LogicalDeviceRange devices_0 = LogicalDeviceRange({0, 0}, {1, 1});
        LogicalDeviceRange devices_1 = LogicalDeviceRange({0, 1}, {1, 2});
        LogicalDeviceRange devices_2 = LogicalDeviceRange({1, 0}, {2, 1});
        LogicalDeviceRange devices_3 = LogicalDeviceRange({1, 1}, {2, 2});
        LogicalDeviceRange devices_4 = LogicalDeviceRange({2, 0}, {3, 1});
        LogicalDeviceRange devices_5 = LogicalDeviceRange({2, 1}, {3, 2});
        LogicalDeviceRange devices_6 = LogicalDeviceRange({3, 0}, {4, 1});
        LogicalDeviceRange devices_7 = LogicalDeviceRange({3, 1}, {4, 2});

        random_workload->add_program(devices_0, *programs[i]);
        random_workload->add_program(devices_1, *programs[i + 1]);
        random_workload->add_program(devices_2, *programs[i + 2]);
        random_workload->add_program(devices_3, *programs[i + 3]);
        random_workload->add_program(devices_4, *programs[i + 4]);
        random_workload->add_program(devices_5, *programs[i + 5]);
        random_workload->add_program(devices_6, *programs[i + 6]);
        random_workload->add_program(devices_7, *programs[i + 7]);
        random_workload->enqueue(mesh_device_, 0, false);
        mesh_workloads.push_back(random_workload);
    }

    for (int i = 0; i < num_iterations; i++) {
        if (i % 100 == 0) {
            log_info(tt::LogTest, "Run MeshWorkloads for iteration {}", i);
        }
        for (auto workload : mesh_workloads) {
            workload->enqueue(mesh_device_, 0, false);
        }
    }
    for (auto device : mesh_device_->get_devices()) {
        Finish(device->command_queue(0));
    }
}

TEST_F(MeshDevice_T3000, TestRandomizedMeshWorkload) {
    uint32_t num_programs = 60;
    uint32_t num_iterations = 1500;
    auto random_seed = 10;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    log_info(tt::LogTest, "Using Test Seed: {}", seed);
    srand(seed);

    auto programs = create_random_programs(num_programs, mesh_device_->compute_with_storage_grid_size(), seed);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> gen_x(1, 4);
    std::uniform_int_distribution<int> gen_y(1, 2);
    std::vector<std::shared_ptr<MeshWorkload>> mesh_workloads = {};

    // Create multiple mesh workloads on grids of random sizes.
    // Compile the workload (lower + send binaries to mesh device here as well)
    log_info(tt::LogTest, "Compile and load {} MeshWorkloads", num_programs);
    for (int i = 0; i < num_programs; i += 1) {
        // Choose a grid of random dimensions and run a MeshWorkload on it
        LogicalDeviceRange device_range = LogicalDeviceRange({0, 0}, {gen_x(rng), gen_y(rng)});
        auto random_workload = std::make_shared<MeshWorkload>();
        random_workload->add_program(device_range, *programs[i]);
        random_workload->enqueue(mesh_device_, 0, false);
        mesh_workloads.push_back(random_workload);
    }
    for (int i = 0; i < num_iterations; i++) {
        if (i % 100 == 0) {
            log_info(tt::LogTest, "Run MeshWorkloads for iteration {}", i);
        }
        for (auto workload : mesh_workloads) {
            workload->enqueue(mesh_device_, 0, false);
        }
    }
    log_info(tt::LogTest, "Calling Finish");
    for (auto device : mesh_device_->get_devices()) {
        Finish(device->command_queue(0));
    }
}

std::vector<CBHandle> initialize_dummy_circular_buffers(
    Program& program, const CoreRangeSet& cr_set, const std::vector<CBConfig>& cb_configs) {
    std::vector<CBHandle> cb_handles;
    for (uint32_t i = 0; i < cb_configs.size(); i++) {
        const CBConfig& cb_config = cb_configs[i];
        const uint32_t cb_id = cb_config.cb_id;
        const uint32_t cb_num_pages = cb_config.num_pages;
        const uint32_t page_size = cb_config.page_size;
        const uint32_t cb_size = cb_num_pages * page_size;
        const tt::DataFormat data_format = cb_config.data_format;
        const CircularBufferConfig circular_buffer_config =
            CircularBufferConfig(cb_size, {{cb_id, data_format}}).set_page_size(cb_id, page_size);
        const CBHandle cb_handle = CreateCircularBuffer(program, cr_set, circular_buffer_config);
        cb_handles.push_back(cb_handle);
    }
    return cb_handles;
}

void initialize_dummy_kernels(Program& program, const CoreRangeSet& cr_set) {
    auto dummy_reader_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/blank.cpp",
        cr_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto dummy_writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/blank.cpp",
        cr_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto dummy_compute_kernel = CreateKernel(program, "tt_metal/kernels/compute/blank.cpp", cr_set, ComputeConfig{});
}

std::shared_ptr<Program> initialize_dummy_program(CoreCoord worker_grid_size) {
    std::shared_ptr<Program> program = std::make_shared<Program>();
    CoreRange cr = CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    CoreRangeSet cr_set({cr});

    CBConfig cb_config_0 = {.cb_id = 0, .num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_1 = {.cb_id = 1, .num_pages = 2, .page_size = 4096, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_2 = {.cb_id = 2, .num_pages = 2, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_3 = {.cb_id = 3, .num_pages = 4, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    std::vector<CBConfig> cb_config_vector = {cb_config_0, cb_config_1, cb_config_2, cb_config_3};

    initialize_dummy_kernels(*program, cr_set);
    initialize_dummy_circular_buffers(*program, cr_set, cb_config_vector);
    return program;
}

std::vector<std::shared_ptr<Program>> create_eltwise_bin_programs(
    std::shared_ptr<MeshDevice> mesh_device,
    std::vector<std::shared_ptr<Buffer>>& src0_bufs,
    std::vector<std::shared_ptr<Buffer>>& src1_bufs,
    std::vector<std::shared_ptr<Buffer>>& output_bufs) {
    const std::vector<std::string> op_id_to_op_define = {"add_tiles", "mul_tiles"};
    const std::vector<std::string> op_id_to_op_type_define = {"EltwiseBinaryType::ELWADD", "EltwiseBinaryType::ELWMUL"};

    CoreCoord worker_grid_size = mesh_device->compute_with_storage_grid_size();

    std::vector<std::shared_ptr<Program>> programs = {std::make_shared<Program>(), std::make_shared<Program>()};
    auto full_grid = CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});

    for (std::size_t eltwise_op = 0; eltwise_op < op_id_to_op_define.size(); eltwise_op++) {
        auto& program = *programs[eltwise_op];
        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 2048;
        uint32_t dram_buffer_size =
            single_tile_size * num_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        uint32_t page_size = single_tile_size;

        for (auto device : mesh_device->get_devices()) {
            tt_metal::InterleavedBufferConfig dram_config{
                .device = device,
                .size = dram_buffer_size,
                .page_size = page_size,
                .buffer_type = tt_metal::BufferType::DRAM};
            for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
                for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                    auto src0_dram_buffer = CreateBuffer(dram_config);
                    src0_bufs.push_back(src0_dram_buffer);

                    auto src1_dram_buffer = CreateBuffer(dram_config);
                    src1_bufs.push_back(src1_dram_buffer);

                    auto dst_dram_buffer = CreateBuffer(dram_config);
                    output_bufs.push_back(dst_dram_buffer);
                }
            }
        }

        uint32_t src0_cb_index = tt::CBIndex::c_0;
        uint32_t num_input_tiles = 2;
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, full_grid, cb_src0_config);

        uint32_t src1_cb_index = tt::CBIndex::c_1;
        tt_metal::CircularBufferConfig cb_src1_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src1_cb_index, single_tile_size);
        auto cb_src1 = tt_metal::CreateCircularBuffer(program, full_grid, cb_src1_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        uint32_t num_output_tiles = 2;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, full_grid, cb_output_config);

        auto binary_reader_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dual_8bank.cpp",
            full_grid,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
            full_grid,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        std::vector<uint32_t> compute_kernel_args = {};

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        std::map<string, string> binary_defines = {
            {"ELTWISE_OP", op_id_to_op_define[eltwise_op]}, {"ELTWISE_OP_TYPE", op_id_to_op_type_define[eltwise_op]}};
        auto eltwise_binary_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/compute/eltwise_binary.cpp",
            full_grid,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = binary_defines});

        SetRuntimeArgs(program, eltwise_binary_kernel, full_grid, {2048, 1});

        for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
            for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                CoreCoord curr_core = {col_idx, row_idx};
                const std::array<uint32_t, 7> reader_args = {
                    src0_bufs.at(col_idx * worker_grid_size.y + row_idx)->address(),
                    0,
                    num_tiles,
                    src1_bufs.at(col_idx * worker_grid_size.y + row_idx)->address(),
                    0,
                    num_tiles,
                    0};

                const std::array<uint32_t, 3> writer_args = {
                    output_bufs.at(col_idx * worker_grid_size.y + row_idx)->address(), 0, num_tiles};

                SetRuntimeArgs(program, unary_writer_kernel, curr_core, writer_args);
                SetRuntimeArgs(program, binary_reader_kernel, curr_core, reader_args);
            }
        }
    }
    return programs;
}
TEST_F(MeshDevice_T3000, TestEltwiseBinaryMeshWorkload) {
    std::vector<std::shared_ptr<Buffer>> src0_bufs = {};
    std::vector<std::shared_ptr<Buffer>> src1_bufs = {};
    std::vector<std::shared_ptr<Buffer>> output_bufs = {};

    CoreCoord worker_grid_size = mesh_device_->compute_with_storage_grid_size();

    auto programs = create_eltwise_bin_programs(mesh_device_, src0_bufs, src1_bufs, output_bufs);
    auto mesh_workload = MeshWorkload();
    LogicalDeviceRange devices_0 = LogicalDeviceRange({0, 0}, {4, 1});
    LogicalDeviceRange devices_1 = LogicalDeviceRange({0, 1}, {4, 2});

    mesh_workload.add_program(devices_0, *programs[0]);
    mesh_workload.add_program(devices_1, *programs[1]);
    std::vector<uint32_t> src0_vec = create_constant_vector_of_bfloat16(src0_bufs[0]->size(), 2);
    std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(src0_bufs[0]->size(), 3);

    uint32_t buffer_idx = 0;
    for (auto device : mesh_device_->get_devices()) {
        for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
            for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                EnqueueWriteBuffer(device->command_queue(), src0_bufs.at(buffer_idx), src0_vec, false);
                EnqueueWriteBuffer(device->command_queue(), src1_bufs.at(buffer_idx), src1_vec, false);
                buffer_idx++;
            }
        }
    }
    // Run workload multiple times
    for (int i = 0; i < 1000; i++) {
        mesh_workload.enqueue(mesh_device_, 0, false);
    }

    buffer_idx = 0;
    uint32_t dev_idx = 0;
    for (auto device : mesh_device_->get_devices()) {
        for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
            for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                std::vector<bfloat16> dst_vec = {};
                EnqueueReadBuffer(device->command_queue(), output_bufs.at(buffer_idx), dst_vec, true);
                if (dev_idx < 4) {
                    for (int i = 0; i < dst_vec.size(); i++) {
                        EXPECT_EQ(dst_vec[i].to_float(), 5);
                    }
                } else {
                    for (int i = 0; i < dst_vec.size(); i++) {
                        EXPECT_EQ(dst_vec[i].to_float(), 6);
                    }
                }
                buffer_idx++;
            }
        }
        dev_idx++;
    }
}

TEST_F(MeshDevice_N300, TestHomogenousMeshWorkload) {
    CoreCoord worker_grid_size = mesh_device_->compute_with_storage_grid_size();
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::Float16_b);

    uint32_t num_tiles = 1;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    // Create buffers
    std::vector<std::shared_ptr<Buffer>> input_buffers = {};
    std::vector<std::shared_ptr<Buffer>> output_buffers = {};
    for (auto device : mesh_device_->get_devices()) {
        InterleavedBufferConfig dram_config{
            .device = device, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};

        for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
            for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                input_buffers.push_back(CreateBuffer(dram_config));
                output_buffers.push_back(CreateBuffer(dram_config));
            }
        }
    }
    // Create MeshWorkload
    Program program = CreateProgram();
    auto full_grid = CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    auto reader_writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/full_grid_eltwise_device_reuse.cpp",
        full_grid,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto sem_scaling_factor = 2;
    auto scaling_sem_idx = CreateSemaphore(program, full_grid, sem_scaling_factor);
    uint32_t scaling_height_toggle = 16;
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(dram_buffer_size, {{src0_cb_index, DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    uint32_t add_factor = 64;
    for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
        for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
            CoreCoord curr_core = {col_idx, row_idx};
            SetRuntimeArgs(
                program,
                reader_writer_kernel,
                curr_core,
                {input_buffers.at(col_idx * worker_grid_size.y + row_idx)->address(),
                 output_buffers.at(col_idx * worker_grid_size.y + row_idx)->address(),
                 0, /* src_bank_id */
                 0, /* dst_bank_id */
                 add_factor,
                 constants::TILE_HEIGHT,
                 constants::TILE_WIDTH,
                 scaling_sem_idx,
                 scaling_height_toggle});
            CBHandle cb_src0 = CreateCircularBuffer(program, curr_core, cb_src0_config);
        }
    }
    auto program_1 = initialize_dummy_program(worker_grid_size);
    auto mesh_workload = MeshWorkload();
    LogicalDeviceRange devices_0 = LogicalDeviceRange({0, 0}, {1, 1});
    LogicalDeviceRange devices_1 = LogicalDeviceRange({1, 0}, {2, 1});
    mesh_workload.add_program(devices_0, program);
    mesh_workload.add_program(devices_1, *program_1);

    std::size_t buffer_idx = 0;
    std::vector<uint32_t> src_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 1);
    for (auto device : mesh_device_->get_devices()) {
        for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
            for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                EnqueueWriteBuffer(device->command_queue(), input_buffers.at(buffer_idx), src_vec, false);
                buffer_idx++;
            }
        }
    }

    mesh_workload.enqueue(mesh_device_, 0, false);

    buffer_idx = 0;
    for (auto device : mesh_device_->get_devices()) {
        for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
            for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                std::vector<bfloat16> dst_vec = {};
                EnqueueReadBuffer(device->command_queue(), output_buffers.at(buffer_idx), dst_vec, true);
                buffer_idx++;
                if (device->id() == 4) {
                    for (int i = 0; i < dst_vec.size(); i++) {
                        float ref_val = std::pow(2, 1);
                        if (i >= 512) {
                            ref_val = std::pow(2, 2);
                        }
                        EXPECT_EQ(dst_vec[i].to_float(), ref_val);
                    }
                }
            }
        }
    }
}

TEST_F(MeshDevice_T3000, SimpleMeshDeviceTest) {
    EXPECT_EQ(mesh_device_->num_devices(), 8);
    EXPECT_EQ(mesh_device_->num_rows(), 2);
    EXPECT_EQ(mesh_device_->num_cols(), 4);
}

TEST(MeshDeviceSuite, Test1x1SystemMeshInitialize) {
    auto& sys = tt::tt_metal::distributed::SystemMesh::instance();

    auto config =
        tt::tt_metal::distributed::MeshDeviceConfig(MeshShape(1, 1), MeshOffset(0, 0), {}, MeshType::RowMajor);

    EXPECT_NO_THROW({
        auto mesh = tt::tt_metal::distributed::MeshDevice::create(
            config, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, tt::tt_metal::DispatchCoreType::WORKER);
        mesh->close_devices();
    });
}

}  // namespace tt::tt_metal::distributed::test
