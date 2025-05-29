// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/core.h>

#include <CLI/CLI.hpp>

#include "autograd/module_base.hpp"
#include "common.hpp"
#include "core/distributed/distributed.hpp"
#include "core/tt_tensor_utils.hpp"
#include "datasets/utils.hpp"
#include "models/distributed/gpt2.hpp"
#include "models/gpt2.hpp"
#include "tokenizers/bpe_tokenizer.hpp"
#include "tokenizers/char_tokenizer.hpp"

using SortedParameters = std::map<std::string, ttml::autograd::TensorPtr>;
// using three_tier_arch::SortedParameters;

using Rank = ttml::core::distributed::Rank;
using Tag = ttml::core::distributed::Tag;

void send_aggregated_gradients_from_workers_to_optimizer(
    const ttml::autograd::DistributedContext &workers_and_aggregator_ctx,
    const ttml::autograd::DistributedContext &aggregator_and_optimizer_ctx,
    const SortedParameters &sorted_model_parameters,
    int workers) {
    Rank optimizer_rank{*aggregator_and_optimizer_ctx.rank() + 1};

    std::vector<ttml::core::distributed::SendRequestGuard> send_guards;
    send_guards.reserve(sorted_model_parameters.size());
    int tag_counter = 0;

    for (auto &[name, tensor_ptr] : sorted_model_parameters) {
        if (!tensor_ptr->get_requires_grad()) {
            continue;
        }

        auto tensor = tensor_ptr->get_value();
        ttml::core::distributed::recv_tensor(workers_and_aggregator_ctx, tensor, ttml::core::distributed::Rank{0});
        for (int worker_id = 1; worker_id < workers; ++worker_id) {
            auto tensor_to_add = ttnn::empty_like(tensor_ptr->get_value());
            ttml::core::distributed::recv_tensor(
                workers_and_aggregator_ctx, tensor_to_add, ttml::core::distributed::Rank{worker_id});
            tensor = ttnn::add(tensor, tensor_to_add);
        }
        tensor = ttnn::multiply(tensor, 1.0F / static_cast<float>(workers));

        // ttml::core::distributed::send_tensor(aggregator_and_optimizer_ctx, tensor, optimizer_rank);
        send_guards.emplace_back(ttml::core::distributed::isend_tensor(
            aggregator_and_optimizer_ctx, tensor, optimizer_rank, Tag{tag_counter++}));
    }
}

// void aggregate_gradients_and_broadcast_weights(
//     const ttml::autograd::DistributedContext &workers_and_aggregator_ctx,
//     const ttml::autograd::DistributedContext &aggregator_and_optimizer_ctx,
//     const SortedParameters &sorted_model_parameters,
//     int workers) {
//     Rank optimizer_rank{*aggregator_and_optimizer_ctx.rank() + 1};
//     for (auto &[name, tensor_ptr] : sorted_model_parameters) {
//         if (!tensor_ptr->get_requires_grad()) {
//             continue;
//         }

//         auto tensor = ttnn::empty_like(tensor_ptr->get_value());
//         ttml::core::distributed::recv_tensor(workers_and_aggregator_ctx, tensor, ttml::core::distributed::Rank{0});
//         for (int worker_id = 1; worker_id < workers; ++worker_id) {
//             auto tensor_to_add = ttnn::empty_like(tensor_ptr->get_value());
//             ttml::core::distributed::recv_tensor(
//                 workers_and_aggregator_ctx, tensor_to_add, ttml::core::distributed::Rank{worker_id});
//             tensor = ttnn::add(tensor, tensor_to_add);
//         }
//         tensor = ttnn::multiply(tensor, 1.0F / static_cast<float>(workers));
//         ttml::core::distributed::send_tensor(aggregator_and_optimizer_ctx, tensor, optimizer_rank);

//         ttml::core::distributed::recv_tensor(
//             aggregator_and_optimizer_ctx, tensor, ttml::core::distributed::Rank{optimizer_rank});
//         ttml::core::distributed::broadcast_tensor(
//             workers_and_aggregator_ctx, tensor, workers_and_aggregator_ctx.rank());
//     }
// }

// void send_aggregated_gradients_from_workers_to_optimizer(
//     const ttml::autograd::DistributedContext &workers_and_aggregator_ctx,
//     const ttml::autograd::DistributedContext &aggregator_and_optimizer_ctx,
//     const SortedParameters &sorted_model_parameters,
//     int workers) {
//     Rank optimizer_rank{*aggregator_and_optimizer_ctx.rank() + 1};

//     std::vector<std::vector<ttnn::Tensor>> tensors_per_worker(workers);
//     for (int worker_id = 0; worker_id < workers; ++worker_id) {
//         tensors_per_worker[worker_id].reserve(sorted_model_parameters.size());
//     }

//     for (auto &[name, tensor_ptr] : sorted_model_parameters) {
//         if (!tensor_ptr->get_requires_grad()) {
//             continue;
//         }

//         for (auto &tensors : tensors_per_worker) {
//             tensors.push_back(tensor_ptr->get_value());
//         }
//     }

//     for (int worker_id = 0; worker_id < workers; ++worker_id) {
//         auto &tensors = tensors_per_worker[worker_id];
//         auto rank = ttml::core::distributed::Rank{worker_id};
//         ttml::core::distributed::receive_all_tensors(workers_and_aggregator_ctx, tensors, rank);
//     }

//     auto model_param_idx = 0;
//     for (auto &[name, tensor_ptr] : sorted_model_parameters) {
//         if (!tensor_ptr->get_requires_grad()) {
//             continue;
//         }

//         auto tensor = tensors_per_worker[0][model_param_idx];
//         for (int worker_id = 1; worker_id < workers; ++worker_id) {
//             auto tensor_to_add = tensors_per_worker[worker_id][model_param_idx];
//             tensor = ttnn::add(tensor, tensor_to_add);
//         }

//         tensor = ttnn::multiply(tensor, 1.0F / static_cast<float>(workers));
//         model_param_idx++;
//         ttml::core::distributed::send_tensor(aggregator_and_optimizer_ctx, tensor, optimizer_rank);
//     }
// }

void recv_and_broadcast_tensor(
    const ttml::autograd::DistributedContext &recv_ctx,
    const ttml::autograd::DistributedContext &broadcast_ctx,
    ttnn::Tensor &tensor,
    Rank source,
    Rank root) {
    auto cpu_tensor = tensor.cpu();
    auto buffers = ttml::core::get_bytes_from_cpu_tensor(cpu_tensor);

    for (auto buffer : buffers) {
        recv_ctx.recv(buffer, source, Tag{0});
    }

    for (auto buffer : buffers) {
        broadcast_ctx.broadcast(buffer, root);
    }
}

void recv_and_broadcast_all(
    const ttml::autograd::DistributedContext &recv_ctx,
    const ttml::autograd::DistributedContext &broadcast_ctx,
    const SortedParameters &sorted_model_parameters,
    Rank source,
    Rank root) {
    fmt::println("receive and broadcast all tensors from rank {} to rank {}", source, root);

    std::vector<ttnn::Tensor> cpu_tensors;
    cpu_tensors.reserve(sorted_model_parameters.size());

    std::size_t total_size = 0;
    for (auto &[name, tensor_ptr] : sorted_model_parameters) {
        auto tensor = tensor_ptr->get_value();

        auto cpu_tensor = tensor.cpu();
        cpu_tensors.push_back(cpu_tensor);
        auto buffers = ttml::core::get_bytes_from_cpu_tensor(cpu_tensor);
        for (auto buffer : buffers) {
            total_size += buffer.size();
        }
    }

    std::vector<std::byte> combined_buffer(total_size);
    recv_ctx.recv(combined_buffer, source, Tag{0});

    auto *ptr = combined_buffer.data();
    for (auto &tensor : cpu_tensors) {
        auto buffers = ttml::core::get_bytes_from_cpu_tensor(tensor);
        for (auto &buffer : buffers) {
            std::copy(ptr, ptr + buffer.size(), buffer.begin());
            ptr += buffer.size();

            broadcast_ctx.broadcast(buffer, root);
        }
    }

    fmt::println("[done] receive and broadcast all tensors from rank {} to rank {}", source, root);
}

void send_weights_from_optimizer_to_workers(
    const ttml::autograd::DistributedContext &workers_and_aggregator_ctx,
    const ttml::autograd::DistributedContext &aggregator_and_optimizer_ctx,
    const SortedParameters &sorted_model_parameters,
    int workers) {
    Rank optimizer_rank{*aggregator_and_optimizer_ctx.rank() + 1};
    recv_and_broadcast_all(
        aggregator_and_optimizer_ctx,
        workers_and_aggregator_ctx,
        sorted_model_parameters,
        optimizer_rank,
        workers_and_aggregator_ctx.rank());
    // for (auto &[name, tensor_ptr] : sorted_model_parameters) {
    //     auto tensor = tensor_ptr->get_value();
    //     recv_and_broadcast_tensor(
    //         aggregator_and_optimizer_ctx,
    //         workers_and_aggregator_ctx,
    //         tensor,
    //         optimizer_rank,
    //         workers_and_aggregator_ctx.rank());
    // }
}

int main(int argc, char **argv) {
    auto &ctx = ttml::autograd::ctx();
    ctx.initialize_distributed_context(argc, argv);
    auto &distributed_ctx = ctx.get_distributed_context();

    CLI::App app{"Multihost Example"};
    fmt::print("Size {}, Rank {}: Initializing MPI context\n", *distributed_ctx.size(), *distributed_ctx.rank());
    argv = app.ensure_utf8(argv);

    std::string config_name = std::string(CONFIGS_FOLDER) + "/training_shakespear_nanogpt_3tier.yaml";

    bool ddp = false;
    bool enable_tp = false;
    app.add_option("-c,--config", config_name, "Yaml Config name")->default_val(config_name);
    app.add_option("-d,--ddp", ddp, "Enable DDP")->default_val(ddp);
    app.add_option("-p,--tp", enable_tp, "Enable TP")->default_val(enable_tp);

    CLI11_PARSE(app, argc, argv);

    // tensor parallel is not supported yet
    three_tier_arch::initialize_device(ddp, enable_tp);

    auto yaml_config = YAML::LoadFile(config_name);
    three_tier_arch::TrainingConfig config = three_tier_arch::parse_config(yaml_config);

    fmt::println("Aggregator config setup finished");

    auto [steps_per_dataset, vocab_size] = three_tier_arch::get_steps_per_dataset_and_vocab_size(config);
    auto *device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();

    auto num_devices = static_cast<uint32_t>(device->num_devices());
    auto should_be_divisible_by = (enable_tp ? num_devices : 1U) * 32U;
    vocab_size = three_tier_arch::round_up_to_tile(vocab_size, should_be_divisible_by);
    config.transformer_config.vocab_size = vocab_size;

    auto create_model = [enable_tp](const auto &config) -> std::shared_ptr<ttml::autograd::ModuleBase> {
        if (enable_tp) {
            return ttml::models::distributed::gpt2::create(config);
        }
        return ttml::models::gpt2::create(config);
    };
    auto model = create_model(config.transformer_config);

    auto model_parameters = model->parameters();
    auto sorted_model_parameters = SortedParameters(model_parameters.begin(), model_parameters.end());
    // auto sorted_model_parameters = SortedParameters(model_parameters);

    auto workers = config.num_mh_workers;

    auto workers_and_aggregator_ranks =
        three_tier_arch::get_workers_and_aggregator_ranks(static_cast<uint32_t>(*distributed_ctx.rank()));
    auto workers_and_aggregator_ctx =
        ttml::autograd::ctx().get_distributed_context().create_sub_context(workers_and_aggregator_ranks);

    auto aggregator_and_optimizer_ranks = std::vector<int>{*distributed_ctx.rank(), *distributed_ctx.rank() + 1};
    auto aggregator_and_optimizer_ctx =
        ttml::autograd::ctx().get_distributed_context().create_sub_context(aggregator_and_optimizer_ranks);

    send_weights_from_optimizer_to_workers(
        *workers_and_aggregator_ctx, *aggregator_and_optimizer_ctx, sorted_model_parameters, workers);

    uint32_t global_step = 0;
    for (uint32_t epoch = 0; epoch < config.num_epochs; ++epoch) {
        for (uint32_t step = 0; step < steps_per_dataset; ++step, ++global_step) {
            send_aggregated_gradients_from_workers_to_optimizer(
                *workers_and_aggregator_ctx, *aggregator_and_optimizer_ctx, sorted_model_parameters, workers);
            send_weights_from_optimizer_to_workers(
                *workers_and_aggregator_ctx, *aggregator_and_optimizer_ctx, sorted_model_parameters, workers);
            // aggregate_gradients_and_broadcast_weights(
            //     *workers_and_aggregator_ctx, *aggregator_and_optimizer_ctx, sorted_model_parameters, workers);
            if (global_step >= config.max_steps) {
                break;
            }
        }
        if (global_step >= config.max_steps) {
            break;
        }
    }

    distributed_ctx.barrier();
    fmt::print("Rank {}: Finalized MPI context\n", *distributed_ctx.rank());
    return 0;
}
