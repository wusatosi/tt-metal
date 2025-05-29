// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "autograd/auto_context.hpp"
#include "core/distributed/distributed.hpp"
#include "optimizers/optimizer_base.hpp"

// class SortedParameters {
// public:
//     using value_type = std::pair<std::string, ttml::autograd::TensorPtr>;
//     using iterator = std::vector<value_type>::iterator;
//     using const_iterator = std::vector<value_type>::const_iterator;

//     SortedParameters() = default;
//     SortedParameters(const SortedParameters&) = default;
//     SortedParameters(SortedParameters&&) noexcept = default;
//     SortedParameters& operator=(const SortedParameters&) = default;
//     SortedParameters& operator=(SortedParameters&&) noexcept = default;

//     // explicit SortedParameters(std::vector<value_type> parameters) : m_parameters(std::move(parameters)) {}
//     explicit SortedParameters(const ttml::serialization::NamedParameters& parameters) {
//         m_parameters.reserve(parameters.size());
//         for (const auto& [key, tensor_ptr] : parameters) {
//             m_parameters.emplace_back(key, tensor_ptr);
//         }
//         std::sort(m_parameters.begin(), m_parameters.end(), [](const value_type& a, const value_type& b) {
//             return a.second->get_tensor_id() > b.second->get_tensor_id() || a.second->get_tensor_id() == 1UL;
//         });

//         auto rank = ttml::autograd::ctx().get_distributed_context().rank();
//         for (auto& [name, tensor_ptr] : m_parameters) {
//             fmt::println("Rank {}: Parameter: {}, Tensor ID: {}", *rank, name, tensor_ptr->get_tensor_id());
//         }
//     }

//     [[nodiscard]] size_t size() const {
//         return m_parameters.size();
//     }
//     [[nodiscard]] bool empty() const {
//         return m_parameters.empty();
//     }

//     [[nodiscard]] ttml::autograd::TensorPtr& operator[](const std::string& key) {
//         for (auto& [name, tensor_ptr] : m_parameters) {
//             if (name == key) {
//                 return tensor_ptr;
//             }
//         }
//         throw std::out_of_range("Parameter not found: " + key);
//     }
//     [[nodiscard]] iterator begin() {
//         return m_parameters.begin();
//     }
//     [[nodiscard]] iterator end() {
//         return m_parameters.end();
//     }
//     [[nodiscard]] const_iterator begin() const {
//         return m_parameters.begin();
//     }
//     [[nodiscard]] const_iterator end() const {
//         return m_parameters.end();
//     }

// private:
//     std::vector<value_type> m_parameters;
// };

using SortedParameters = std::map<std::string, ttml::autograd::TensorPtr>;

class RemoteOptimizer : public ttml::optimizers::OptimizerBase {
public:
    RemoteOptimizer(ttml::serialization::NamedParameters parameters, int aggregator_rank);

    void zero_grad() override;

    void step() override;

    [[nodiscard]] ttml::serialization::StateDict get_state_dict() const override;

    void set_state_dict(const ttml::serialization::StateDict& dict) override;

    [[nodiscard]] size_t get_steps() const override;

    void set_steps(size_t steps) override;

    SortedParameters get_sorted_parameters() const;

    void send_gradients();

    void receive_weights();

    void set_lr(float lr) override;

    [[nodiscard]] float get_lr() const override;

private:
    size_t m_steps{0};
    SortedParameters m_sorted_parameters;
    ttml::core::distributed::Rank m_aggregator_rank{0};
    std::shared_ptr<ttml::autograd::DistributedContext> m_distributed_ctx;
};
