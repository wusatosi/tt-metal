// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>

#include "core/distributed/distributed.hpp"
#include "models/gpt2.hpp"
#include "serialization/serializable.hpp"

// namespace name can't start with a digit
namespace three_tier_arch {

constexpr auto gpt2_tokenizer_file_name = "/gpt2-tokenizer.json";

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

struct TrainingConfig {
    std::string project_name;
    std::string model_type;  // one of "gpt2", "llama"
    uint32_t seed = 5489U;
    uint32_t model_save_interval = 500;
    uint32_t batch_size = 64;
    uint32_t num_epochs = 1;
    uint32_t max_steps = 5000;
    float learning_rate = 3e-4F;
    float weight_decay = 1e-2F;
    bool use_moreh_adamw = false;
    // works only for AdamW
    bool use_kahan_summation = false;
    // accumulate batches for gradient update
    uint32_t gradient_accumulation_steps = 1;
    std::string model_path;
    std::string data_path;
    std::string tokenizer_type = "char";
    std::string scheduler_type = "identity";
    std::string tokenizer_path = std::string(DATA_FOLDER) + gpt2_tokenizer_file_name;
    bool use_clip_grad_norm = false;
    float clip_grad_norm_max_norm = 1.0F;
    ttml::models::gpt2::TransformerConfig transformer_config;

    bool enable_mpi = false;
    uint32_t num_mh_workers = 1U;
};

TrainingConfig parse_config(const YAML::Node& yaml_config);

std::pair<uint32_t, uint32_t> get_steps_per_dataset_and_vocab_size(const TrainingConfig& config);

std::vector<int> get_workers_and_aggregator_ranks(uint32_t workers);

std::string read_file_to_str(const std::string& file_path);

uint32_t round_up_to_tile(uint32_t value, uint32_t tile_size = 32U);

void initialize_device(bool ddp, bool tp);

class SortedParameters {
public:
    using value_type = std::pair<std::string, ttml::autograd::TensorPtr>;
    using iterator = std::vector<value_type>::iterator;
    using const_iterator = std::vector<value_type>::const_iterator;

    SortedParameters() = default;
    SortedParameters(const SortedParameters&) = default;
    SortedParameters(SortedParameters&&) noexcept = default;
    SortedParameters& operator=(const SortedParameters&) = default;
    SortedParameters& operator=(SortedParameters&&) noexcept = default;

    // explicit SortedParameters(std::vector<value_type> parameters) : m_parameters(std::move(parameters)) {}
    explicit SortedParameters(const ttml::serialization::NamedParameters& parameters) {
        m_parameters.reserve(parameters.size());
        for (const auto& [key, tensor_ptr] : parameters) {
            m_parameters.emplace_back(key, tensor_ptr);
        }
        std::sort(m_parameters.begin(), m_parameters.end(), [](const value_type& a, const value_type& b) {
            return a.second->get_tensor_id() > b.second->get_tensor_id() || a.second->get_tensor_id() == 1UL;
        });
    }

    [[nodiscard]] size_t size() const {
        return m_parameters.size();
    }
    [[nodiscard]] bool empty() const {
        return m_parameters.empty();
    }

    [[nodiscard]] ttml::autograd::TensorPtr& operator[](const std::string& key) {
        for (auto& [name, tensor_ptr] : m_parameters) {
            if (name == key) {
                return tensor_ptr;
            }
        }
        throw std::out_of_range("Parameter not found: " + key);
    }
    [[nodiscard]] iterator begin() {
        return m_parameters.begin();
    }
    [[nodiscard]] iterator end() {
        return m_parameters.end();
    }
    [[nodiscard]] const_iterator begin() const {
        return m_parameters.begin();
    }
    [[nodiscard]] const_iterator end() const {
        return m_parameters.end();
    }

private:
    std::vector<std::pair<std::string, ttml::autograd::TensorPtr>> m_parameters;
};

}  // namespace three_tier_arch
