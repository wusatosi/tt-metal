#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace TTNN_OP_PERF_MODELS {

/*
    @brief: Encapsulate all (potentially) relevant aspects of an operand tensor
*/
struct TENSOR_PARAMS {
    std::vector<size_t> shape;
    uint64_t tile_size;
    // dtype
    // memory config:
    //     - L1 sharded, DRAM sharded or interleaved?
    //     - if sharded, shard orientation and strategy
    // layout
};

class OP_PERF_MODEL {
public:
    virtual void load(std::string path) = 0;

    virtual std::optional<uint32_t>
    get_op_duration(const std::unordered_map<std::string, std::string>& op_params, const std::vector<TENSOR_PARAMS>& args) = 0;

    virtual ~OP_PERF_MODEL() = default;

    bool m_is_loaded = false;
};

class RESHARD_OP_PERF_MODEL_V1 : public OP_PERF_MODEL {
    virtual void load(std::string path) override {
        //read large file from disk
    };

    virtual std::optional<uint32_t>
    get_op_duration(const std::unordered_map<std::string, std::string>& op_params, const std::vector<TENSOR_PARAMS>& args) override {
        if (args.size() != 2) {
            // invalid for reshard
            return std::nullopt;
        }

        // actual model
        return 123;
    }
};

class RESHARD_OP_PERF_MODEL_V2 : public OP_PERF_MODEL {

};

class ELTWISE_MAX_OP_PERF_MODEL : public OP_PERF_MODEL {
    virtual void load(std::string path) override {
        // parameters directly in the model, so ignore the path
    };
};

}; // namespace TTNN_OP_MODELS
