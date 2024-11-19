#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace TTNN_OP_MODELS {

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

class OP_MODEL {
public:
    virtual void load(std::string path) = 0;

    virtual std::optional<uint32_t>
    get_op_duration(const std::unordered_map<std::string, std::string>& op_params, TENSOR_PARAMS arg) = 0;

    virtual std::optional<uint32_t>
    get_op_duration(const std::unordered_map<std::string, std::string>& op_params, TENSOR_PARAMS arg1, TENSOR_PARAMS arg2) = 0;

    virtual std::optional<uint32_t>
    get_op_duration(const std::unordered_map<std::string, std::string>& op_params, TENSOR_PARAMS arg1, TENSOR_PARAMS arg2, TENSOR_PARAMS arg3) = 0;

    virtual ~OP_MODEL() = default;

    bool m_is_loaded = false;
};

class RESHARD_OP_MODEL_V1 : public OP_MODEL {
    virtual void load(std::string path) override {
        //read large file from disk
    };

    virtual std::optional<uint32_t>
    get_op_duration(const std::unordered_map<std::string, std::string>& op_params, TENSOR_PARAMS arg) override {
        // log(reshard requires input and output tensor args)
        return std::nullopt;
    };

    virtual std::optional<uint32_t>
    get_op_duration(const std::unordered_map<std::string, std::string>& op_params, TENSOR_PARAMS arg1, TENSOR_PARAMS arg2) override {
        // actual model
        return 123;
    };

    virtual std::optional<uint32_t>
    get_op_duration(const std::unordered_map<std::string, std::string>& op_params, TENSOR_PARAMS arg1, TENSOR_PARAMS arg2, TENSOR_PARAMS arg3) override {
        // log(N/A to reshard)
        return std::nullopt;
    };
};

class RESHARD_OP_MODEL_V2 : public OP_MODEL {

};

class ELTWISE_MAX_OP_MODEL : public OP_MODEL {
    virtual void load(std::string path) override {
        // parameters directly in the model, so ignore the path
    };
};

}; // namespace TTNN_OP_MODELS
