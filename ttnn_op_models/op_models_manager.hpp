#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <ttnn_op_models/op_model.hpp>



namespace TTNN_OP_MODELS {

enum class OP_T {
    ELTWISE_MAX,
    RESHARD
};

enum class DEVICE_T;

static const std::string LATEST_VERSION = "LATEST_VERSION";

static const std::unordered_map<OP_T, std::unordered_map<std::string, std::string>>
    OP_MODEL_FILES_WORMHOLE_B0 {
    {OP_T::RESHARD,
        {
            {"v1", "model_params/wormhole_b0/reshard_v1.param"},
            {LATEST_VERSION, "model_params/wormhole_b0/reshard_v2.param"}
        }},
    {OP_T::ELTWISE_MAX, {{LATEST_VERSION, ""}}} // parameters directly in the model
};

/*
    @brief: RAII manager for all op models. Responsible for loading all models and providing query access
*/
class OP_MODELS_MANAGER {
public:
    /*
        @brief: Constructor

        @arg[device]: Load models for this device
        @arg[ops]: Optional. If specified, only load models for these ops. Otherwise, load all models
    */
    OP_MODELS_MANAGER(DEVICE_T device, std::optional<std::vector<OP_T>> ops = std::nullopt) {
        // Based on device:
        //    1. Create the m_models mapping by constructing the model objects but *not* loading them.
        //          Likely ugly code
        //    2. For every model in m_models, call load() and pass in the file path
    };

    /*
        @brief: Main entry point for all external callers

        @arg[op]: The ttnn operation
        @arg[op_params]: Parameter list of the operation. May be different for each op
        @arg[version]: For ops with multiple model versions, select the model. Defaults to the latest
        @arg[tensor_args]: Variadic pack of parameters for the tensor operand(s).

        @return: Expected kernel duration in nanoseconds. std::nullopt if the model is not implemented
    */
    template<typename... T>
    typename std::enable_if<(std::is_same<TENSOR_PARAMS, T>::value && ...), std::optional<uint32_t>>::type
    get_op_duration(OP_T op, const std::unordered_map<std::string, std::string>& op_params, std::string version = LATEST_VERSION, T... tensor_args);

private:
    DEVICE_T m_device;

    // mapping of OP -> model version -> model implementation
    std::unordered_map<OP_T, std::unordered_map<std::string, std::unique_ptr<OP_MODEL>>> m_models;

    /*
        @brief: Load op models from disk

        @arg[device]: Load models for this device
        @arg[ops]: Optional. If specified, only load models for these ops. Otherwise, load all models
    */
    void load_models(DEVICE_T device, std::optional<std::vector<OP_T>> ops = std::nullopt);
};

}; // namespace TTN_OP_MODELS
