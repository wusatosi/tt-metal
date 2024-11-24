# Assumptions
1. Many ops will be modeled
2. Models will differ signficantly in their arguments and internal composition
3. The goal is to have one model per op, but there may temporarily be several models for an op. In these cases, the caller should be able to pick their desired model version
4. It is desirable to have a unified query interface for all ops
5. Models are device specific
6. Failure modes for op models include: failing to load/initialize a model, quering an op or op-mode that is not modeled yet, or quering an op with invalid parameters
7. A failure in the op models should not cause a crash or put the models in an invalid state. The caller should be informed and should have the option of ignoring the failure(s)


# Proposed High-Level Structure
## Organization
- A new top level folder in the `tt-metal` repo, named `ttnn_op_perf_models`
- "Real" code in C++, python bindings to be written if required
- A new namespace `TTNN_OP_PERF_MODELS`
- Model parameters to be stored on github
  - For models with few parameters, this may be directly in code
  - For models with many parameters, it will be up to the model to serialize/deserialize the parameters as desired

## `OP_PERF_MODELS_MANAGER`
An RAII resource manager. The manager creates and maintains a mapping, `m_models`, of op -> version -> `OP_PERF_MODEL` object.

The `OP_PERF_MODELS_MANAGER` will:
- load all or a set of models on construction
- provide the query entry point for external callers
- determine the `OP_PERF_MODEL` object to service a particular query

The `OP_PERF_MODELS_MANAGER` will **not**:
- have any knowledge of how a model is loaded/initialized
- perform any validation on a query beyond matching it to an `OP_PERF_MODEL`

## `OP_PERF_MODEL`
A pure virtual base class that describes the interface of a model. Any model will be requied to implement:
1. a `load()` function that initializes the model by handling any parameter serialization/deserialization
2. a `get_op_duration()` function that responds to a particular query


## Anatomy of a query
### Type of operation: `OP_T`
An enum of supported operations

### Version:
Indicates the desired version for the op model. Defaults to the latest version

### The "relevant" parameters for the provided operation: `op_params`
A key-value pair. Greatly limits static analysis, but allows for maximum flexibility with low development overhead

### Tensor operand(s): `TENSOR_PARAMS`
A struct that encapsulates all potentially relevant information about one tensor operand of an op. Ops with multiple operands will have a vector of `TENSOR_PARAMS` objects passed to them. There will be utility functions to create this struct given a tensor object.

```
struct TENSOR_PARAMS {
    std::vector<size_t> shape;
    uint64_t tile_size;
    // dtype
    // memory config:
    //     - L1 sharded, DRAM sharded or interleaved?
    //     - if sharded, shard orientation and strategy
    // layout
};
```
## Why not allow callers to directly pass in a ttnn op object?
Short answer: I think we can have a cleaner and more stable code base this way. I am very open to feedback on this question.

Long answer:
`ttnn` operations are handled using incredibly generic code. For example, see [decorators.hpp](../ttnn/cpp/ttnn/decorators.hpp). My current understanding is that:
- there is no common interface for an op
- there is no common structure for op parameters and operands
  - New ops will follow [this](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/adding_new_ttnn_operation.html) structure that *does* provide a common grouping of tensor vs non-tensor args, but existing ops broadly do *not*. Furthermore, this is *not* enforced programatically
- each op is its own type

If we wanted to directly accept a ttnn op object, the high level `get_op_duration()` API will need to:
- completely erase the type of the op (so it can accept any op), but still pick the right model for each op
  - I think this is possible but potentially very complicated templated code
- take a variadic pack of op args and pass them directly to each `op_perf_model`. Then, the `op_perf_model` for each op must unpack the parameters *exactly* identical to how the op does it
  - Any changes to an op's interface will require updates to it's `op_perf_model` code, even if there are no functional changes or performance impacts
  - E.g. if someone wants to modify the [reshard op](../ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/reshard.hpp) to accept arguments in a different order or to modify/change the `MemoryConfig` struct, they will also need to modify the implementation of any `op_perf_model`s for this op. This appears to negate most of the abstraction of having an API for the perf models
