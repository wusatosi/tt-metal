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
- A new top level folder in the `tt-metal` repo, named `ttnn_op_models`
- "Real" code in C++, python bindings to be written if required
- A new namespace `TTNN_OP_MODELS`
- Model parameters to be stored on github
  - For models with few parameters, this may be directly in code
  - For models with many parameters, it will be up to the model to serialize/deserialize the parameters as desired
- No linking to `ttnn` or `tt-metal`. Model API will be callable from outside the compiler

## `OP_MODELS_MANAGER`
An RAII resource manager. The manager creates and maintains a mapping, `m_models`, of op -> version -> `OP_MODEL` object.

The `OP_MODELS_MANAGER` will:
- load all or a set of models on construction
- provide the query entry point for external callers
- determine the `OP_MODEL` object to service a particular query

The `OP_MODELS_MANAGER` will **not**:
- have any knowledge of how a model is loaded/initialized
- perform any validation on a query beyond matching it to an `OP_MODEL`

## `OP_MODEL`
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
A struct that encapsulates all potentially relevant information about one tensor operand of an op. Ops with multiple operands will have multiple `TENSOR_PARAMS` objects passed to them. Ideally, there will be utility functions in `ttnn` to create this struct given a tensor object.

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
