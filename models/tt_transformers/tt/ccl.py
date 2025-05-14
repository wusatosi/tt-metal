# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os
import traceback

import ttnn

# Global set to store seen configurations
_seen_configs = None


# def tt_all_reduce(input_tensor, mesh_device, cluster_axis=0, dim=0, num_links=2, memory_config=None, sharded=False):
def tt_all_reduce(
    input_tensor,
    mesh_device,
    cluster_axis=0,
    dim=0,
    num_reduce_scatter_links=1,
    num_all_gather_links=2,
    topology=ttnn.Topology.Linear,
    memory_config=None,
    sharded=False,
    dtype=ttnn.bfloat16,
    use_composite=False,
):
    # N150
    if list(mesh_device.shape) == [1, 1] or (cluster_axis == 1 and 1 in list(mesh_device.shape)):
        return input_tensor

    # Log tensor details before collective
    log_tensor_details(
        input_tensor=input_tensor,
        mesh_device=mesh_device,
        cluster_axis=cluster_axis,
        dim=dim,
        num_reduce_scatter_links=num_reduce_scatter_links,
        num_all_gather_links=num_all_gather_links,
        topology=topology,
        memory_config=memory_config,
        sharded=sharded,
        dtype=dtype,
        use_composite=use_composite,
    )

    # Ensure dim 0 and 1 are 1
    original_shape = input_tensor.shape
    if original_shape[0] != 1 or original_shape[1] != 1:
        input_tensor = ttnn.reshape(
            input_tensor, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

    # N300 and T3K: reduce_scatter
    if 1 in list(mesh_device.shape):
        if input_tensor.is_sharded() and not sharded:
            input_tensor_sharded = input_tensor
            input_tensor = ttnn.sharded_to_interleaved(input_tensor_sharded, ttnn.L1_MEMORY_CONFIG)
            input_tensor_sharded.deallocate(True)
        reduced = ttnn.reduce_scatter(
            input_tensor,
            dim=dim,
            math_op=ttnn.ReduceType.Sum,
            num_links=num_reduce_scatter_links,
            topology=topology,
            memory_config=memory_config,
        )
        input_tensor.deallocate(True)
        return reduced

    # TG: all_reduce
    # Cast to CCL dtype
    if input_tensor.dtype != dtype:
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG, dtype)  # typecast and to interleaved
        if sharded and memory_config is not None:
            input_tensor = ttnn.to_memory_config(input_tensor, memory_config, dtype)  # to sharded

    # Ensure the input tensor is in the correct memory configuration
    if not sharded:  # prefill
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    if not use_composite:
        gathered_tensor = ttnn.all_gather(
            input_tensor,
            dim,
            num_links=num_all_gather_links,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            topology=topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if not sharded else memory_config,
        )

        if sharded:
            gathered_tensor = ttnn.to_memory_config(gathered_tensor, ttnn.L1_MEMORY_CONFIG)

        reduced_tensor = ttnn.experimental.fast_reduce_nc(
            gathered_tensor,
            dims=[dim],
            output=None,
            compute_kernel_config=None,
            memory_config=ttnn.L1_MEMORY_CONFIG if sharded else ttnn.DRAM_MEMORY_CONFIG,
        )
        gathered_tensor.deallocate(True)
    else:
        input_mem_cfg = input_tensor.memory_config()
        reduced_tensor = ttnn.reduce_scatter(
            input_tensor,
            dim=dim,
            num_links=num_reduce_scatter_links,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            math_op=ttnn.ReduceType.Sum,
            topology=topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if not sharded else memory_config,
        )

        reduced_tensor = ttnn.all_gather(
            reduced_tensor,
            dim,
            num_links=num_all_gather_links,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            topology=topology,
            memory_config=input_mem_cfg,
        )

    # Reshape the reduced tensor to the original shape
    reduced_tensor = ttnn.reshape(reduced_tensor, original_shape)

    return reduced_tensor


def tt_all_gather(
    input_tensor,
    mesh_device,
    cluster_axis,
    dim,
    num_links=2,
    memory_config=None,
    sharded=False,
    topology=ttnn.Topology.Linear,
    dtype=ttnn.bfloat16,
):
    # N150
    if list(mesh_device.shape) == (1, 1) or (cluster_axis == 1 and 1 in list(mesh_device.shape)):
        return input_tensor

    # Log tensor details before collective
    log_tensor_details(
        input_tensor=input_tensor,
        mesh_device=mesh_device,
        cluster_axis=cluster_axis,
        dim=dim,
        num_links=num_links,
        memory_config=memory_config,
        sharded=sharded,
        topology=topology,
        dtype=dtype,
    )

    # Ensure the input tensor is in the correct memory configuration
    if not sharded:
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    # Cast to CCL dtype
    if input_tensor.dtype != dtype:
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG, dtype)  # typecast and to interleaved
        if sharded and memory_config is not None:
            input_tensor = ttnn.to_memory_config(input_tensor, memory_config, dtype)  # to sharded

    if cluster_axis is None:
        gathered = ttnn.all_gather(
            input_tensor,
            dim,
            num_links=num_links,
            topology=topology,
            memory_config=memory_config,
        )
    else:
        gathered = ttnn.all_gather(
            input_tensor,
            dim,
            num_links=num_links,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            topology=topology,
            memory_config=memory_config,
        )
    input_tensor.deallocate(True)
    return gathered


def tt_distributed_rmsnorm(inp, epsilon, gamma, mesh_device, compute_kernel_config):
    # Run distributed rmsnorm part 1
    tt_stats = ttnn.rms_norm_pre_all_gather(inp, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16)
    padded_shape = (1, 1, inp.shape[-2], 32)
    tt_stats = ttnn.reshape(tt_stats, ttnn.Shape(padded_shape))  # TODO: Figure out why we need this

    # Log tensor details before collective
    log_tensor_details(
        input_tensor=tt_stats,
        mesh_device=mesh_device,
        dim=3,
        cluster_axis=1,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
    )

    tt_stats_gathered = tt_all_gather(
        tt_stats,
        mesh_device=mesh_device,
        dim=3,
        cluster_axis=1,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_stats.deallocate(True)

    # Run distributed rmsnorm part 2
    tt_out = ttnn.rms_norm_post_all_gather(
        inp, tt_stats_gathered, epsilon=epsilon, weight=gamma, compute_kernel_config=compute_kernel_config
    )

    tt_stats_gathered.deallocate(True)
    # inp.deallocate(True)

    return tt_out


def tt_sharded_distributed_rmsnorm(
    inp, epsilon, gamma, mesh_device, ln_sharded_input_memcfg, ln_sharded_progcfg, ln_sharded_stats_memcfg
):
    inp = ttnn.to_memory_config(inp, memory_config=ln_sharded_input_memcfg)

    # Run distributed rmsnorm part 1
    tt_stats = ttnn.rms_norm_pre_all_gather(inp, program_config=ln_sharded_progcfg)

    # Log tensor details before collective
    log_tensor_details(
        input_tensor=tt_stats,
        mesh_device=mesh_device,
        dim=3,
        num_links=1,
        cluster_axis=1,
        memory_config=ln_sharded_stats_memcfg,
        topology=ttnn.Topology.Linear,
    )

    # All gather stats
    tt_stats = ttnn.all_gather(
        tt_stats,
        3,
        num_links=1,
        cluster_axis=1,
        mesh_device=mesh_device,
        memory_config=ln_sharded_stats_memcfg,
        topology=ttnn.Topology.Linear,
    )

    # Run distributed rmsnorm part 2
    tt_out = ttnn.rms_norm_post_all_gather(
        inp,
        epsilon=epsilon,
        weight=gamma,
        program_config=ln_sharded_progcfg,
        stats=tt_stats,
    )
    tt_stats.deallocate(True)

    return tt_out


def log_tensor_details(**kwargs):
    """
    Log tensor and environment details to a JSONL file, but only for unique configurations.
    A configuration is considered unique based on all parameters except input tensors.
    Input tensors are logged as their shape and memory config.

    Args:
        **kwargs: All parameters of the ttnn function being logged, where input tensors
                 will be logged as their shape and memory config
    """
    global _seen_configs

    # Extract input tensors and convert them to shape/memory_config
    processed_kwargs = {}
    for name, value in kwargs.items():
        if isinstance(value, ttnn.Tensor):
            processed_kwargs[f"{name}_shape"] = tuple(value.shape)  # Convert list to tuple
            processed_kwargs[f"{name}_memory_config"] = str(value.memory_config())
        else:
            processed_kwargs[name] = str(value)

    # Create a unique key for this configuration
    config_key = tuple(sorted(processed_kwargs.items()))

    # Initialize seen configs from file if not done yet
    if _seen_configs is None:
        _seen_configs = set()
        if os.path.exists("tensor_details.jsonl"):
            with open("tensor_details.jsonl", "r") as f:
                for line in f:
                    entry = json.loads(line)
                    # Convert entry back to sorted tuple of items
                    convert_list = lambda x: tuple(x) if isinstance(x, list) else x
                    entry_items = [
                        (k, convert_list(v))
                        for k, v in entry.items()
                        if k not in ["caller_location", "model_path", "mesh_device_env"]
                    ]
                    _seen_configs.add(tuple(sorted(entry_items)))

    # Skip if we've seen this configuration before
    if config_key in _seen_configs:
        return

    # Get the calling function location from the stack trace
    stack = traceback.extract_stack()
    current_file = os.path.basename(__file__)

    # Find the closest caller from a different file
    caller_location = None
    for frame in reversed(stack[:-1]):  # Skip the current frame
        if os.path.basename(frame.filename) not in [current_file, "distributed_norm.py", "lightweightmodule.py"]:
            caller_location = f"{os.path.basename(frame.filename)}:{frame.lineno}"
            break

    # Get environment variables
    llama_dir = os.environ.get("LLAMA_DIR", "")
    hf_model = os.environ.get("HF_MODEL", "")
    mesh_device_env = os.environ.get("MESH_DEVICE", "")

    # Create log entry
    log_entry = {
        **processed_kwargs,
        "model_path": llama_dir if llama_dir else hf_model,
        "mesh_device_env": mesh_device_env,
        "caller_location": caller_location,
    }

    # Add to seen configs and write to file
    _seen_configs.add(config_key)
    with open("tensor_details.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
