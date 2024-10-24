# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def tt_all_reduce(input_tensor, mesh_device, cluster_axis, dim=0, num_links=2, memory_config=None):
    # Ensure the input tensor is in the correct memory configuration
    input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    gathered_tensor = ttnn.all_gather(
        input_tensor,
        dim,
        num_links=num_links,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
    )
    reduced_tensors = ttnn.experimental.fast_reduce_nc(
        gathered_tensor, dims=[dim], output=None, compute_kernel_config=None
    )

    return reduced_tensors


def tt_sharded_all_reduce(input_tensor, mesh_device, cluster_axis, dim=0, num_links=2, memory_config=None):
    gathered_tensor = ttnn.all_gather(
        input_tensor,
        dim,
        num_links=num_links,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        memory_config=memory_config,
        topology=ttnn.Topology.Linear,
    )
    # Fast_reduce_nc does not support sharded memory configuration, convert to interleaved
    gathered_tensor = ttnn.to_memory_config(gathered_tensor, ttnn.L1_MEMORY_CONFIG)
    reduced_tensors = ttnn.experimental.fast_reduce_nc(
        gathered_tensor, dims=[dim], output=None, compute_kernel_config=None, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    return reduced_tensors


def tt_sharded_all_gather(input_tensor, mesh_device, cluster_axis, dim, num_links=2, memory_config=None):
    # Ensure the input tensor is in the correct memory configuration

    return ttnn.all_gather(
        input_tensor,
        dim,
        num_links=num_links,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        memory_config=memory_config,
        topology=ttnn.Topology.Linear,
    )


def tt_distributed_rmsnorm(inp, epsilon, gamma, mesh_device, compute_kernel_config):
    # Run distributed rmsnorm part 1
    tt_stats = ttnn.rms_norm_pre_all_gather(inp, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16)

    padded_shape = (1, 1, inp.shape[-2], 32)
    tt_stats = ttnn.reshape(tt_stats, ttnn.Shape(padded_shape, padded_shape))  # TODO: Figure out why we need this
    tt_stats = tt_all_gather(
        tt_stats,
        mesh_device=mesh_device,
        dim=3,
        cluster_axis=1,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run distributed rmsnorm part 2
    tt_out = ttnn.rms_norm_post_all_gather(
        inp, tt_stats, epsilon=epsilon, weight=gamma, compute_kernel_config=compute_kernel_config
    )

    tt_stats.deallocate(True)

    return tt_out


def tt_sharded_distributed_rmsnorm(
    inp, epsilon, gamma, mesh_device, ln_sharded_input_memcfg, ln_sharded_progcfg, ln_sharded_stats_memcfg
):
    inp = ttnn.to_memory_config(inp, memory_config=ln_sharded_input_memcfg)

    # Run distributed rmsnorm part 1
    tt_stats = ttnn.rms_norm_pre_all_gather(inp, program_config=ln_sharded_progcfg)

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
