import ttnn
import torch
import ttnn.graph

from tests.ttnn.utils_for_testing import check_with_pcc


def to_graphviz(trace):
    output = ["digraph G {"]

    node_colors = {
        "buffer": "lightcyan",
        "buffer_allocate": "lightblue",
        "buffer_deallocate": "lightblue3",
        "tensor": "lightyellow",
        "circular_buffer_allocate": "hotpink",
        "circular_buffer_deallocate_all": "hotpink3",
        "function_start": "grey62",
        "function_end": "grey44",
        "capture_start": "lightgoldenrod",
        "capture_end": "lightgoldenrod4",
    }

    # Create nodes with labels
    for item in trace:
        counter = item["counter"]
        node_type = item["node_type"]

        # Start label with node type and counter
        label = f"{node_type} ({counter})\\n"

        # Add params to label if available
        if "params" in item and isinstance(item["params"], dict):
            for key, value in item["params"].items():
                label += f"{key}: {value}\\n"

        label = label.replace('"', "'")

        # Set node color based on node type
        color = node_colors.get(node_type, "white")

        # Add the node with its label
        output.append(f'  "{counter}" [label="{label}", style=filled, fillcolor="{color}"];')

    # Create edges with labels based on connections
    for item in trace:
        counter = item["counter"]
        if "connections" in item and isinstance(item["connections"], list):
            connection_label = 1  # Start label count from 1 for each node
            for connection in item["connections"]:
                output.append(f'  "{counter}" -> "{connection}" [label="{connection_label}"];')
                connection_label += 1

    output.append("}")
    return "\n".join(output)


def run_with_graph_capture(lambda_func, torch_input, torch_weights, torch_output):
    device = ttnn.open_device(
        device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH, ttnn.DispatchCoreAxis.ROW)
    )

    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)

    ttnn_output = lambda_func(device, torch_input, torch_weights)
    ttnn_host_out = ttnn.to_torch(ttnn_output)
    passed, value = check_with_pcc(torch_output, ttnn_host_out, 0.98)
    ttnn.close_device(device)

    if not passed:
        print(value)

    captured_graph = ttnn.graph.end_graph_capture()

    return passed, value, to_graphviz(captured_graph)


def test_simple_matmul():
    torch_input = torch.rand(2 * 32, 8 * 32)
    torch_weights = torch.rand(8 * 32, 16 * 32)
    torch_output = torch_input @ torch_weights

    def lambda_func(device, torch_input, torch_weights):
        ttnn_input = ttnn.from_torch(
            torch_input,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        ttnn_weights = ttnn.from_torch(
            torch_weights,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        return ttnn_input @ ttnn_weights

    passed, value, graphviz_output = run_with_graph_capture(lambda_func, torch_input, torch_weights, torch_output)
    print(graphviz_output)
    return passed


def test_2d_matmul():
    torch_input = torch.rand(2 * 32, 8 * 32)
    torch_weights = torch.rand(8 * 32, 16 * 32)
    torch_output = torch_input @ torch_weights

    def lambda_func(device, torch_input, torch_weights):
        r_cores = 2
        c_cores = 2

        input_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                grid=ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(r_cores - 1, c_cores - 1))]
                ),
                shard_shape=[torch_input.shape[0] // r_cores, torch_input.shape[1] // c_cores],
                shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
                halo=False,
            ),
        )

        output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                grid=ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(r_cores - 1, c_cores - 1))]
                ),
                shard_shape=[torch_input.shape[0] // r_cores, torch_weights.shape[1] // c_cores],
                shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
                halo=False,
            ),
        )

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=input_memory_config
        )
        ttnn_weights = ttnn.from_torch(
            torch_weights,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        matmul_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(2, 2),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=output_memory_config.shard_spec.shape[0] // 32,
            per_core_N=output_memory_config.shard_spec.shape[1] // 32,
            transpose_mcast=False,
            fused_activation=None,
        )
        return ttnn.matmul(
            input_tensor_a=ttnn_input,
            input_tensor_b=ttnn_weights,
            memory_config=output_memory_config,
            program_config=matmul_program_config,
        )

    passed, value, graphviz_output = run_with_graph_capture(lambda_func, torch_input, torch_weights, torch_output)
    print(graphviz_output)
    return passed


def test_1d_matmul_ws():
    torch_input = torch.rand(2 * 32, 8 * 32)
    torch_weights = torch.rand(8 * 32, 16 * 32)
    torch_output = torch_input @ torch_weights

    def lambda_func(device, torch_input, torch_weights):
        r_cores = 2
        c_cores = 2

        input_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                grid=ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(r_cores - 1, c_cores - 1))]
                ),
                shard_shape=[torch_input.shape[0], torch_input.shape[1] // (r_cores * c_cores)],
                shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
                halo=False,
            ),
        )

        output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                grid=ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(r_cores - 1, c_cores - 1))]
                ),
                shard_shape=[torch_input.shape[0], torch_weights.shape[1] // (r_cores * c_cores)],
                shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
                halo=False,
            ),
        )

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=input_memory_config
        )
        ttnn_weights = ttnn.from_torch(
            torch_weights,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        matmul_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(2, 2),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            # out_block_h
            # out_block_w
            per_core_M=output_memory_config.shard_spec.shape[0] // 32,
            per_core_N=output_memory_config.shard_spec.shape[1] // 32,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        return ttnn.matmul(
            input_tensor_a=ttnn_input,
            input_tensor_b=ttnn_weights,
            memory_config=output_memory_config,
            program_config=matmul_program_config,
        )

    passed, value, graphviz_output = run_with_graph_capture(lambda_func, torch_input, torch_weights, torch_output)
    print(graphviz_output)
    return passed


def test_1d_matmul_hs():
    torch_input = torch.rand(2 * 32, 8 * 32)
    torch_weights = torch.rand(8 * 32, 16 * 32)
    torch_output = torch_input @ torch_weights

    def lambda_func(device, torch_input, torch_weights):
        r_cores = 2
        c_cores = 2

        input_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                grid=ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(r_cores - 1, c_cores - 1))]
                ),
                shard_shape=[torch_input.shape[0] // (r_cores * c_cores), torch_input.shape[1]],
                shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
                halo=False,
            ),
        )

        output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                grid=ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(r_cores - 1, c_cores - 1))]
                ),
                shard_shape=[torch_input.shape[0] // (r_cores * c_cores), torch_weights.shape[1]],
                shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
                halo=False,
            ),
        )

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=input_memory_config
        )
        ttnn_weights = ttnn.from_torch(
            torch_weights,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        matmul_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(2, 2),
            in0_block_w=8 * 32,
            out_subblock_h=1,
            out_subblock_w=1,
            # out_block_h
            # out_block_w
            per_core_M=1,  # output_memory_config.shard_spec.shape[0] // 32,
            per_core_N=1,  # output_memory_config.shard_spec.shape[1] // 32,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        return ttnn.matmul(
            input_tensor_a=ttnn_input,
            input_tensor_b=ttnn_weights,
            memory_config=output_memory_config,
            program_config=matmul_program_config,
        )

    passed, value, graphviz_output = run_with_graph_capture(lambda_func, torch_input, torch_weights, torch_output)
    print(graphviz_output)
    return passed
