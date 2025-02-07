import re
import pytest
import csv
import sys
import argparse
from colorama import Fore, Style


# d = "'config_': 'SlidingWindowConfig(batch_size=1; input_hw=(288;288); window_hw=(5;5); stride_hw=(1;1); pad_hw=(0;0); dilation_hw=(1;1); num_cores_nhw=64; num_cores_c=1; core_range_set_={[(x=0;y=0) - (x=7;y=7)]})'; 'is_out_tiled_': 'true'; 'max_out_nsticks_per_core_': '2456'; 'output_memory_config_': 'MemoryConfig(memory_layout=TensorMemoryLayout::HEIGHT_SHARDED;buffer_type=BufferType::L1;shard_spec=ShardSpec(grid={[(x=0;y=0) - (x=7;y=7)]};shape={1312; 16};orientation=ShardOrientation::ROW_MAJOR;mode=ShardMode::PHYSICAL;physical_shard_shape=std::nullopt))'; 'pad_val_': '0'; 'parallel_config_': 'ParallelConfig(grid={[(x=0;y=0) - (x=7;y=7)]}; shard_scheme=HEIGHT_SHARDED; shard_orientation=ROW_MAJOR)'; 'remote_read_': 'false'; 'reshard_num_cores_nhw_': '0'; 'transpose_mcast_': 'false'"
def parse_config_string(config_string: str) -> dict:
    config_dict = {}
    pattern = re.compile(r"'(\w+)'[:=]\s*'([^']*)'")
    matches = pattern.findall(config_string)
    for key, value in matches:
        if value.lower() == "true":
            config_dict[key] = True
        elif value.lower() == "false":
            config_dict[key] = False
        elif value.isdigit():
            config_dict[key] = int(value)
        else:
            config_dict[key] = value

    for key, value in config_dict.items():
        print(Fore.GREEN + f"{key}" + Style.RESET_ALL + f" = {value}")

    return config_dict


@pytest.mark.parametrize(
    "config_string",
    [
        "{'config_': 'SlidingWindowConfig(batch_size=1; input_hw=(288;288); window_hw=(5;5); stride_hw=(1;1); pad_hw=(0;0); dilation_hw=(1;1); num_cores_nhw=64; num_cores_c=1; core_range_set_={[(x=0;y=0) - (x=7;y=7)]})'; 'is_out_tiled_': 'true'; 'max_out_nsticks_per_core_': '2456'; 'output_memory_config_': 'MemoryConfig(memory_layout=TensorMemoryLayout::HEIGHT_SHARDED;buffer_type=BufferType::L1;shard_spec=ShardSpec(grid={[(x=0;y=0) - (x=7;y=7)]};shape={1312; 16};orientation=ShardOrientation::ROW_MAJOR;mode=ShardMode::PHYSICAL;physical_shard_shape=std::nullopt))'; 'pad_val_': '0'; 'parallel_config_': 'ParallelConfig(grid={[(x=0;y=0) - (x=7;y=7)]}; shard_scheme=HEIGHT_SHARDED; shard_orientation=ROW_MAJOR)'; 'remote_read_': 'false'; 'reshard_num_cores_nhw_': '0'; 'transpose_mcast_': 'false'}",
        "{'block_config': 'OptimizedConvBlockConfig(act_block_h_ntiles=40;act_block_w_ntiles=5;out_subblock_h_ntiles=8;out_subblock_w_ntiles=1)'; 'compute_kernel_config': 'WormholeComputeKernelConfig(math_fidelity=HiFi4;math_approx_mode=1;fp32_dest_acc_en=0;packer_l1_acc=0;dst_full_sync_en=0)'; 'dtype': 'DataType::BFLOAT16'; 'enable_act_double_buffer': 'false'; 'enable_split_reader': 'false'; 'enable_subblock_padding': 'false'; 'enable_weights_double_buffer': 'false'; 'fuse_relu': 'true'; 'has_bias': 'true'; 'input_tensor_shape': '{1; 288; 288; 4}'; 'memory_config': 'MemoryConfig(memory_layout=TensorMemoryLayout::HEIGHT_SHARDED;buffer_type=BufferType::L1;shard_spec=ShardSpec(grid={[(x=0;y=0) - (x=7;y=7)]};shape={1280; 32};orientation=ShardOrientation::ROW_MAJOR;mode=ShardMode::PHYSICAL;physical_shard_shape=std::nullopt))'; 'output_channels': '32'; 'parallelization_config': 'OptimizedConvParallelizationConfig(grid_size=(x=8;y=8);num_cores_nhw=64;num_cores_c=1;per_core_out_matrix_height=1280;per_core_out_matrix_width=32)'; 'sliding_window_config': 'SlidingWindowConfig(batch_size=1; input_hw=(288;288); window_hw=(5;5); stride_hw=(1;1); pad_hw=(0;0); dilation_hw=(1;1); num_cores_nhw=64; num_cores_c=1; core_range_set_={[(x=0;y=0) - (x=7;y=7)]})'; 'untilize_out': 'false'; 'use_shallow_conv_variant': 'false'}",
    ],
)
def test_parse_config_string(config_string):
    conv2d_config = parse_config_string(config_string)
    print(conv2d_config.keys())


def main(csv_file):
    with open(csv_file, mode="r") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            print(
                Fore.RED
                + Style.BRIGHT
                + f'{row["OP CODE"]} ({row["GLOBAL CALL COUNT"]})'
                + Style.RESET_ALL
                + f' {row["DEVICE KERNEL DURATION [ns]"]}'
            )
            parse_config_string(row["ATTRIBUTES"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse configuration from a CSV file.")
    parser.add_argument("csv_file", help="Path to the CSV file")
    args = parser.parse_args()

    main(args.csv_file)
