# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import sys
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

# required tracy report columns
filter_columns = [
    "CORE COUNT",
    "DEVICE KERNEL DURATION [ns]",
    "INPUT_0_Y",
    "INPUT_0_X",
    "ATTRIBUTES",
    "INPUT_0_DATATYPE",
    "INPUT_0_MEMORY",
    "OUTPUT_0_MEMORY",
]


# HELPER FUNCTIONS
def load_csv_data(path):
    return pd.read_csv(path)


def filter_data_frame(df, columns):
    f_df = df[df["OP CODE"] == "Reduce"]
    return f_df[columns]


def sort_by_cc(df):
    return df.sort_values(by=["CORE COUNT"], ascending=[True])


def parse_dim(entry):  # extract reduction dimension from ATTRIBUTES column
    attr = (((entry.split(";"))[5]).split(":", 1))[1]
    return attr.rsplit(":", 1)[1].strip("'")


def find_dim(df):
    df["DIM"] = df["ATTRIBUTES"].apply(parse_dim)
    return df


def find_tpc(df):
    df["TILE / CORE"] = df.apply(get_tile_count, axis=1)
    return df


def find_mem_layout(df):
    df.loc[:, "MEMORY TRANSFER"] = df.apply(mem_layout_type, axis=1)
    return df


def get_tile_count(row):
    if row["DIM"] == "H":
        return (row["INPUT_0_Y"] // 32) * (
            math.ceil((row["INPUT_0_X"] // 32) / row["CORE COUNT"])
        )  # some cores may do more work so we need to ceil
    else:
        return (row["INPUT_0_X"] // 32) * (math.ceil((row["INPUT_0_Y"] // 32) / row["CORE COUNT"]))


def mem_layout_type(row):
    ret = ""
    if "L1" in row["INPUT_0_MEMORY"]:
        ret += "L1 -> "
    else:
        ret += "DRAM -> "
    if "L1" in row["OUTPUT_0_MEMORY"]:
        ret += "L1"
    else:
        ret += "DRAM"
    return ret


def type_color(entry):
    if entry["DIM"] == "H":
        return "blue"
    else:
        return "red"


def type_color_steps(entry):
    if entry["DIM"] == "H":
        # to distinguish between different dram reading patterns
        dram_rotation = math.gcd(int(entry["INPUT_0_X"]) // 32, 12)
        if dram_rotation == 1:
            return "yellow"
        elif dram_rotation == 2:
            return "blue"
        elif dram_rotation == 3:
            return "green"
        elif dram_rotation == 4:
            return "purple"
        elif dram_rotation == 6:
            return "orange"
        else:
            return "red"
    else:
        return "red"


def get_dram_pattern(entry):
    return int(12 / math.gcd(int(entry["INPUT_0_X"]) // 32, 12))


def get_shape(row):
    return f"({row['INPUT_0_X']}, {row['INPUT_0_Y']})"


def tile_size_kB(dtype):
    if dtype == "FLOAT32":
        return 4
    elif dtype == "BFLOAT16":
        return 2
    else:
        return 1.0625


def calc_expected_time(tile_per_core, dtype):  # rough estimate of expected DRAM read time
    ts = tile_size_kB(dtype)  # kB
    core_read = tile_per_core * ts  # kB
    core_num = 64

    full_read = core_read * core_num  # kB
    dram_bw = 200  # GB/s
    return ((full_read / np.power(1024, 2)) / dram_bw) * np.power(10, 9)  # ns


def get_overworked_cores(row):  # number of cores that do more work than others
    if row["DIM"] == "H":
        return (row["INPUT_0_X"] // 32) % row["CORE COUNT"]
    else:
        return (row["INPUT_0_Y"] // 32) % row["CORE COUNT"]


# DATA PREPARATION
def prepare_df(df):
    df = filter_data_frame(df, filter_columns)
    df = sort_by_cc(df)
    df = find_dim(df)
    df = find_tpc(df)
    df = find_mem_layout(df)
    return df


# ANALYSIS
def calc_deviation(df):  # standard deviation of kernel duration per tile/core
    grouped_stats = df.groupby(["TILE / CORE"])["DEVICE KERNEL DURATION [ns]"].agg(["mean", "std"]).reset_index()
    grouped_stats = grouped_stats.rename(columns={"mean": "DURATION MEAN", "std": "DURATION STD"})
    grouped_stats["%"] = 100 * grouped_stats["DURATION STD"] / grouped_stats["DURATION MEAN"]
    grouped_stats = grouped_stats.sort_values(by=["%"], ascending=[False])
    print(grouped_stats)


# RW vs. CW PERFORMANCE
def rw_vs_cv(df_rw, df_cw):  # comparison of row-wise and column-wise core assignment when not all 64 cores are used
    df_rw = df_rw[
        [
            "INPUT_0_X",
            "INPUT_0_Y",
            "DIM",
            "INPUT_0_DATATYPE",
            "INPUT_0_MEMORY",
            "OUTPUT_0_MEMORY",
            "DEVICE KERNEL DURATION [ns]",
        ]
    ]
    df_cw = df_cw[
        [
            "INPUT_0_X",
            "INPUT_0_Y",
            "DIM",
            "INPUT_0_DATATYPE",
            "INPUT_0_MEMORY",
            "OUTPUT_0_MEMORY",
            "DEVICE KERNEL DURATION [ns]",
        ]
    ]

    merged_df = pd.merge(
        df_rw,
        df_cw,
        on=["INPUT_0_X", "INPUT_0_Y", "DIM", "INPUT_0_DATATYPE", "INPUT_0_MEMORY", "OUTPUT_0_MEMORY"],
        suffixes=("_rw", "_cw"),
    )
    merged_df["DIFF"] = merged_df["DEVICE KERNEL DURATION [ns]_cw"] - merged_df["DEVICE KERNEL DURATION [ns]_rw"]

    merged_df = merged_df.sort_values(by=["DIFF"], ascending=[False])
    print(merged_df)

    merged_df = merged_df[merged_df["DIFF"] < 0]
    print(merged_df)


# PLOTTING HELPER FUNCTIONS
def plot_per_tilecore(
    tilecore, dtype, mem_conf, df, path
):  # plot for each tile/core, different number of 'overworked' cores
    df["COLOR"] = df.apply(type_color, axis=1)
    plt.figure(figsize=(14, 6))
    plt.scatter(df["OVW NUM"], df["DEVICE KERNEL DURATION [ns]"], marker="o", color=df["COLOR"])

    plt.xlabel("Shape")
    plt.ylabel("Device Kernel Duration[ns]")
    plt.title(f"Scatter Plot for Data Type {dtype}, Mem. Config {mem_conf}, Tile/Core {tilecore}")

    plt.savefig(f"{path}/{tilecore}_{dtype}_{mem_conf}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_per_type_memconf_tilecore(dim, dtype, mem_conf, df, path, full):  # plot of tile/core impact on kernel duration
    df["COLOR"] = df.apply(type_color, axis=1)
    plt.figure(figsize=(8, 6))
    plt.scatter(df["TILE / CORE"], df["DEVICE KERNEL DURATION [ns]"], marker="o", color=df["COLOR"])

    if full and mem_conf.startswith("DRAM"):
        x_line = np.sort(df["TILE / CORE"].unique())
        plt.plot(x_line, calc_expected_time(x_line, dtype), color="red", label="DRAM BW")

    plt.xlabel("Number of Tiles per Core")
    plt.ylabel("Device Kernel Duration[ns]")
    plt.title(f"Scatter Plot for Data Type {dtype}, Mem. Config {mem_conf}")

    plt.savefig(f"{path}/{dim}_{dtype}_{mem_conf}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_per_type_memconf_tilecore_steps(
    dim, dtype, mem_conf, df, path
):  # plot of impact of worker core increase on kernel duration
    df["COLOR"] = df.apply(type_color_steps, axis=1)
    df["PATTERN"] = df.apply(get_dram_pattern, axis=1)
    plt.figure(figsize=(14, 6))

    if dim == "H":
        x_axis = df["INPUT_0_X"]
    else:
        x_axis = df["INPUT_0_Y"]

    for pattern, group in df.groupby("PATTERN"):
        plt.scatter(
            (group[x_axis] / 32), group["DEVICE KERNEL DURATION [ns]"], marker="o", color=group["COLOR"], label=pattern
        )

    plt.legend(title="Bank hits per column")
    plt.xlabel("Tensor Width(Tiles)")
    plt.ylabel("Device Kernel Duration[ns]")
    plt.title(f"Scatter Plot for Data Type {dtype}, Mem. Config {mem_conf}")

    plt.savefig(f"{path}/{dim}_{dtype}_{mem_conf}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()


# PLOT FUNCTIONS
def plot_df(df, path, full):  # plot tile/core impact per data type, dimension and memory configuration
    for (dtype, mem_conf, dim), group_df in df.groupby(["INPUT_0_DATATYPE", "MEMORY TRANSFER", "DIM"]):
        plot_per_type_memconf_tilecore(dim, dtype, mem_conf, group_df, path, full)


def plot_tilecore(
    df, path
):  # plot 'overworked' cores impact per data type, dimension, tile/core and memory configuration
    df["COLOR"] = df.apply(type_color, axis=1)
    df["OVW NUM"] = df.apply(get_overworked_cores, axis=1)

    for (dtype, mem_conf, tilecore, dim), group_df in df.groupby(
        ["INPUT_0_DATATYPE", "MEMORY TRANSFER", "TILE / CORE", "DIM"]
    ):
        plot_per_tilecore(tilecore, dtype, mem_conf, group_df, path)


def plot_full_grid_steps(
    df, path, full
):  # plot of worker core increase impact per data type, dimension and memory configuration
    for (dtype, mem_conf, dim), group_df in df.groupby(["INPUT_0_DATATYPE", "MEMORY TRANSFER", "DIM"]):
        plot_per_type_memconf_tilecore_steps(dim, dtype, mem_conf, group_df, path)


if __name__ == "__main__":
    # usage example:
    df = load_csv_data("perf_estimates/reduction/interleaved/example_non_full_grid.csv")
    df = prepare_df(df)
    plot_df(df, "perf_estimates/reduction/interleaved/plots", False)
