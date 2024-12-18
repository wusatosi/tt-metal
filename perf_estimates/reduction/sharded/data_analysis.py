# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import sys
import re
import json
import pandas as pd
import matplotlib.pyplot as plt

# required tracy report columns
filter_columns = [
    "CORE COUNT",
    "DEVICE KERNEL DURATION [ns]",
    "INPUT_0_Y",
    "INPUT_0_X",
    "ATTRIBUTES",
    "INPUT_0_DATATYPE",
]


# HELPER FUNCTIONS
def load_csv_data(path):
    return pd.read_csv(path)


def filter_data_frame(df, columns):
    f_df = df[df["OP CODE"] == "Reduce"]
    return f_df[columns]


def sort_by_shape(df):
    return df.sort_values(by=["INPUT_0_X", "INPUT_0_Y"], ascending=[True, True])


def parse_dim(entry):  # extract reduction dimension from ATTRIBUTES column
    attr = (((entry.split(";"))[5]).split(":", 1))[1]
    return attr.rsplit(":", 1)[1].strip("'")


def find_dim(df):
    df["DIM"] = df["ATTRIBUTES"].apply(parse_dim)
    return df


def parse_core_grid(entry):  # extract core grid shape from ATTRIBUTES column
    pattern = r"x=(\d+);y=(\d+)"
    m = re.findall(pattern, entry)
    grid_str = m[-1]
    return grid_str


def gen_core_grid_column(df):
    df["CORE GRID"] = df["ATTRIBUTES"].apply(parse_core_grid)
    return df


def tuple_to_string(tup):
    return f"(x={tup[0]}, y={tup[1]})"


def get_shape(row):
    return f"({row['INPUT_0_X']}, {row['INPUT_0_Y']})"


def get_tile_count(row):
    return (row["INPUT_0_Y"] * (row["INPUT_0_X"] // row["CORE COUNT"])) // (
        32 * 32
    )  # only height reduction supported for sharding


def get_dtype(entry):
    if entry == "FLOAT32":
        return 32
    elif entry == "BFLOAT16":
        return 16
    else:
        return 8


def type_color(entry):
    if entry == "FLOAT32":
        return "blue"
    elif entry == "BFLOAT16":
        return "red"
    else:
        return "yellow"


# STD PERCENTAGE CALCULATION
def deviation_percentage_tile(df):  # deviation percentage for same TpC number
    df["TILE / CORE"] = df.apply(get_tile_count, axis=1)
    grouped_stats = df.groupby(["TILE / CORE"])["DEVICE KERNEL DURATION [ns]"].agg(["mean", "std"]).reset_index()

    grouped_stats = grouped_stats.rename(columns={"mean": "DURATION MEAN", "std": "DURATION STD"})
    grouped_stats["%"] = 100 * grouped_stats["DURATION STD"] / grouped_stats["DURATION MEAN"]
    print(grouped_stats)


def runs_deviation(df):  # deviation percentage for several op runs
    grouped_stats = (
        df.groupby(["INPUT_0_X", "INPUT_0_Y", "INPUT_0_DATATYPE", "ATTRIBUTES"])["DEVICE KERNEL DURATION [ns]"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped_stats = grouped_stats.rename(columns={"mean": "DURATION MEAN", "std": "DURATION STD"})
    grouped_stats["%"] = 100 * grouped_stats["DURATION STD"] / grouped_stats["DURATION MEAN"]
    grouped_stats.sort_values(by=["%"], ascending=[False])
    print(grouped_stats)
    print(set(grouped_stats["count"].unique()))


def deviation_percentage(df):  # deviation percentage for different core grids
    df["SHAPE"] = df.apply(get_shape, axis=1)
    grouped_stats = (
        df.groupby(["CORE COUNT", "SHAPE"])["DEVICE KERNEL DURATION [ns]"].agg(["mean", "std"]).reset_index()
    )

    grouped_stats = grouped_stats.rename(columns={"mean": "DURATION MEAN", "std": "DURATION STD"})
    grouped_stats["%"] = 100 * grouped_stats["DURATION STD"] / grouped_stats["DURATION MEAN"]
    grouped_stats = grouped_stats.sort_values(by=["%"], ascending=[False])
    print(grouped_stats)


# PLOTTING HELPER FUNCTIONS
def plot_for_shape(shape, df, path):  # plot per grid shape
    plt.figure(figsize=(8, 6))
    plt.scatter(df["CORE COUNT"], df["DEVICE KERNEL DURATION [ns]"], color="blue", marker="o")

    plt.xlabel("Number of Cores")
    plt.ylabel("Device Kernel Duration[ns]")
    plt.title(f"Scatter Plot for {shape} Shape")

    plt.savefig(f"{path}/{shape}_shape.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_for_core_num(core_num, df, path):  # plot per core count
    plt.figure(figsize=(8, 6))
    shape = [str(tuple(pair)) for pair in zip(df["INPUT_0_X"], df["INPUT_0_Y"])]
    plt.scatter(shape, df["DEVICE KERNEL DURATION [ns]"], color="blue", marker="o")

    plt.xlabel("Input Shape")
    plt.ylabel("Device Kernel Duration[ns]")
    plt.title(f"Scatter Plot for {core_num} cores")

    plt.savefig(f"{path}/{core_num}_core_plot.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_grids_for_shape(shape, df, path):  # plot per grid shape
    df = df.sort_values(by=["CORE COUNT"], ascending=[True])
    df["CORE GRID STR"] = df["CORE GRID"].apply(tuple_to_string)
    plt.figure(figsize=(14, 10))
    plt.scatter(df["CORE GRID STR"], df["DEVICE KERNEL DURATION [ns]"], color="blue", marker="o")

    plt.xlabel("Grid Shape")
    plt.ylabel("Device Kernel Duration[ns]")
    plt.title(f"Scatter Plot for {shape} Shape")

    plt.savefig(f"{path}/{shape}_shape.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()


# PLOTTING FUNCTIONS
def plot_tile_per_core(df, path):  # plot kernel duration based on tiles per core
    df["TILE / CORE"] = df.apply(get_tile_count, axis=1)
    df["COLOR"] = df["INPUT_0_DATATYPE"].apply(type_color)
    plt.figure(figsize=(8, 6))
    plt.scatter(df["TILE / CORE"], df["DEVICE KERNEL DURATION [ns]"], marker="o", color=df["COLOR"])

    plt.xlabel("Number of Tiles per Core")
    plt.ylabel("Device Kernel Duration[ns]")
    plt.title(f"Scatter Plot for Device Kernel Duration")

    plt.savefig(f"{path}/tile_core.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_data(df, key, path):  # grouped plot
    g_df = df.groupby(key)
    for k, group in g_df:
        # plot_for_core_num(k, group)
        # plot_for_shape(k, group)
        plot_grids_for_shape(k, group, path)


# DATA PREPARATION
def prepare_df(df):
    df = filter_data_frame(df, filter_columns)
    df = gen_core_grid_column(df)
    df = sort_by_shape(df)
    return df


if __name__ == "__main__":
    # usage example:
    df = load_csv_data("perf_estimates/reduction/sharded/example.csv")
    df = prepare_df(df)
    plot_tile_per_core(df, "perf_estimates/reduction/sharded/plots")
