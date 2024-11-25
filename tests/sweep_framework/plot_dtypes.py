import json
import pathlib
import csv
from collections import defaultdict
from framework.sweeps_logger import sweeps_logger as logger
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import os


SWEEPS_DIR = pathlib.Path(__file__).parent
TILE_SIZE = 32

DATUM_SIZE = {"DataType.BFLOAT16": 2, "DataType.FLOAT32": 4, "DataType.BFLOAT8_B": 1}


def relative_root_mean_squared_error(y_actual, y_predicted):
    squared_loss = np.sum(np.square(y_actual - y_predicted)) / np.sum(np.square(y_predicted)) / len(y_actual)
    return np.sqrt(squared_loss)


def root_mean_squared_relative_error(y_actual, y_predicted):
    squared_loss = np.sum(np.square(np.divide(y_actual - y_predicted, y_actual))) / len(y_actual)
    return np.sqrt(squared_loss)


def get_vectors(suite: str, path):
    with open(path) as file:
        vectors = json.load(file)
    return {k: v for k, v in vectors[suite].items() if v["validity"] == "VectorValidity.VALID"}


def group_vectors(vectors):  # -> dict[dict[dict]]
    grouped_vectors = {}
    for k, v in vectors.items():
        key = f"{v['input_shard_strategy']}-{v['input_shard_orientation']}-{v['output_shard_strategy']}-{v['output_shard_orientation']}"
        if key not in grouped_vectors:
            grouped_vectors[key] = {}
        grouped_vectors[key][k] = v
        v.pop("input_shard_strategy")
        v.pop("input_shard_orientation")
        v.pop("output_shard_strategy")
        v.pop("output_shard_orientation")

    return grouped_vectors


def add_tile_and_data_size(vectors):  # -> dict[dict]
    for v in vectors.values():
        v["tile_size"] = (
            int(v["shape"][1:-1].split(",")[2]) * int(v["shape"][1:-1].split(",")[3]) / (TILE_SIZE * TILE_SIZE)
        )
        v["data_size"] = v["tile_size"] * DATUM_SIZE[v["dtype"]] * TILE_SIZE * TILE_SIZE
    return vectors


def get_perfs(suite: str, path):  # -> dict[float]
    with open(path) as file:
        sweep_results = json.load(file)
    sweep_results = [
        sweep for sweep in sweep_results if sweep["status"] == "TestStatus.PASS" and sweep["suite_name"] == suite
    ]
    perfs = {}
    for sweep in sweep_results:
        perfs[sweep["vector_id"]] = float(sweep["device_perf"]["DEVICE KERNEL DURATION [ns]"])
    return perfs


def get_perfs_by_field(vectors, perfs, field_x: str):
    perfs_for_field = defaultdict(list)
    for vec_id, vec in vectors.items():
        perfs_for_field[vec[field_x]].append(perfs[vec_id])
    perfs_for_field = {k: np.mean(v) for k, v in perfs_for_field.items()}
    return perfs_for_field


def compare_dtypes(d1_vecs, d1_perfs, d2_vecs, d2_perfs):
    d1_by_size = get_perfs_by_field(d1_vecs, d1_perfs, "data_size")
    d2_by_size = get_perfs_by_field(d2_vecs, d2_perfs, "data_size")

    intersection = d1_by_size.keys() & d2_by_size.keys()
    relative_diffs = np.abs([(d1_by_size[k] - d2_by_size[k]) / d1_by_size[k] for k in intersection])
    return np.mean(relative_diffs), np.std(relative_diffs), len(intersection)


def get_stats(reg, x, y):
    return {
        "r^2": reg.score(x, y),
        "RRMSE": relative_root_mean_squared_error(y, reg.predict(x)),
        "RMSRE": root_mean_squared_relative_error(y, reg.predict(x)),
    }


def fit(vectors, perfs, field_x: str):
    x = []
    y = []

    for field, runs in get_perfs_by_field(vectors, perfs, field_x).items():
        x.append(field)
        y.append(runs)

    x = np.asarray(x)
    y = np.asarray(y)
    reg = LinearRegression().fit(x, y)
    return reg, get_stats(reg, x, y)


if __name__ == "__main__":
    comps = []
    for in_grid in sorted(os.listdir(SWEEPS_DIR / "results_export")):
        for i, res_file in enumerate(sorted(os.listdir(SWEEPS_DIR / "results_export" / in_grid))):
            out_grid = res_file.split("-")[-1]
            out_grid = out_grid[:-5]  # remove .json

            perfs_by_dtype = {}
            grouped_vecs_by_dtype = {}
            for dtype in DATUM_SIZE.keys():
                suite = f"{in_grid}-{out_grid}-{dtype}"
                print(suite)
                perfs_by_dtype[dtype] = get_perfs(suite, SWEEPS_DIR / "results_export" / in_grid / res_file)
                vecs = get_vectors(suite, SWEEPS_DIR / "vectors_export" / in_grid / res_file)
                vecs = {k: vecs[k] for k in vecs if k in perfs_by_dtype[dtype]}
                vecs = add_tile_and_data_size(vecs)
                grouped_vecs_by_dtype[dtype] = group_vectors(vecs)

            for group in grouped_vecs_by_dtype["DataType.BFLOAT16"]:
                try:
                    mean, std, count = compare_dtypes(
                        grouped_vecs_by_dtype["DataType.BFLOAT16"][group],
                        perfs_by_dtype["DataType.BFLOAT16"],
                        grouped_vecs_by_dtype["DataType.FLOAT32"][group],
                        perfs_by_dtype["DataType.FLOAT32"],
                    )
                    comps.append((in_grid, out_grid, group, "float16-float32", mean, std, count))
                except:
                    pass
            for group in grouped_vecs_by_dtype["DataType.BFLOAT16"]:
                try:
                    mean, std, count = compare_dtypes(
                        grouped_vecs_by_dtype["DataType.BFLOAT16"][group],
                        perfs_by_dtype["DataType.BFLOAT16"],
                        grouped_vecs_by_dtype["DataType.BFLOAT8_B"][group],
                        perfs_by_dtype["DataType.BFLOAT8_B"],
                    )
                    comps.append((in_grid, out_grid, group, "float16-float8", mean, std, count))
                except:
                    pass
            for group in grouped_vecs_by_dtype["DataType.BFLOAT8_B"]:
                try:
                    mean, std, count = compare_dtypes(
                        grouped_vecs_by_dtype["DataType.BFLOAT8_B"][group],
                        perfs_by_dtype["DataType.BFLOAT8_B"],
                        grouped_vecs_by_dtype["DataType.FLOAT32"][group],
                        perfs_by_dtype["DataType.FLOAT32"],
                    )
                    comps.append((in_grid, out_grid, group, "float32-float8", mean, std, count))
                except:
                    pass
    with open("dtypes.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "input_grid",
                "output_grid",
                "strategy-orientation",
                "dtypes",
                "mean_abs_rel_diff",
                "std_abs_rel_diff",
                "num_pts",
            ]
        )
        writer.writerows(comps)
    # for suite in suites:
    #     vectors = get_vectors(suite)
    #     logger.info(f"Loaded {len(vectors)} valid vectors for {suite}")
    #     perfs = get_perfs(suite)
    #     logger.info(f"Got {len(perfs)} passing vector configs for {suite}")
    #     for vec_id, perf in perfs.items():
    #         output_core_grid = vectors[vec_id]["output_core_grid"]
    #         shape = vectors[vec_id]["shape"]
    #         size = int(shape[1:-1].split(",")[2]) * int(shape[1:-1].split(",")[3]) / (TILE_SIZE * TILE_SIZE)
    #         results[output_core_grid][size].extend([float(x) for x in perf])

    # with open("data.csv", "w") as file:
    #     writer = csv.DictWriter(file, fieldnames=["output_grid", "size", "mean_duration", "std", "std_over_mean"])
    #     writer.writeheader()
    #     for grid, result in results.items():
    #         output = []
    #         for size, runs in result.items():
    #             output.append(
    #                 {
    #                     "output_grid": grid,
    #                     "size": size,
    #                     "mean_duration": np.mean(runs),
    #                     "std": np.std(runs),
    #                     "std_over_mean": np.std(runs) / np.mean(runs),
    #                 }
    #             )
    #         output.sort(key=lambda x: x["size"])
    #         writer.writerows(output)
