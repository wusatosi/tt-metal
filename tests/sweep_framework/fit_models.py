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
RESULTS_FOLDER = "results_export_batch"
TILE_SIZE = 32

DATUM_SIZE = {"DataType.BFLOAT16": 2}


def relative_root_mean_squared_error(y_actual, y_predicted):
    rmse = np.sqrt(np.mean((y_actual - y_predicted) ** 2))
    return rmse / np.mean(y_actual)


def root_mean_squared_relative_error(y_actual, y_predicted):
    relative_errors = (y_actual - y_predicted) / y_actual
    return np.sqrt(np.mean(relative_errors**2))


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


def get_stats(reg, x, y):
    return {
        "r^2": reg.score(x, y),
        "RRMSE": relative_root_mean_squared_error(y, reg.predict(x)),
        "RMSRE": root_mean_squared_relative_error(y, reg.predict(x)),
        "num_points": len(x),
    }


def fit(vectors, perfs, field_x: str):
    x = []
    y = []

    for field, runs in get_perfs_by_field(vectors, perfs, field_x).items():
        x.append(float(field))
        y.append(runs)

    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)
    reg = LinearRegression().fit(x, y)
    return reg, get_stats(reg, x, y)


if __name__ == "__main__":
    results = []
    for in_grid in sorted(os.listdir(SWEEPS_DIR / RESULTS_FOLDER)):
        for i, res_file in enumerate(sorted(os.listdir(SWEEPS_DIR / RESULTS_FOLDER / in_grid))):
            out_grid = res_file.split("-")[-1]
            out_grid = out_grid[:-5]  # remove .json

            perfs_by_dtype = {}
            grouped_vecs_by_dtype = {}
            for dtype in DATUM_SIZE.keys():
                suite = f"{in_grid}-{out_grid}-{dtype}"
                print(suite)
                perfs_by_dtype[dtype] = get_perfs(suite, SWEEPS_DIR / RESULTS_FOLDER / in_grid / res_file)
                vecs = get_vectors(suite, SWEEPS_DIR / "vectors_export" / in_grid / res_file)
                vecs = {k: vecs[k] for k in vecs if k in perfs_by_dtype[dtype]}
                vecs = add_tile_and_data_size(vecs)
                grouped_vecs_by_dtype[dtype] = group_vectors(vecs)
            for dtype in DATUM_SIZE.keys():
                for group, vecs in grouped_vecs_by_dtype[dtype].items():
                    reg, stats = fit(vecs, perfs_by_dtype[dtype], "tile_size")
                    results.append(
                        {
                            "in_grid": in_grid,
                            "out_grid": out_grid,
                            "group": group,
                            "coef": reg.coef_[0],
                            "intercept": reg.intercept_,
                        }
                        | stats
                    )
    # print(json.dumps(results, indent=2))
    for i in range(12):
        with open(f"models.csv.{i}", "w") as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results[i * len(results) // 12 : (i + 1) * len(results) // 12])
