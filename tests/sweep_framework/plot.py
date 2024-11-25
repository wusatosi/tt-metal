import json
import pathlib
import csv
from collections import defaultdict
from framework.sweeps_logger import sweeps_logger as logger
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


SWEEPS_DIR = pathlib.Path(__file__).parent
TILE_SIZE = 32


def get_vectors(suite):
    with open(SWEEPS_DIR / "vectors_export" / "data_movement.reshard.reshard.json") as file:
        vectors = json.load(file)
    return {k: v for k, v in vectors[suite].items() if v["validity"] == "VectorValidity.VALID"}


def get_perfs(suite):
    with open(SWEEPS_DIR / "results_export" / "data_movement.reshard.reshard.json") as file:
        sweep_results = json.load(file)
    sweep_results = [
        sweep for sweep in sweep_results if sweep["status"] == "TestStatus.PASS" and sweep["suite_name"] == suite
    ]
    perfs = defaultdict(list)
    for sweep in sweep_results:
        perfs[sweep["vector_id"]].append(sweep["device_perf"]["DEVICE KERNEL DURATION [ns]"])
    return perfs


def get_mlr(results):
    x = []
    y = []
    for grid, result in results.items():
        grid_size = int(grid[16]) * int(grid[21])
        for size, runs in result.items():
            x.append([grid_size, size])
            y.append(np.mean(runs))
    x = np.asarray(x)
    y = np.asarray(y)
    reg = LinearRegression().fit(x, y)
    print(reg.score(x, y))
    return reg


def get_mlr_preds(reg: LinearRegression, x, grid):
    grid_size = int(grid[16]) * int(grid[21])
    x = np.asarray([[grid_size, _x] for _x in x])
    return reg.predict(x)


if __name__ == "__main__":
    suites = ["up-rect-1", "up-rect-2", "up-rect-0"]
    colours = ["b", "r", "g", "y", "m", "k"]

    # output shard : { size : [durations] }
    results = defaultdict(lambda: defaultdict(list))

    for suite in suites:
        vectors = get_vectors(suite)
        logger.info(f"Loaded {len(vectors)} valid vectors for {suite}")
        perfs = get_perfs(suite)
        logger.info(f"Got {len(perfs)} passing vector configs for {suite}")
        for vec_id, perf in perfs.items():
            output_core_grid = vectors[vec_id]["output_core_grid"]
            shape = vectors[vec_id]["shape"]
            size = int(shape[1:-1].split(",")[2]) * int(shape[1:-1].split(",")[3]) / (TILE_SIZE * TILE_SIZE)
            results[output_core_grid][size].extend([float(x) for x in perf])

    # reg = get_mlr(results)
    for i, [grid, result] in enumerate(results.items()):
        x = []
        y = []
        for size, runs in result.items():
            x.append(size)
            y.append(np.mean(runs))
        plt.scatter(x, y, color=colours[i], label=f"Sample from {grid}")
        plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), label=f"Fit to {grid}", color=colours[i], linestyle=":")
        # plt.plot(x, get_mlr_preds(reg, x, grid), label="Fit to all output grids", color="c", linestyle="--")
        plt.ylabel("Kernel Duration [ns]")
        plt.xlabel("Number of Tiles in Tensor (ttnn.bfloat16)")
        plt.title("Reshard Op Mean Duration Starting from a (4,4) Core Grid")
        plt.legend()
        plt.savefig(f"fits_{i}")
        plt.close()

    for i, [grid, result] in enumerate(results.items()):
        x = []
        y = []
        for size, runs in result.items():
            x.append(size)
            y.append(np.mean(runs))
        plt.scatter(x, y, label=grid, color=colours[i])
    plt.ylabel("Kernel Duration [ns]")
    plt.xlabel("Number of Tiles in Tensor (ttnn.bfloat16)")
    plt.title("Reshard Op Mean Duration Starting from a (4,4) Core Grid")
    plt.legend()
    plt.savefig("tile sizes")
    plt.close()

    with open("data.csv", "w") as file:
        writer = csv.DictWriter(file, fieldnames=["output_grid", "size", "mean_duration", "std", "std_over_mean"])
        writer.writeheader()
        for grid, result in results.items():
            output = []
            for size, runs in result.items():
                output.append(
                    {
                        "output_grid": grid,
                        "size": size,
                        "mean_duration": np.mean(runs),
                        "std": np.std(runs),
                        "std_over_mean": np.std(runs) / np.mean(runs),
                    }
                )
            output.sort(key=lambda x: x["size"])
            writer.writerows(output)

    for i, [grid, result] in enumerate(results.items()):
        x = []
        y = []
        for size, runs in result.items():
            x.append(size)
            y.append(np.mean(runs))
        plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), label=f"Fit to {grid}", color=colours[i], linestyle=":")
    plt.ylabel("Kernel Duration [ns]")
    plt.xlabel("Number of Tiles in Tensor (ttnn.bfloat16)")
    plt.title("Reshard Op Mean Duration Starting from a (4,4) Core Grid")
    plt.legend()
    plt.savefig("fits")
    plt.close()
