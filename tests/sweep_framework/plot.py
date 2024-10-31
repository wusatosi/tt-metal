import json
import pathlib
import csv
from collections import defaultdict
from framework.sweeps_logger import sweeps_logger as logger
import matplotlib.pyplot as plt
import numpy as np

SWEEPS_DIR = pathlib.Path(__file__).parent


def get_vectors():
    with open(SWEEPS_DIR / "vectors_export" / "data_movement.reshard.reshard.json") as file:
        vectors = json.load(file)
    return {k: v for k, v in vectors["nightly"].items() if v["validity"] == "VectorValidity.VALID"}


def get_perfs():
    with open(SWEEPS_DIR / "results_export" / "data_movement.reshard.reshard.json") as file:
        sweep_results = json.load(file)
    sweep_results = [sweep for sweep in sweep_results if sweep["status"] == "TestStatus.PASS"]
    perfs = defaultdict(list)
    for sweep in sweep_results:
        perfs[sweep["vector_id"]].append(sweep["device_perf"]["DEVICE KERNEL DURATION [ns]"])
    return perfs


if __name__ == "__main__":
    vectors = get_vectors()
    logger.info(f"Loaded {len(vectors)} valid vectors")
    perfs = get_perfs()
    logger.info(f"Got {len(perfs)} passing vector configs")
    # output shard : { size : [durations] }
    results = defaultdict(lambda: defaultdict(list))
    for vec_id, perf in perfs.items():
        output_core_grid = vectors[vec_id]["output_core_grid"]
        shape = vectors[vec_id]["shape"]
        size = int(shape[1:-1].split(",")[2]) * int(shape[1:-1].split(",")[3])
        results[output_core_grid][size].extend([float(x) for x in perf])

    colours = ["b", "r", "g"]
    for i, [grid, result] in enumerate(results.items()):
        x = []
        y = []
        for size, runs in result.items():
            x.append(size / (32 * 32))
            y.append(np.mean(runs))
        plt.scatter(x, y, label=grid, color=colours[i])
    plt.ylabel("Kernel Duration [ns]")
    plt.xlabel("Number of Tiles in Tensor (ttnn.bfloat16)")
    plt.title("Reshard Op Mean Duration Starting from a (1,1) Core Grid")
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
                        "size": size / (32 * 32),
                        "mean_duration": np.mean(runs),
                        "std": np.std(runs),
                        "std_over_mean": np.std(runs) / np.mean(runs),
                    }
                )
            output.sort(key=lambda x: x["size"])
            writer.writerows(output)

    colours = ["b", "r", "g"]
    for i, [grid, result] in enumerate(results.items()):
        x = []
        y = []
        for size, runs in result.items():
            x.append(size / (32 * 32))
            y.append(np.mean(runs))
        plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), label=grid, color=colours[i], linestyle=":")
    plt.ylabel("Kernel Duration [ns]")
    plt.xlabel("Number of Tiles in Tensor (ttnn.bfloat16)")
    plt.title("Reshard Op Mean Duration Starting from a (1,1) Core Grid")
    plt.legend()
    plt.savefig("fits")
    plt.close()
