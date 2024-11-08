import subprocess
import argparse
import pathlib
import os
import itertools


def get_vector_paths(x, y):
    VECTORS_FOLDER = pathlib.Path(__file__).parent / "vectors_export" / f"{x}x{y}"
    return [p for p in os.listdir(VECTORS_FOLDER) if p.endswith("json")]


def get_results_path(in_x, in_y, out_x, out_y):
    RESULTS_DIR = pathlib.Path(__file__).parent / "results_export" / f"{in_x}x{in_y}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR / f"reshard-{in_x}x{in_y}-{out_x}x{out_y}.json"


def get_out_grid(path):
    return int(path[-8]), int(path[-6])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", type=int)
    parser.add_argument("--start-y", type=int)
    parser.add_argument("--end-y", type=int)
    parser.add_argument("--runs", type=int)
    args = parser.parse_args()

    if args.x < 1 or args.x > 9:
        exit(1)
    if args.start_y >= args.end_y:
        exit(1)
    if args.start_y < 1 or args.end_y > 9:
        exit(1)

    for in_y in range(args.start_y, args.end_y):
        for vec_file in get_vector_paths(args.x, in_y):
            out_x, out_y = get_out_grid(vec_file)
            results_path = get_results_path(args.x, in_y, out_x, out_y)
            print(f"({args.x}, {in_y}) -> ({out_x}, {out_y}) : {vec_file} -> {results_path}")
