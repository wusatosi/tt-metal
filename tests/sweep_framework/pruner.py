import json
import pathlib
from collections import defaultdict
import os

SWEEPS_DIR = pathlib.Path(__file__).parent

with open(SWEEPS_DIR / "vectors_export" / "data_movement.reshard.reshard.json") as file:
    vectors = json.load(file)

new_vectors = defaultdict(dict)

for s in vectors:
    for v in vectors[s]:
        if vectors[s][v]["validity"] == "VectorValidity.VALID":
            new_vectors[s][v] = vectors[s][v]

if os.path.exists(SWEEPS_DIR / "results_export" / "data_movement.reshard.reshard.json"):
    with open(SWEEPS_DIR / "results_export" / "data_movement.reshard.reshard.json") as file:
        results = json.load(file)

DROP_EVERY_AGGRESSIVE = 2
DROP_EVERY_STD = 3

for suite in new_vectors:
    if suite in ["down-rect-1", "down-rect-2"] or suite.startswith("up-rect"):
        if len(new_vectors[suite]) > 1000:
            print(f"Pruning {suite}. Starting length: {len(new_vectors[suite])}")
            keys = list(new_vectors[suite].keys())
            keys_to_del = keys[::DROP_EVERY_AGGRESSIVE]
            for key in keys_to_del:
                del new_vectors[suite][key]
            print(f"After pruning {len(new_vectors[suite])}")
        elif len(new_vectors[suite]) > 500:
            print(f"Pruning {suite}. Starting length: {len(new_vectors[suite])}")
            keys = list(new_vectors[suite].keys())
            keys_to_del = keys[::DROP_EVERY_STD]
            for key in keys_to_del:
                del new_vectors[suite][key]
            print(f"After pruning {len(new_vectors[suite])}")

with open(SWEEPS_DIR / "vectors_export" / "data_movement.reshard.reshard.json.new", "w") as file:
    json.dump(new_vectors, file, indent=4)
