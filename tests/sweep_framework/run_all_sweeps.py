import os
import glob
import pandas as pd
import pathlib
from pathlib import Path
import importlib
import time
from tests.sweep_framework.framework.permutations import *
import ttnn
from datetime import datetime

tests_dir = "/home/ubuntu/tt-metal"
os.chdir(tests_dir)

device_id = 0

sweeps_path = pathlib.Path(__file__).parent / "sweeps"
print(sweeps_path)


start = datetime.now()
start_time = start.strftime("%H:%M:%S")

for file in sorted(sweeps_path.glob("**/*.py")):
    sweep_name = str(pathlib.Path(file).relative_to(sweeps_path))[:-3].replace("/", ".")
    test_module = importlib.import_module("sweeps." + sweep_name)

    print(f"Running sweep {test_module}")

    for suite in test_module.parameters.keys():
        device = ttnn.open_device(device_id=device_id)
        print(f"Running suite {suite}: ")
        suite_vectors = list(permutations(test_module.parameters[suite]))

        for vector in suite_vectors:
            if test_module.invalidate_vector(vector)[0]:
                continue
            try:
                passed, _ = test_module.run(**vector, device=device)
            except Exception as e:
                pass
        ttnn.close_device(device)

end = datetime.now()
end_time = end.strftime("%H:%M:%S")


with open(os.path.join(os.getcwd(), "sweeps_run_time.txt"), "w") as f:
    f.write(f"{start_time}\n")
    f.write(end_time)
