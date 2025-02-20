git checkout divanovic/old_commit_fix_non_contiguous_cb_indices

./build_metal.sh -p
# run the test and save to a cbs_with_opts.log file
FAKE_DEVICE=TG python -m tracy -r -m  "pytest  models/demos/llama3/demo/demo.py::test_llama_demo[wormhole_b0-True-mesh_device0-device_params0-performance-batch-1]" > cbs_with_opts.log 2>&1

# do git revert of specific commit
git revert -m "#0: Revert contiguous CB ids" 8080304b66

# build the env
./build_metal.sh -p
# run the test and save to a cbs_without_opts.log file
FAKE_DEVICE=TG TT_METAL_LOGGER_LEVEL=ERROR python -m tracy -r -m  "pytest  models/demos/llama3/demo/demo.py::test_llama_demo[wormhole_b0-True-mesh_device0-device_params0-performance-batch-1]" > cbs_without_opts.log 2>&1

# run both scripts
