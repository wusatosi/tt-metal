# Reshard Modeling

## Goal
Develop a set of models that predict the runtime of the reshard op given a particular prametrization.

## High Level Approach
- The scope of modeling is limited to tiled tensors in L1. Transfers to/from DRAM are out of scope
- For a fixed { input core grid, output core grid, input shard strategy, input shard orientation, output shard strategy, output shard orientation }, the runtime of reshard is linear in the size of the tensor (in tiles)
- The sweep framework provides infrastructure to run an op with various parameters and collect device performance
- This branch includes modifications to the sweep framework to enable generating and running a large number of sweeps

## The Reshard Sweep
For a given reshard parametrization, profiling reshard is as simple as allocating a tensor in the input memory config, and calling `to_memory_config()` to the output config. We repeat this 10 times and take the average *kernel duration*, as reported by the device profiler. This happens in [`reshard.py:run()`](./sweeps/data_movement/reshard/reshard.py).

We take some care to avoid tensors that will not fit in L1, but this is only a best effort and some tests will hit L1 out of memory errors or other errors. Manually deallocating the tensors is *critical*, as the garbage collector is not always aggressive enough to clean the out of scope tensors in between reshards.

## Generating Sweep Vectors
Due to the sheer number of sweep vectors for this profiling effort, vectors are generated somewhat differently than a normal suite.
1. Sweep vectors are written to disk and *not* sent to elasticsearch. This is enabled using the `--dump-file` command line arg to [`sweeps_parameter_generator.py`](./sweeps_parameter_generator.py)
2. The vectors are written to `./vectors_export/`. This directory is approximately 2.3GB on disk. The directory structure is as follows:
```
vectors_export/
    1x1/    # all reshards from a 1x1 core grid
        reshard-1x1-1x2.json    # all reshards from a 1x1 core grid to a 1x2 core grid
        reshard-1x1-1x3.json    # all reshards from a 1x1 core grid to a 1x3 core grid
        ...
    1x2/    # all reshards from a 1x2 core grid
        ...
    ...
```
3. Within each `reshard-${input_grid}-${output_grid}.json` file, there are vectors testing different tensor shapes for different input/output sharding strategies and orientations. These are all in one suite.
4. The strategy for sufficiently varied sweeps is:
    1. Generate all possible tensor shapes { (TILE_SIZE \* x, TILE_SIZE \* y) such that 0<x<250 and 0<y<250 }
    2. For a given input core grid, output core grid, input shard strategy, and output shard strategy
        1. Invalidate all tensors that are not tileable
        2. Invalidate all tensors that are guaranteed to not fit in L1
5. Unlike regular suites, we do not write out invalidated vectors. The generation process is IO bound, so limiting interactions with the filesystem improves runtime.
6. We downsample suites that have more than 1600 valid vectors. This is done by repeatedly deleting every other valid vector until the total remaining is less than 1600

To run:
```
python tests/sweep_framework/sweeps_parameter_generator.py --module-name data_movement.reshard.reshard --dump-file -x <input_core_grid_x> -y <input_core_grid_starting_y>
```

e.g. setting `-x` to 2 and `-y` to 3 will generate all sweeps for input core grids: 2x3, 2x4, 2x5, 2x6, 2x7, 2x8

## Running the Sweeps
Generated sweep vectors are executed by running [`sweeps_runner.py`](./sweeps_runner.py).
```
python tests/sweep_framework/sweeps_runner.py --module-name data_movement.reshard.reshard --read-file <vector_file> --device-perf --result-file <output_file>.json --card <card_id>
```
This will execute all suites in `<vector_file>` and write the results in `<output_file>`. The `<card_id>` is only used if the card hangs, in which case the script will automatically call tt-smi and resume running vectors from the last good result.

There is a separate runner script, [`runner.py`](./runner.py) which will repeatedly run the above command execute sweeps for a series of core grids. Using this runner script will organize the results in a similar directory stucture as the sweep vectors under `./results_export_batch/`
```
python tests/sweep_framework/runner.py -x <input_core_grid_x> --start-y <input_core_grid_starting_y> --end-y <input_core_grid_ending_y> --runs 1
```
Setting `-x` to 3, `--start-y` to 4, and `--end-y` to 9 will run all sweeps for input core grids: 3x4, 3x5, 3x6, 3x7, 3x8. `--runs` is obsolete and should be set to 1.

This modeling effort requires running 100,000s of vectors, 10 times each! We used 2-3 cards for a few days to generate this data. As of this writing, this script can execute ~14 vectors/s with each vector being run 10 times. This bottleneck is specifically due to the `get_device_data_generate_report()` call to read and parse the device profiler data. Empirical testing this function has an overhead of ~7ms to read the profile for each kernel executed.

The [`sweeps_runner.py`](./sweeps_runner.py) on this branch incorporates several significant changes for improved throughput compared to main:
1. The interprocess communication has been rewritten. On the main branch, vectors are fed to the executor process one by one. Here, all sweep vectors are fed in one batch. The main process handles hangs/errors by tracking how many results have been received and resending the appropriate number of vectors.
2. The python stdlib interprocess queues have been replaces with [`faster-fifo`](https://github.com/alex-petrenko/faster-fifo)
3. For a given vector, the device profiler stats are read only after all 10 executions. The runner will aggregate all 10 runs and record the average and standard deviation instead of the raw output.
4. Instead of executing vectors one by one, we execute a `chunk` of vectors prior to reading the device profiler stats. The size of the chunk is controlled by `chunk_size` and needs to be tweaked on each system for optimal performance.

3 and 4 cause some complexity when interacting with the device profiler data, and we need to take care that we are associating the results with the right sweep vectors. The main complexity comes from having to deal with vectors that either hit an exception, or timed out. I have verified that the results with batch execution match the results of one by one execution for reshard, but this feature is not robust enough to be merged into main.

## Fitting Models
To fit the linear models, simply run `fit_models.py`. This script assumes the directory structure for **both results and vectors** are exactly as described in previous sections. At a high level this script:
1. finds the corresponding sweep vector for each result to obtain its shape
2. groups the results in each result file into groups that all have the same input/output shard strategy and orientation in addition to input/output core grid
3. fits a linear regression model to *each group* with the indepenent variable being tensor size (in tiles) and the dependent variable being the kernel duration
4. calculates `R^2`, `RRMSE`, and `RMSRE` for each model
5. dumps all model coefficients and error measurements into csv files

The generated csv files are attached. The data underlying these models was generated using x1 wormhole_b0 cards
