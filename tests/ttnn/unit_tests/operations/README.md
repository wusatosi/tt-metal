## This commit contains Avg_Pool2d unit test for VoVNet model.

1. `test_run_average_pool2d_yolov9c` method in `tests/ttnn/unit_tests/operations/test_global_avg_pool2d.py` file will contain unit test for ttnn.global_avg_pool2d with Kernel_size 2.

    To run the test, use the following command : `pytest tests/ttnn/unit_tests/operations/test_global_avg_pool2d.py::test_run_average_pool2d_yolov9c`

## Expected Behaviour / Error(s):

### On WH(n150, n300):
For Input_shape: [1, 256, 80, 80], Expected Shape: [1, 256, 79, 79].
`E       AssertionError: list(expected_pytorch_result.shape)=[1, 256, 79, 79] vs list(actual_pytorch_result.shape)=[1, 256, 1, 1]`
