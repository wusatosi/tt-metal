## Yolov8s_world Conv issue

While trying to run convolution test_cases with activation enabled("silu") and auto_shard=True. Some of the cases of yolov8s_world convolution configurations are failing due to low pcc.

To run the test use the below command,
`pytest tests/ttnn/unit_tests/operations/conv/test_conv2d.py::test_conv_for_yolov8s_world`
