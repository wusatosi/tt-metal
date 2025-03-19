## Commands to run the unit tests of Yolov6x model:
All the unit test configurations passed.
- Conv2d: `pytest tests/ttnn/unit_tests/operations/conv/test_conv2d.py::test_conv_yolov6x`
- Relu: `pytest tests/ttnn/unit_tests/operations/eltwise/test_activation.py::test_relu_yolov6x`
- Concat:
  - For 2 inputs: `pytest tests/ttnn/unit_tests/operations/test_concat.py::test_concat_yolov6x_2inputs`
  - For 4 inputs: `pytest tests/ttnn/unit_tests/operations/test_concat.py::test_concat_yolov6x_4inputs`
- Maxpool2d: `pytest tests/ttnn/unit_tests/operations/pool/test_maxpool2d.py::test_run_max_pool_yolov6x`
- Convtranspose2d: `pytest tests/ttnn/unit_tests/operations/conv/test_conv_transpose2d.py::test_simple_conv_t2d_yolov6x`
