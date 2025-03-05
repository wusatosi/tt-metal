## ttnn.Conv2d Fails with "Statically allocated circular buffers issue" for larger in_channel size = 2560

### Run the following command to run the test
```
pytest tests/ttnn/integration_tests/yolov10/test_yolov10x_conv_test.py
```

The test is expected to pass as it passes in unit tests, but fails with model pipeline.
