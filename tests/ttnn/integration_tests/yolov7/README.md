## Issue Description
In Yolov7 model, the indexing item assignment is not supported in TTNN ops. And the workaround is failing with low PCC.

OPS used: mul, sub, add, pow.

To run the test, use the following commands:
```
pytest tests/ttnn/integration_tests/yolov7/test_yolov7_unit.py
```

Expected to pass with good PCC but fails with low PCC.
