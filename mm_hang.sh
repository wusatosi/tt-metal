#!/bin/bash

for i in $(seq 1 10);
do
    # pytest tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_fmod.py
    # pytest tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_frac.py
    # pytest tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_glu_variants.py
    # pytest tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_layernorm_sharded.py
    TT_METAL_CLEAR_L1=1 pytest tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_matmul.py
done
