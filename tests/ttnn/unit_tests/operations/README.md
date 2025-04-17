Run the following command to generate perf sheet got the conv with block sharding :
`
./tt_metal/tools/profiler/profile_this.py -n unit_test -c "pytest tests/ttnn/unit_tests/operations/test_conv2d.py::test_conv_bs"
`

Run the following command to generate perf sheet got the conv with width sharding :
`
./tt_metal/tools/profiler/profile_this.py -n unit_test -c "pytest tests/ttnn/unit_tests/operations/test_conv2d.py::test_conv_ws"
`
