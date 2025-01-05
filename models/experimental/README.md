Among 7 input configurations of Group Norm, 4 of the cases are failing with Out of Memory issue
```
E       RuntimeError: TT_THROW @ ../tt_metal/impl/allocator/allocator.cpp:145: tt::exception
E       info:
E       Out of Memory: Not enough space to allocate 35651584 B L1 buffer across 64 banks, where each bank needs to store 557056 B
```
Whereas 3 input configurations of Group Norm are failing with the following error:
```
E       RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_op.cpp:40: a.get_legacy_shape()[3] == gamma.value().get_legacy_shape()[3]
E       info:
E       256 != 32
```

Run the command to test: `pytest tests/ttnn/unit_tests/operations/test_group_norm.py::test_sd35_vae_512`
