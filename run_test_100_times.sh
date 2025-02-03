#!/bin/bash
for i in {1..100}
do
    WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest --disable-warnings models/demos/wormhole/resnet50/demo/demo.py::test_demo_imagenet
done
