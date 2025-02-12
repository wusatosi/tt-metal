# Welcome
Thanks for participating in the Fabric/Multichip Programming Hackathon!

You chose the fabric hack track where you will optimize the 1D fabric.

This hackathon will be good for you if you are interested in some low level details for
how a basic fabric works OR you are interested in performance golfing on an existing
implementation.

## Testing your changes

### Performance reporting
Before you start making changes, it is recommended to get a baseline performance measurement to understand the improvements as your changes are made.

Run the following commands and collect the performance data. After each command performance data can be obtained in the following way:

- Run the test
- Open performance dump file: `$workspace_root/generated/profiler/...`
- Find the `MAIN-WRITE-ZONE` start and end lines.
- Calculate the difference between end and start: this is the total time spent in the main send loop of the basic test.


Tests:
`TT_METAL_DEVICE_PROFILER=1  ./build/test/ttnn/unit_tests_ttnn_ccl --gtest_filter="*BasicMcastThroughputTest_3_onehop"`

`TT_METAL_DEVICE_PROFILER=1  ./build/test/ttnn/unit_tests_ttnn_ccl --gtest_filter="*BasicMcastThroughputTest_3"`

### Correctness (don't introduce a race condition or bug;))
`source python_env/bin/activate`
`pytest /test_new_all_gather.py`
`pytest /test_reduce_scatter_async.py`


## Optimizations
Feel free to optimize the fabric in any way you'd like although some suggestions to look at first are outlined below.

### Optimization: Fabric Routing Packet Header Field Optimization
For smaller fabric sizes, the routing fields in the packet header can be optimized to save a few instructions in the packet header processing code.

The packet header definition is in `ttnn/cpp/ttnn/fabric/packet_header.hpp` as the `PacketHeader` struct. The `RoutingFields routing_fields` field defines how the packet is forwarded through the fabric.

The routing fields are processed primarily in `ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_transmission.hpp`.

For smaller fabric sizes (like single galaxy or t3k), we could change the encoding to look like the following:

Encoded the entire route into the packet header (Unified unicast / mcast encoding, per each hop), into sequence of bit pairs:
0b11 (or 3): Do both a write and forward the packet (WF).
0b10 (or 2): Do not write, but forward (0F).
0b01 (or 1): Write and do not forward (W0), i.e. terminal hop
0b00 (or 0): No operation (could be used for padding or error checking)

Example encoding:
For a multicast to 5 chips, routing_info :[0b11, 0b11, 0b11, 0b11, 0b01]
WFWFWFWFW0,
For a unicast 5 hops away, routing_info :[0b10, 0b10, 0b10, 0b10, 0b01]
0F0F0F0FW0

Updates to both the packet header definition and the processing code will be needed.

### Optimization: Limit use of volatiles
Volatiles are used more liberally than needed and force L1 loads for every field lookup in the packet header. This is highly inefficient.

Take a look at the main users of volatiles in `ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_transmission.hpp`.


### Optimization: Optimize the Main Control Loop
`run_fabric_edm_main_loop` in `ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_erisc_datamover.cpp` has a lot of branching. Consider simplifying the logic to minimize the per iteration time for idle iterations.

### Unguided Optimization:
If you'd like to freely explore and try optimizing elsewhere in the implementation, you can make use of the performance capture tools here:

https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/tracy_profiler.html

and

https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/device_program_profiler.html

Good luck!
