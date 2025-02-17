# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from loguru import logger
import pytest
import csv
from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config


from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG

FILE_NAME = PROFILER_LOGS_DIR / "test_ethernet_link_latency.csv"

if os.path.exists(FILE_NAME):
    os.remove(FILE_NAME)


def append_to_csv(file_path, header, data, write_header=True):
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists or write_header:
            writer.writerow(header)
        writer.writerows([data])


def get_device_freq():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    freq = deviceData["deviceInfo"]["freq"]
    return freq


def profile_results(sample_size, sample_count, channel_count, output_latency):
    freq = get_device_freq() / 1000.0
    logger.info(f"frequencye {freq}")
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    main_test_body_string = "MAIN-TEST-BODY"
    setup.timerAnalysis = {
        main_test_body_string: {
            "across": "device",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "ERISC", "zone_name": main_test_body_string},
            "end": {"core": "ANY", "risc": "ERISC", "zone_name": main_test_body_string},
        },
    }
    devices_data = import_log_run_stats(setup)
    device_0 = list(devices_data["devices"].keys())[0]
    device_1 = list(devices_data["devices"].keys())[1]

    # MAIN-TEST-BODY
    main_loop_cycle = devices_data["devices"][device_0]["cores"]["DEVICE"]["analysis"][main_test_body_string]["stats"][
        "Average"
    ]
    main_loop_latency = main_loop_cycle / freq / sample_count / channel_count
    bw = sample_size / main_loop_latency

    if output_latency:
        header = [
            "SAMPLE_SIZE",
            "Latency",
        ]
    else:
        header = [
            "SAMPLE_SIZE",
            "BW (B/c)",
        ]
    write_header = not os.path.exists(FILE_NAME)
    append_to_csv(
        FILE_NAME,
        header,
        [sample_size, main_loop_latency],
        write_header,
    )
    return main_loop_latency


@pytest.mark.parametrize("sample_count", [1])  # , 8, 16, 64, 256],
@pytest.mark.parametrize(
    "sample_size",
    [(16), (128), (256), (512), (1024), (2048), (4096), (8192), (16384)],
)  # , 1024, 2048, 4096],
@pytest.mark.parametrize(
    "channel_count",
    [1],
)
def test_unidirectional_erisc_bandwidth(sample_count, sample_size, channel_count):
    test_string_name = f"test_ethernet_send_data_microbenchmark - \
            sample_count: {sample_count}, \
                sample_size: {sample_size}, \
                    channel_count: {channel_count}"
    print(f"{test_string_name}")
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

    ARCH_NAME = os.getenv("ARCH_NAME")
    rc = os.system(
        f"TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_link_ping_latency_no_edm_{ARCH_NAME} \
                {sample_count} \
                {sample_size} \
                {channel_count} "
    )
    if rc != 0:
        logger.info("Error in running the test")
        assert False

    main_loop_latency = profile_results(sample_size, sample_count, channel_count, sample_count == 1)
    logger.info(f"sender_loop_latency {main_loop_latency} cycles")

    return True
