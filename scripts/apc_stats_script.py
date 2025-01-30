# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import requests
import re
import os
from datetime import datetime
from datetime import timedelta
import json
import csv

# first set these variables
GITHUB_TOKEN = ""  # replace with your GitHub token
REPO_OWNER = "tenstorrent"
REPO_NAME = "tt-metal"
RUN_ID = ""  # set the ID of the GitHub APC run


# fetch log files
def fetch_logs(run_id):
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/runs/{RUN_ID}/jobs"

    all_jobs = []

    while url:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            all_jobs.extend(data["jobs"])

            # for pagination
            if "next" in response.links:
                url = response.links["next"]["url"]
            else:
                url = None
        else:
            print(f"failed to fetch jobs with status code: {response.status_code}")
            break

    # write the log files and store their paths
    log_files = []
    for job in all_jobs:
        job_id = job["id"]
        print(f"fetching logs for job: {job['name']} (ID: {job_id})")

        logs_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/jobs/{job_id}/logs"

        log_response = requests.get(logs_url, headers=headers)

        if log_response.status_code == 200:
            log_filename = f"job_{job_id}_logs.tar.gz"
            log_files.append(log_filename)
            with open(log_filename, "wb") as f:
                f.write(log_response.content)
            print(f"logs for job {job['name']} saved as {log_filename}")
        else:
            print(f"failed to fetch logs for job {job['name']} with status code: {log_response.status_code}")

    return log_files


# function to extract timestamp to match the ones in the log files and the format that python expects
def extract_timestamp(line):
    match = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3,7}Z)", line)
    if match:
        timestamp_str = match.group(1)
        timestamp_str_without_z = timestamp_str.rstrip("Z")

        frac_seconds = timestamp_str_without_z.split(".")[-1]
        if len(frac_seconds) > 6:
            timestamp_str_without_z = timestamp_str_without_z[:23]
        elif len(frac_seconds) < 6:
            timestamp_str_without_z = timestamp_str_without_z[:23] + "0" * (6 - len(frac_seconds))

        return timestamp_str_without_z + "Z"
    return None


# convert timestamp to datatime
def parse_timestamp(timestamp_str):
    timestamp_str_without_z = timestamp_str.rstrip("Z")
    try:
        return datetime.fromisoformat(timestamp_str_without_z)
    except ValueError as e:
        print(f"error parsing timestamp {timestamp_str}: {e}")
        return None


# Function to read and add the total time spent by all the log files
def calculate_total_time_spent(log_files):
    total_duration = timedelta()

    for log_file in log_files:
        if not os.path.exists(log_file):
            print(f"log file {log_file} does not exist.")
            continue

        file_duration = timedelta()
        start = None
        end = None

        with open(log_file, "r") as file:
            for line in file:
                timestamp_str = extract_timestamp(line)
                if timestamp_str:
                    timestamp = parse_timestamp(timestamp_str)
                    if timestamp:
                        if start is None or timestamp < start:
                            start = timestamp

                        if end is None or timestamp > end:
                            end = timestamp

        if start and end:
            file_duration = end - start
            total_duration += file_duration
        else:
            print(f"No valid timestamps found in log file: {log_file}")

    return total_duration


# funtion that returns distinct python tests from tests directory in each log file
def extract_python_tests(log_files):
    tests_dict = {}

    python_test_pattern = r"([a-zA-Z0-9_/]+\.py)"

    stats_pattern = r"=\s*\d+\s*passed,\s*\d+\s*skipped,\s*\d+\s*deselected,\s*\d+\s*warnings\s*in\s*\d+\.\d+s"

    for log_file in log_files:
        if not os.path.exists(log_file):
            print(f"Log file {log_file} does not exist.")
            continue

        python_tests = set()
        collecting_tests = False

        with open(log_file, "r") as file:
            for line in file:
                # start looking between short test summary line and before reaching the pattern above"
                if "short test summary info" in line:
                    collecting_tests = True
                    continue

                if re.search(stats_pattern, line):
                    break

                if collecting_tests:
                    matches = re.findall(python_test_pattern, line)
                    for match in matches:
                        if match.startswith("tests/"):
                            python_tests.add(match)

        if python_tests:
            tests_dict[log_file] = list(python_tests)

    return tests_dict


# function to calculate the time difference between two timestamps in seconds
def calculate_time_diff(start_time, end_time):
    if start_time and end_time:
        return (end_time - start_time).total_seconds()
    return 0.0


# find time for individual python tests
# search between the line that has test session starts and the one that has warnings summary
# compute the duration of each test by subtracting the timestamps of the lines with the start and the "PASSED" log output
def extract_individual_times_from_tests(log_files):
    test_times = {}

    for log_file in log_files:
        if not os.path.exists(log_file):
            print(f"Log file {log_file} does not exist.")
            continue

        with open(log_file, "r") as file:
            test_start_time = None
            current_test = None
            search_started = False
            current_test_total_time = 0.0

            for line in file:
                if "test session starts" in line:
                    search_started = True
                    continue

                if "warnings summary" in line and search_started:
                    search_started = False
                    continue

                if search_started:
                    timestamp_str = extract_timestamp(line)
                    if timestamp_str:
                        timestamp = parse_timestamp(timestamp_str)
                        if timestamp is None:
                            continue

                        test_match = re.search(r"(tests/.+?\.py)", line)
                        if test_match:
                            current_test = test_match.group(1)
                            test_start_time = timestamp
                            if current_test not in test_times:
                                test_times[current_test] = 0

                        if "PASSED" in line and current_test and test_start_time:
                            test_duration = timestamp - test_start_time
                            test_times[current_test] += test_duration.total_seconds()

                            test_start_time = None

    return test_times


# return total time of ops and percentage with respect to total APC
def calculate_ops_total_and_percentage(time, test_times_dict):
    input_time_seconds = time.total_seconds()
    total_time = sum(test_times_dict.values())

    percentage_time = (total_time / input_time_seconds) * 100 if input_time_seconds > 0 else 0

    return total_time, percentage_time


# group similar python tests (by name)
def group_tests_by_base_name(test_times):
    grouped_test_times = {}

    for test_file, time in test_times.items():
        base_name = test_file.split("/")[-1].replace(".py", "")

        if base_name in grouped_test_times:
            grouped_test_times[base_name] += time
        else:
            grouped_test_times[base_name] = time

    filtered_grouped_test_times = {k: v for k, v in grouped_test_times.items() if v > 0}
    return filtered_grouped_test_times


# convert a dictionary to CSV
def dict_to_csv(data_dict, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Test Name", "Time Spent (s)"])
        for key, value in data_dict.items():
            writer.writerow([key, value])


def main():
    # fetch the log files of the specific APC run
    log_files = fetch_logs(RUN_ID)

    # find the accumulated time of all log files
    total_time = calculate_total_time_spent(log_files)

    # return the paths for the python tests in each raw log file (only if in tests directory)
    python_tests_dict = extract_python_tests(log_files)

    # fetch the time of each python test
    individual_tests_time = extract_individual_times_from_tests(python_tests_dict)

    # group the tests by name
    grouped_dict = group_tests_by_base_name(individual_tests_time)
    # print(json.dumps(grouped_dict, indent=4))

    # calculate the percentage and total time of these python tests
    time, percentage = calculate_ops_total_and_percentage(total_time, grouped_dict)
    print(time, percentage)

    # save the duration of each test in a csv file
    dict_to_csv(grouped_dict, "/path_to_csv_file")


if __name__ == "__main__":
    main()
