#!/bin/bash
if [ -z "$1" ]; then
    echo "Please provide the machine number"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Please provide the start iteration number"
    exit 1
fi

machine_number=$1
start_iter=$2

if [[ $machine_number -lt 10 ]]; then
    machine_number="0$machine_number"
fi

mkdir -p outputs
# run the command in loop
for ((i=$start_iter; i<6; i++)); do
    echo "Running iteration $i"
    # add timeout to the command
    FAKE_DEVICE=TG timeout 1800 pytest models/demos/llama3_subdevices/demo/demo_decode.py -k full > outputs/file_${machine_number}_${i}.txt 2>&1
    exit_code=$?

    if [ $exit_code -eq 124 ] || [ $exit_code -ne 0 ]; then
        echo "Iteration $i timed out"
        echo "Resetting the device"
        cd ..
        echo "./tt-smi-3.0.3 -r reset_${machine_number}.json"
        ./tt-smi-3.0.3 -r reset_${machine_number}.json
        cd tt-metal
    fi
done
