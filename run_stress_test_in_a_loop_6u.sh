#!/bin/bash
if [ -z "$1" ]; then
    echo "Please provide the machine id"
    exit 1
fi
machine_id=$1

export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=/mnt/MLPerf/divanovic/tt-metal
export PYTHONPATH=/mnt/MLPerf/divanovic/tt-metal
export LLAMA_DIR=/mnt/MLPerf/tt_dnn-models/llama/Llama3.1-70B-Instruct/

mkdir -p outputs
# run the command in loop
for ((i=0; i<5; i++)); do
    echo "Running iteration $i"
    # add timeout to the command
    FAKE_DEVICE=TG timeout 1200 pytest models/demos/llama3_subdevices/demo/demo_decode.py -k "stress-test and not mini-stress-test" > outputs/file_${machine_id}_${i}.txt 2>&1
    exit_code=$?

    if [ $exit_code -eq 124 ] || [ $exit_code -ne 0 ]; then
        echo "Iteration $i timed out"
        echo "Resetting the device"
        sudo ipmitool raw 0x30 0x8B 0xF 0xFF 0x0 0xF | sleep 180
        echo "Resetting the device 2nd time"
        sudo ipmitool raw 0x30 0x8B 0xF 0xFF 0x0 0xF | sleep 180
    fi
done
