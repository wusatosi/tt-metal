#!/bin/bash
if [ -z "$1" ]; then
    echo "Please provide the machine number"
    exit 1
fi

mkdir -p ~/mytemp
export TEMP=~/mytemp

echo $1
# echo $2
machine_number=$1
# start_iter=$2

# Used only for aus-glx-N machines, can't be applied to CI machines
if [[ $machine_number -lt 10 ]]; then
    machine_number="0$machine_number"
fi

work_dir="/home/$USER/tt-metal"

export LLAMA_DIR=/proj_sw/user_dev/llama33-data/Llama3.3-70B-Instruct
export ARCH_NAME=wormhole_b0
export HOME=/home/divanovic
export TT_METAL_ENABLE_ERISC_IRAM=1
export TT_METAL_HOME=$work_dir
export PYTHONPATH=$work_dir
export TT_METAL_ENV=dev
export PYTHONPATH=${PYTHONPATH}:${TT_METAL_HOME}

cd $work_dir
mkdir -p outputs

source python_env/bin/activate
# run the command in loop
for ((i=0; i<5; i++)); do
    echo "Running iteration $i"
    # add timeout to the command
    FAKE_DEVICE=TG timeout 1800 pytest models/demos/llama3_subdevices/demo/demo_decode.py -k "nd-hang" > outputs/file_${machine_number}_${i}.txt 2>&1
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

# Run stress test for 200K iterations
FAKE_DEVICE=TG timeout 5800 pytest models/demos/llama3_subdevices/demo/demo_decode.py -k "stress-test and not mini" > outputs/file_stress_${machine_number}.txt 2>&1
exit_code=$?

if [ $exit_code -eq 124 ] || [ $exit_code -ne 0 ]; then
    echo "Iteration $i timed out"
    echo "Resetting the device"
    cd ..
    echo "./tt-smi-3.0.3 -r reset_${machine_number}.json"
    ./tt-smi-3.0.3 -r reset_${machine_number}.json
    cd tt-metal
fi
