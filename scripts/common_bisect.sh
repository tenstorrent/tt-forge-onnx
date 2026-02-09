

#!/bin/bash
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

: << 'COMMENT'
SAMPLE:
Script run command : bash ./scripts/common_bisect.sh

INPUTS:
Enter Pytest Command:
pytest forge/test/models/pytorch/vision/mgp_str_base/test_mgp_str_base.py::test_mgp_scene_text_recognition -vss
Enter Passing Commit Id:
76142ec0cd6e89ce1accc280c8262bc4fd4e2b65
Enter Failing Commit Id:
5395a6cd67bd7b0ce5e0ef5584f5e1b0bc520b80


COMMENT

declare -A data_inputs
declare -A data_index
tempfile=$(mktemp)

# Getting inputs from the user
get_inputs() {
    local pytest_cmd
    read -p "Enter Pytest Command: " pytest_cmd
    read -p "Enter Passing Commit Id: " pass_id
    read -p "Enter Failing Commit Id: " fail_id

    echo "$pytest_cmd,$pass_id,$fail_id"
}

# set all environmental flags
set_evn_flags() {
    export HF_TOKEN="your access token" # Replace with your huggingface token
    export PYTEST_ADDOPTS=" -svv"
    export IRD_LF_CACHE=http://aus2-lfcache.aus2.tenstorrent.com/
    export TTMLIR_PYTHON_VERSION=python3.12
    export TTFORGE_PYTHON_VERSION=python3.12
}

# If any build issues, it will show build error and exit
error_handling() {
    if [ $? -ne 0 ]; then
        local stage="$2"
        echo "Error: $stage  Command failed"
        exit 1
    fi
}

# Resetting machine before test runs
reset() {
    echo "wormhole_b0 resetting..."
    /home/software/syseng/wh/tt-smi -lr all wait -er >/dev/null 2>&1
    error_handling "$?" "Reset"
}

# Clean previous all cacche and build folder. Build based on the architecture
env_clean_and_build() {
    git submodule update --init --recursive -f >/dev/null 2>&1
    echo "Submodules Updated"
    if [ -d "build" ]; then
        echo "Build directory exists. Doing a clean up..."
        ./clean_build.sh >/dev/null 2>&1
        error_handling "$?" "Clean"
        echo "Build and cache is cleaned!"
    fi
    source "env/activate"
    cmake -B env/build env >/dev/null 2>&1
    cmake --build env/build >/dev/null 2>&1
    error_handling "$?" "Build FFE"
    echo "FFE Build is Successfully"
    cmake -G Ninja -B build -DCMAKE_CXX_COMPILER=clang++-17 -DCMAKE_C_COMPILER=clang-17 >/dev/null 2>&1
    cmake --build build >/dev/null 2>&1
    error_handling "$?" "Build Forge"
    echo "Forge Build is Successfully"
}

# Runs the pytest command and stores all logs in log path
# Input :
# $1: Pytest Command
# $2: Log path
# Output:
#   Last line of the pytest result
pytest_run() {
    local cmd="$1"
    local log_path=$2
    printf "%s\n" "$cmd"
    printf "%s\n" "$log_path"

    command="$cmd &> $log_path"
    eval "$command" >/dev/null 2>&1
    result=$(tail -5 "$log_path")
    echo "$result"
}

#Based on the pytest result, it will bisect good or bad
# Input :
# $1: pytest results
# $2: Expected string to be replicated
# Output:
#   Current Test case is pass or failed
#   First line of the bisect output
comparison_result() {
    local pytest_result=$1
    local expecting_string=$2
    local replication=1
    local bis_out

    if [ "$expecting_string" = "NA" ]; then
        replication=0
        expecting_string="passed"
    fi

    if echo "$pytest_result" | grep -q "skipped" ; then
        echo "============================= Testcase got skipped ============================="
        exit 1
    fi

    if echo "$pytest_result" | grep -q "$expecting_string" ; then
        if [ "$replication" -eq 0 ] ; then
            echo "============================= Test case got $expecting_string ============================="
        else
            echo "============================= $expecting_string case got replicated ============================="
        fi
    else
        if [ "$replication" -eq 0 ] ; then
            echo "============================= Test case got failed ============================="
            expecting_string="failed"
        else
            echo "============================= Not able to replicate $expecting_string case ============================="
            exit 1
        fi
    fi

    if [ "$expecting_string" = "passed" ] ; then
        bis_out=$(git bisect good | head -n 1 )
    else
        if [ "$expecting_string" = "failed" ] ; then
            bis_out=$(git bisect bad | head -n 1 )
        fi
    fi
    echo "$bis_out"
}

# This function calls build_script, pytest_run function and comparison_result function
# Input :
# $1: Expected string to be replicated
# $2: Pytest Command
# $3: Log Path
# $4: Run count
# Output:
#   First line of the bisect output
bisect_run() {
    replica_string=$1
    pytest_command=$2
    local Log_path
    if [ "$replica_string" = "NA" ]; then
        run_count=$4
        extension="_$run_count.txt"
        Log_path="$3/revision$extension"
    else
        extension="_replication.txt"
        Log_path="$3/$replica_string$extension"
    fi

    # reset function is used to reset machine before every pytest run
    reset

    # env_clean_and_build function is used to clean and build
    env_clean_and_build

    # Activate the Environment
    source "env/activate"

    pytest_result=$(pytest_run "$pytest_command" "$Log_path")
    final_result="${pytest_result#*txt}"
    bisect_output=$(comparison_result "$final_result" "$replica_string")

    {
        echo "$bisect_output"
    }> "$tempfile"
    echo "$bisect_output"

}

########################### main #################################

#INPUTS
# get_inputs function get 3 inputs from user and returns 3 outputs
# Parameters:
#   $1: Pytest Command
#   $2: Passing Commit Id
#   $3: Failing Commit Id
# Returns:
#   pytest_command, pass_id, fail_id

inputs=$(get_inputs)
IFS=',' read -r pytest_command pass_id fail_id <<< "$inputs"
echo "$pytest_command"
echo "$pass_id"
echo "$fail_id"

if ! [ -d "Logs" ]; then
    mkdir "Logs"
fi

# set_evn_flags function is to set all environmental flags
set_evn_flags


#Creating folder for dumping the logs
file_path=$(echo "$pytest_command" | cut -d'.' -f1)
model_name=$(echo "$file_path" | awk -F'/' '{print $NF}')


folder_path="Logs/$model_name"
if ! [ -d "$folder_path" ]; then
    mkdir "$folder_path"
else
    echo "Log Directory exists. Doing a clean up and creating new one..."
    rm -rf "$folder_path"
    mkdir "$folder_path"
fi


run_count=0


#To Avoid clash with previous bisect run we are resetting and starting.
git bisect reset >/dev/null 2>&1
git bisect start

# bisect_run function get 4 inputs from user and returns result
# $1: Expected string to be replicated
# $2: Pytest Command
# $3: Folder Path
# $4: Run count
# Returns:
# bisect result

# Replicating Pipeline last passing commit id in local run
echo -e "\nGoing to replicate pass case in last passing commit id..."
git checkout $pass_id >/dev/null 2>&1
bisect_run "passed" "$pytest_command" "$folder_path" "$run_count"
rm "$tempfile"
tempfile=$(mktemp)

#Replicating Pipeline first failing commit id in local run
echo "Going to replicate fail case in first failing commit id..."
git checkout "$fail_id" >/dev/null 2>&1
bisect_run "failed" "$pytest_command" "$folder_path" "$run_count"
bisect_output=$(tail -n 1 "$tempfile")
rm "$tempfile"

#This loop will be continued untill we are getting first regressed commit id
while ! echo "$bisect_output" | grep -q "first bad commit"; do
    run_count=$((run_count+1))
    tempfile=$(mktemp)
    bisect_run "NA" "$pytest_command" "$folder_path" "$run_count"
    bisect_output=$(tail -n 1 "$tempfile")
    echo "$bisect_output"
    rm "$tempfile"
    sleep 1
done

echo "Bisect results"
extension="/bisect_log.txt"
git bisect log | tee "$folder_path$extension"
