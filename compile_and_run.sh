#!/bin/bash

# run as : ./test.sh HW
# where HW is one of : cpu, amd, nvidia, intel, apple,
# with all of these indicating that we run on GPUs, except
# the first one (i.e. cpu)

# function taken from:
# https://www.baeldung.com/linux/check-variable-exists-in-list
function exists_in_list() {
    LIST=$1
    DELIMITER=$2
    VALUE=$3
    LIST_WHITESPACES=$(echo $LIST | tr "$DELIMITER" ' ')
    for x in $LIST_WHITESPACES; do
        if [ "$x" = "$VALUE" ]; then
            return 0
        fi
    done
    return 1
}

# check that the correct number of params has been passed
if [ "$#" -ne 1 ]; then
    echo "The number of params for test.sh has to be 1"
    exit
fi

HWs="cpu apple nvidia amd intel"

if exists_in_list "$HWs" " " $1; then
    # before running, compile
    ./compile.sh $1

    # this one is for threading within blocks in RGF
    export OPENBLAS_NUM_THREADS=1

    # this is for inter-block threading in DDRGF
    export JULIA_NUM_THREADS=1

    ./libnegf_c_example
else
    echo "The hardware $1 is not in the list, not running the tests"
fi