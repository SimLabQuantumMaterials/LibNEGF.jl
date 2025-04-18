#!/bin/bash

# run as : ./runbenchmrks.sh HW
# where HW is one of : cpu, amd, nvidia, intel, apple,
# with all of these indicating that we run on GPUs, except
# the first one (i.e. cpu)

# function taken from:
# https://www.baeldung.com/linux/check-variable-exists-in-list
function exists_in_list() {
    LIST=$1
    DELIMITER=$2
    VALUE=$3
    LIST_WHITESPACES=`echo $LIST | tr "$DELIMITER" " "`
    for x in $LIST_WHITESPACES; do
        if [ "$x" = "$VALUE" ]; then
            return 0
        fi
    done
    return 1
}

HWs="cpu apple nvidia amd intel"

if exists_in_list "$HWs" " " $1; then
    cp ../Manifest_$1.toml ../Manifest.toml
    export OPENBLAS_NUM_THREADS=3
    export JULIA_NUM_THREADS=2
    julia runbenchmrks.jl $1
else
    echo "The hardware $1 is not in the list, not running the tests"
fi

# restore Project.toml in case it was modified by this execution, save
# the modified version to avoid being too intrusive
cp ../Project.toml ../Project_modif.toml
git restore ../Project.toml