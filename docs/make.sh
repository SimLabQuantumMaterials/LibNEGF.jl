#!/bin/bash

# run as : ./make.sh HW
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

# check that the correct number of params has been passed
if [ "$#" -ne 1 ]; then
    echo "The number of params for runbencharks.sh has to be 1"
    exit
fi

HWs="cpu apple nvidia amd intel"

if exists_in_list "$HWs" " " $1; then
    # get the manifest specific to the chosen HW
    cp ../Manifest_$1.toml ../Manifest.toml
    # create usable copy of Project_common.toml
    cp ../Project_common.toml ../Project.toml

    # variables used to mimic C's ifdef
    export LIBNEGF_HW=$1
    export LIBNEGF_FINER_TIMINGS=1
    # if we want to really mimic C's ifdef, we need to force recompilation,
    # which we do by removing the precompiled binaries
    JULIA_MAJOR_VERSION=`julia --version | egrep -o '[0-9].[0-9][0-9]'`
    BINS_JULIA=`ls ~/.julia/compiled/v$JULIA_MAJOR_VERSION/LibNEGF/*.ji`
    rm $BINS_JULIA

    # create the documentation
    julia --color=yes --project make.jl $1
else
    echo "The hardware $1 is not in the list, not running the tests"
fi