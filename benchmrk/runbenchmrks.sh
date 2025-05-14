#!/bin/bash

# run as : ./runbenchmrks.sh HW 0/1
# where HW is one of : cpu, amd, nvidia, intel, apple,
# with all of these indicating that we run on GPUs, except
# the first one (i.e. cpu). The second parameter is whether
# we want to include timings within the LibNEGF.jl or not

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

# checks on the param that specifies whether we add finer timings or not
if [ "$#" -ne 2 ]; then
    echo "The number of params for runbencharks.sh has to be 2"
    exit
fi
if [ "$2" -ne 0 ] && [ "$2" -ne 1 ]; then
    echo "The second param in runbenchmarks.sh has to be either 0 or 1"
    exit
fi

HWs="cpu apple nvidia amd intel"

if exists_in_list "$HWs" " " $1; then
    # get the manifest specific to the chosen HW
    cp ../Manifest_$1.toml ../Manifest.toml
    # create usable copy of Project_common.toml
    cp ../Project_common.toml ../Project.toml
    export OPENBLAS_NUM_THREADS=2
    export JULIA_NUM_THREADS=3
    # variables used to mimic C's ifdef
    export LIBNEGF_HW=$1
    export LIBNEGF_FINER_TIMINGS=$2
    # if we want to really mimic C's ifdef, we need to force recompilation,
    # which we do by removing the precompiled binaries
    JULIA_MAJOR_VERSION=`julia --version | egrep -o '[0-9].[0-9][0-9]'`
    BINS_JULIA=`ls ~/.julia/compiled/v$JULIA_MAJOR_VERSION/LibNEGF/*.ji`
    rm $BINS_JULIA

    # launch the benchmark runs
    julia --threads=$JULIA_NUM_THREADS runbenchmrks.jl $1 $2
else
    echo "The hardware $1 is not in the list, not running the tests"
fi