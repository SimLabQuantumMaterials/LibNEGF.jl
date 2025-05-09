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
    export OPENBLAS_NUM_THREADS=2
    export JULIA_NUM_THREADS=3
    # include the backend for that HW
    sed -i -e "s/backend_HW.jl/backend_$1.jl/g" ../src/LibNEGF.jl
    if [ "$2" -ne 0 ];then
        sed -i -e 's|include("utils/empty_timings.jl")|include("utils/full_timings.jl")|' ../src/LibNEGF.jl
    fi
    # launch the benchmark runs
    julia --threads=$JULIA_NUM_THREADS runbenchmrks.jl $1 $2
    # revert the change to ../src/LibNEGF.jl
    sed -i -e "s/backend_$1.jl/backend_HW.jl/g" ../src/LibNEGF.jl
    if [ "$2" -ne 0 ];then
        sed -i -e 's|include("utils/full_timings.jl")|include("utils/empty_timings.jl")|' ../src/LibNEGF.jl
    fi
else
    echo "The hardware $1 is not in the list, not running the tests"
fi

# restore Project.toml and src/LibNEGF.jl in case it was modified
# by this execution, save the modified versions to avoid being too intrusive
cp ../Project.toml ../Project_modif.toml
git restore ../Project.toml
