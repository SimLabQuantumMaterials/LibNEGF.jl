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
    cp Manifest_$1.toml Manifest.toml
    julia test.jl $1
else
    echo "The hardware $1 is not in the list, not running the tests"
fi