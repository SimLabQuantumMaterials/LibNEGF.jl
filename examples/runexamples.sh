#!/bin/bash

# where is LibNEGF.jl
LIBNEGF_DIR="../"

# some local variables
HW="cpu"
FINER_TIMINGS=0

cp $LIBNEGF_DIR"Manifest_$HW.toml" $LIBNEGF_DIR"Manifest.toml"
cp $LIBNEGF_DIR"Project_common.toml" $LIBNEGF_DIR"Project.toml"

# some environment variables
export OPENBLAS_NUM_THREADS=1

# mimic C's ifdef : force recompilation
export LIBNEGF_HW=$HW
export LIBNEGF_FINER_TIMINGS=$FINER_TIMINGS
export LIBNEGF_TEST_OR_BENCH=example
JULIA_MAJOR_VERSION=$(julia --version | egrep -o '[0-9].[0-9][0-9]')
BINS_JULIA=$(ls ~/.julia/compiled/v$JULIA_MAJOR_VERSION/LibNEGF/*.ji)
rm $BINS_JULIA

# the outer threads are used only by DDRGF
JULIA_NUM_THREADS=1
NUM_BLAS_THREADS_OUTER=$JULIA_NUM_THREADS
NUM_BLAS_THREADS_INNER=1

julia --threads=$JULIA_NUM_THREADS runexamples.jl $HW $FINER_TIMINGS $NUM_BLAS_THREADS_OUTER $NUM_BLAS_THREADS_INNER $LIBNEGF_DIR