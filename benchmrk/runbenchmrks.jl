using Pkg

supportedHWs = ["cpu", "apple"]
# unsupportedHWs = ["nvidia", "amd", "intel"]

if ARGS[1] ∉ supportedHWs
    println("The chosen hardware (", ARGS[1], ") is not yet supported")
    exit()
end

Pkg.activate("../");
# depending on the type of hardware, remove packages as necessary
if ARGS[1] == "cpu"
    Pkg.rm("Metal")
end
Pkg.instantiate()

using LibNEGF, LinearAlgebra, Printf, TimerOutputs, .Threads

include("common_to_benchmrks.jl")

to = TimerOutput()

Printf.@printf("\nBenchmarking, common info:\n")
Printf.@printf("  -- hardware: %s\n", ARGS[1])
Printf.@printf("  -- nr of BLAS threads: %d\n", LinearAlgebra.BLAS.get_num_threads())
Printf.@printf("  -- nr of outer threads: %d\n", Threads.nthreads())
Printf.@printf("  -- nr of energy points: %d\n\n", nrEvals)

# benchmarks common to all of the supported hardwares
# include("benchmrk_matinvertndiag_direct.jl")
include("benchmrk_matinvertndiag_rgf.jl")

println(to)