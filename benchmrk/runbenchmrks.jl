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
# the following line triggers the precompilation of LibNEGF
Pkg.instantiate()

using LibNEGF, LinearAlgebra, Printf, TimerOutputs, .Threads

include("common_to_benchmrks.jl")

Printf.@printf("\nBenchmarking, common info:\n")
Printf.@printf("  -- hardware: %s\n", ARGS[1])
Printf.@printf("  -- nr of BLAS threads outer: %d\n", parse(Int, ARGS[4]))
Printf.@printf("  -- nr of BLAS threads inner: %d\n", parse(Int, ARGS[5]))
Printf.@printf("  -- nr of outer threads: %d\n", Threads.nthreads())
Printf.@printf("  -- nr of energy points: %d\n\n", size(Epoints)[1])

# benchmarks common to all of the supported hardwares
# include("benchmrk_matinvertndiag_direct.jl")
to = TimerOutput()
include("benchmrk_matinvertndiag_ddrgf.jl")
println(to)
println("")
# to = TimerOutput()
# include("benchmrk_keldyshndiag.jl")
# println(to)