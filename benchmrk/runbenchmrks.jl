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
Printf.@printf("  -- nr of Julia threads: %d\n", parse(Int, ARGS[3]))
Printf.@printf("  -- nr of BLAS threads: %d\n", parse(Int, ARGS[4]))

# choose one of "rgf", "ddrgf", "rkd", "general_rgf"
whichBM = "general_rgf_fused"

if whichBM == "rgf"
    to = TimerOutput()
    include("benchmrk_matinvertndiag_rgf.jl")
    println(to)
elseif whichBM == "rkd"
    to = TimerOutput()
    include("benchmrk_keldyshndiag.jl")
    println(to)
elseif whichBM == "ddrgf"
    to = TimerOutput()
    include("benchmrk_matinvertndiag_ddrgf.jl")
    println(to)
elseif whichBM == "general_rgf"
    to = TimerOutput()
    include("benchmrk_matinvertndiag_general_rgf.jl")
    println(to)
elseif whichBM == "general_rgf_fused"
    to = TimerOutput()
    include("benchmrk_matinvertndiag_general_rgf_fused.jl")
    println(to)
elseif whichBM == "selected_inverse"
    to = TimerOutput()
    include("benchmrk_selected_inverse.jl")
    println(to)
else
    error("The chosen method is not one of rgf, ddrgf or rkd")
end

# legacy
# include("benchmrk_matinvertndiag_direct.jl")
