"A Julia package consisting of a rework of some parts of [libNEGF](https://github.com/libnegf/libnegf)."
module LibNEGF

using SparseArrays, CSV, MAT, LinearAlgebra, Random

include("utils/parsing.jl")

@ifdef "LIBNEGF_TEST_OR_BENCH" begin
    if ENV["LIBNEGF_TEST_OR_BENCH"]=="test" include("exports_test.jl")
    elseif ENV["LIBNEGF_TEST_OR_BENCH"]=="benchmark" include("exports_benchmark.jl")
    elseif ENV["LIBNEGF_TEST_OR_BENCH"]=="documentation" include("exports_doc.jl")
    elseif ENV["LIBNEGF_TEST_OR_BENCH"]=="example" include("exports_example.jl") end
end

# from utils/
# this is the core set of utils, where some macros are included
# empty, and if then utils_optnl.jl is included those empty macros
# are replaced
include("utils/common.jl")
@ifdef "LIBNEGF_FINER_TIMINGS" begin
    if ENV["LIBNEGF_FINER_TIMINGS"]=="0" which_timer = "empty"
    else which_timer = "full" end
    include("utils/"*which_timer*"_timings.jl")
end

# # include the backend for the desired harwdware, replace
# # HW by one of cpu, apple, nvidia, etc
@ifdef "LIBNEGF_HW" begin
    include("backend_"*ENV["LIBNEGF_HW"]*".jl")
end

include("matloader.jl")
include("block.jl")
include("blockmatrix.jl")
include("matinvertndiag_direct.jl")
include("matinvertndiag_ddrgf.jl")
include("keldyshndiag.jl")
include("selected_inverse.jl")

end