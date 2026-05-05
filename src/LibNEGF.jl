"A Julia package consisting of a rework of some parts of [libNEGF](https://github.com/libnegf/libnegf)."
module LibNEGF

using SparseArrays, LinearAlgebra, Random, Polyester

include("utils/parsing.jl")

@ifdef "LIBNEGF_TEST_OR_BENCH" begin
    if ENV["LIBNEGF_TEST_OR_BENCH"] == "test"
        include("exports_test.jl")
    elseif ENV["LIBNEGF_TEST_OR_BENCH"] == "benchmark"
        include("exports_benchmark.jl")
    elseif ENV["LIBNEGF_TEST_OR_BENCH"] == "documentation"
        include("exports_doc.jl")
    elseif ENV["LIBNEGF_TEST_OR_BENCH"] == "compile"
        include("exports_compile.jl")
    elseif ENV["LIBNEGF_TEST_OR_BENCH"] == "example"
        include("exports_example.jl")
    end
end

# from utils/
# this is the core set of utils, where some macros are included
# empty, and if then utils_optnl.jl is included those empty macros
# are replaced@ifdef "LIBNEGF_COMPILE" begin
@ifdef "LIBNEGF_COMPILE" begin
    if ENV["LIBNEGF_COMPILE"] == "yes"
        # for C compilation and shared library generation
        include("utils/empty_common.jl")
    else
        include("utils/common.jl")
    end
end

@ifdef "LIBNEGF_FINER_TIMINGS" begin
    if ENV["LIBNEGF_FINER_TIMINGS"] == "0"
        which_timer = "empty"
    else
        which_timer = "full"
    end
    include("utils/" * which_timer * "_timings.jl")
end

# specifying the underlying data type
const FieldType = ComplexF64
# we must give the dimension of Array to the compiler, so it maps
# it directly to Matrix
const ArrayWithType = Array{FieldType,2}
# the previous line will be the following in case of using Metal.jl
# const ArrayWithType = Metal.MtlArray{FieldType, 2}

# # include the backend for the desired harwdware, replace
# # HW by one of cpu, apple, nvidia, etc
@ifdef "LIBNEGF_HW" begin
    include("backend_" * ENV["LIBNEGF_HW"] * ".jl")
end

# for now, we are using random synthetic data only
# include("matloader.jl")

include("blockmatrix.jl")
include("matinvertndiag_direct.jl")
include("matinvertndiag_rgf.jl")
include("matinvertndiag_general_rgf.jl")
include("matinvertndiag_ddrgf_core.jl")
include("matinvertndiag_ddrgf_setup.jl")
include("matinvertndiag_ddrgf_utils.jl")
include("keldyshndiag.jl")

@ifdef "LIBNEGF_COMPILE" begin
    if ENV["LIBNEGF_COMPILE"] == "yes"
        # for C compilation and shared library generation
        include("c_interface/libnegf_c_interface.jl")
    end
end

end