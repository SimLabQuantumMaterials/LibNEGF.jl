# always test this, as it is the loading of matrices
# from files into CPU memory
include("test_matloader.jl")

# test basic functionality of the GPU libraries
if ARGS[1] == "apple"
    include("test_metal.jl")
end

# the following two are still only available for CPUs
if ARGS[1] == "cpu"
    include("test_matinverter.jl")
    include("test_arrayorlu.jl")
end