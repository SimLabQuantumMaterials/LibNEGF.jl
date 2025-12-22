# always test this, as it is the loading of matrices
# from files into CPU memory
include("test_matloader.jl")

# test basic functionality of the GPU libraries
if ARGS[1] == "apple"
    include("test_metal.jl")
end

# tests common to all of the supported hardwares
# include("test_blockmatrix.jl")
# include("test_backend.jl")
# include("test_matinvert.jl")
# include("test_keldyshndiag.jl")
include("test_block.jl")
include("test_selected_inverse.jl")
include("test_blockmatrix.jl")
include("test_backend.jl")
include("test_matinvert.jl")
include("test_keldyshndiag.jl")
