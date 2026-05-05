using LibNEGF, LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(Int(parse(Float64, ARGS[4])))

# in this example we show how to compute, for a given block tridiagonal
# matrix, the block tridiagonal part of its inverse. This is done in this
# example via the sequential RGF method

npl = 10
blockSize = 64
precx = ComplexF64

# NOTE :
# if the user has their own sparse matrix, let's call that M,
# of type SparseArrays.SparseMatrixCSC{precx,Int}, then they
# can convert it first to block tridiagonal by doing:
# Min = bm_convert(M, blockSizes, Dict("in" => 3, "out" => 3), false),
# where blockSizes is an array with the sizes of the blocks (corresponding
# to the sizes of the pricipal layers)

# # check if there's enough memory for the allocations
# check_if_enough_mem_rgf(npl, blockSize, precx)

Min = nothing
Mout = nothing
auxData = nothing
try
    # non-Hermitian matrix
    global Min = bm_create_synthetic_random(npl, blockSize, precx, false)
    global Mout = bm_copy(Min)
    global auxData = allocate_aux_data_RGF(Min)
catch e
    if e isa OutOfMemoryError
        # TODO : handle this better, with perhaps a suggestion in params change
        error("The application tried to allocate beyond the available system memory")
    else rethrow(e) end
end

bndiag_of_inv_rgf_local!(Mout, Min, auxData, TimingData(), CountingData())

# finally, convert Mout to a sparse matrix
Moutsp = bm_convert(Mout)