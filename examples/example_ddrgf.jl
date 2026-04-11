using LibNEGF, LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(Int(parse(Float64, ARGS[4])))

# in this example we show how to compute, for a given block tridiagonal
# matrix, the block tridiagonal part of its inverse. This is done in this
# example via a parallel alternative to sequential RGF, which we here
# call DDRGF

npl = 128
blockSize = 64
precx = ComplexF64

# NOTE :
# if the user has their own sparse matrix, let's call that M,
# of type SparseArrays.SparseMatrixCSC{precx,Int}, then they
# can convert it first to block tridiagonal by doing:
# Min = bm_convert(M, blockSizes, Dict("in" => 3, "out" => 3), false),
# where blockSizes is an array with the sizes of the blocks (corresponding
# to the sizes of the pricipal layers)

check_if_enough_mem_ddrgf(npl, blockSize, precx)

# non-Hermitian matrix
Min = bm_create_synthetic_random(npl, blockSize, precx, false)
Mout = bm_copy(Min)

listOfAuxDataPar = nothing
try
    listOfAuxDataPar = allocate_aux_data_DDRGF(Min, false, parse(Int, ARGS[3]), parse(Int, ARGS[4]), TimingData(), CountingData())
catch
    if e isa OutOfMemoryError
        # TODO : handle this better, with perhaps a suggestion in params change
        error("The application tried to allocate beyond the available system memory")
    else rethrow(e) end
end

bndiag_of_inv_ddrgf!(Min, listOfAuxDataPar, TimingData(), CountingData(), 1)
bm_copy!(Mout, listOfAuxDataPar[1].buffMout)

# finally, convert Mout to a sparse matrix
Moutsp = bm_convert(Mout)