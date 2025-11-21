using LibNEGF, LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(Int(parse(Float64, ARGS[4])))

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

# non-Hermitian matrix
Min = bm_create_synthetic_random(npl, blockSize, precx, false)
Moutbm = bm_copy(Min)

listOfAuxDataPar = allocate_aux_data_DDRGF(Min, false, parse(Int, ARGS[3]), parse(Int, ARGS[4]), TimingData(), CountingData())
bndiag_of_inv_ddrgf!(Min, listOfAuxDataPar, TimingData(), CountingData(), 1)
bm_copy!(Moutbm, listOfAuxDataPar[1].buffMout)

# finally, convert Mout to a sparse matrix
Moutsp = bm_convert(Moutbm)