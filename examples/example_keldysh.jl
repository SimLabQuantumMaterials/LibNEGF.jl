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
# to the sizes of the pricipal layers). Same applies to the middle operator
# in Keldysh, but for that one the last parameter has to be 'true',
# indicating that the operator is Hermitian

# check if there's enough memory for the allocations
check_if_enough_mem_rkd(npl, blockSize, precx)

# non-Hermitian matrix
Min = bm_create_synthetic_random(npl, blockSize, precx, false)

# make Sn Hermitian, this will be the middle operator in Keldysh
Sn = bm_similar(Min, 2)
begin
    Snsp = bm_convert(Sn)
    Snsp = (Snsp + Snsp') / convert(precx, 2.0)
    Sn = bm_convert(Snsp, Sn.blockSizes, Sn.ndiag, true)
end

# pre-allocate buffer data for RKD
auxDataKeldysh = allocate_aux_data_Keldysh(Min, Sn, parse(Int, ARGS[3]), parse(Int, ARGS[4]))
keldyshndiag!(Min, Sn, auxDataKeldysh, TimingData(), CountingData())

# finally, convert Gn to a sparse matrix. IMPORTANT : this gives us the whole
# (Hermitian) matrix, no symmetry considerations in the following function to save memory
Gn = auxDataKeldysh.buffS
Gnsp = bm_convert(Gn)