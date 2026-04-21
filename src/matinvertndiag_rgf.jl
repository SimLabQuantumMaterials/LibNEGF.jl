using TimerOutputs

"""
	AuxDataRGF

Buffers used by RGF. The matrix `buffM` is used at the RGF level,
while `bIdM` is the identity in block 1-diagonal form whic his used
for explicit inversions via `getrs!(..)`.
"""
struct AuxDataRGF
    buffM::BlockMatrix
    bIdM::BlockMatrix
    buildFullInv::Bool
end

function check_if_enough_mem_rgf(npl::Int, blockSize::Int, precx::DataType)
    # total system memory in MB
    totalMem = Sys.total_memory() / 2^20

    # +1 for a buffer identity
    N1diag = 1
    # +1 for the sparse original matrix, +1 for the conversion of that
    # original matrix to the input block tridiagonal matrix, +1 for a buffer
    # block tridiagonal matrix in RGF
    N3diag = 3

    # before allocating, check whether there is enough memory
    # to allocate all the needed buffers
    requiredMem::Float64 = required_mem_non_symm(npl, precx, N1diag, N3diag, blockSize)
    if requiredMem > 0.8 * totalMem
        error("The required memory exceeds 80% of the total memory")
    end
end

# get the memory required by non-symmetric matrices, in MB
# npl : number of principal layers
# N1diag : number of block diagonal matrices to be allocated
# N3diag : number of block tridiagonal matrices to be allocated
function required_mem_non_symm(npl::Int, precx::DataType, N1diag::Int, N3diag::Int, avgBlockSize::Int)::Float64
    requiredMem::Float64 = 0.0

    for ix = 1:npl
        # left
        if ix > 1
            nx = avgBlockSize
            ny = avgBlockSize
            requiredMem += N3diag * (nx * ny)
        end

        # center
        nx = avgBlockSize
        ny = avgBlockSize
        requiredMem += N3diag * (nx * ny)
        requiredMem += N1diag * (nx * ny)

        # right
        if ix < npl
            nx = avgBlockSize
            ny = avgBlockSize
            requiredMem += N3diag * (nx * ny)
        end
    end

    requiredMem *= (2 * sizeof(precx) / 2^20)

    return requiredMem
end

"""
	allocate_aux_data_RGF(M::BlockMatrix, nrBLASThreadsOuter::Int, nrBLASThreadsInner::Int)

Based on the block-sparsity pattern of the input matrix `M`, allocate the
buffers in `AuxDataRGF`.

# Arguments
- `M::BlockMatrix`: the matrix used as reference.
- `nrBLASThreadsOuter::Int`.
- `nrBLASThreadsInner::Int`.
"""
function allocate_aux_data_RGF(M::BlockMatrix)::AuxDataRGF
    npl = size(M.blockSizes)[1]

    # in general, these type of auxiliary block matrices will contain
    # Array-like object and not LU-like, as specified by the last param
    buffM = BlockMatrix(copy(M.blockSizes), ArrayOrLU_(undef, npl, npl),
        M.ndiag, 1, false)
    bm_blocks_define!(buffM, 1)

    bIdM = BlockMatrix(copy(M.blockSizes), ArrayOrLU_(undef, npl, npl),
        Dict("in" => 1, "out" => 1), 0, false)
    bm_blocks_define_identity!(bIdM)

    # the final struct with the buffers
    auxData = AuxDataRGF(buffM, bIdM, 0)

    return auxData
end

"""
    bndiag_of_inv_rgf_local!(Mout::BlockMatrix, Min::BlockMatrix, auxData::AuxDataRGF, td::TimingData,
        cd::CountingData)

For an input matrix `M`, possibly but not necessarily block n-diagonal,
where n is 3, 5, etc., compute the block n-diagonal part of the inverse
of `M`. This function uses the RGF method (soon to be extended to DD-RGF).

# Arguments
- `Min::BlockMatrix`: the matrix to be inverted.
- `Mout::BlockMatrix`: the output matrix.
- `auxData`: auxiliary buffers.
- `td`: struct for fine-level (i.e. of the backend kernels) timing. The user can choose no timing,
in which case `td` is an empty `TimingData` struct.
"""
function bndiag_of_inv_rgf_local!(Mout::BlockMatrix, Min::BlockMatrix, auxData::AuxDataRGF, td::TimingData,
    cd::CountingData)
    # TODO : extend this code to n-diagonal, otherwise rename this function
    #        to keep it as the simple traditional RGF

    # we call this function if we want to call standalone RGF

    minusOneCmplx = convert(FieldType, -1.0)
    plusOneCmplx = convert(FieldType, 1.0)
    zeroCmplx = convert(FieldType, 0.0)

    # IMPORTANT : we assume here that all of the blocks in Min and Mout argument
    #             Array-like, and that those in the block-diagonal of auxData.rgfBuffs.buffM
    #             are LU-like

    npl = size(Mout.blockSizes)[1]
    buffM1 = auxData.buffM
    # Mout is used as a buffer in multiple places, this is just
    # labeling for clarity of the implementation
    buffM2 = Mout
    buffId = auxData.bIdM

    # TODO : we might not need a full block n-diagonal as a buffer. To see this,
    #        go again over the algorithm, first simple RGF, and note that there
    #        are more LAPACK in-place possibilities (namely, due to getrs! within
    #        be_mldivide(..))

    # FIRST, upward pass

    # bottom element
    be_lu!(buffM1.M[npl, npl], Min.M[npl, npl], td, cd)

    # middle elements
    for ix = npl-1:-1:1
        # first run mrdivide, to make use of the mldivide data as a buffer for mrdivide

        # this is how we implement be_mrdivide!(..) via be_mldivide!(..)
        begin
            be_ctranspose!(buffM1.M[ix+1, ix], Min.M[ix, ix+1], td, cd)
            be_mldivide!('C', buffM2.M[ix+1, ix], buffM1.M[ix+1, ix], buffM1.M[ix+1, ix+1], td, cd)
            be_ctranspose!(buffM1.M[ix, ix+1], buffM2.M[ix+1, ix], td, cd)
        end

        be_mldivide!('N', buffM1.M[ix+1, ix], Min.M[ix+1, ix], buffM1.M[ix+1, ix+1], td, cd)

        be_copy_in_hw!(buffM2.M[ix, ix], Min.M[ix, ix])
        be_gemm!('N', 'N', minusOneCmplx, Min.M[ix, ix+1], buffM1.M[ix+1, ix], plusOneCmplx, buffM2.M[ix, ix], td, cd)
        be_lu!(buffM1.M[ix, ix], buffM2.M[ix, ix], td, cd)
    end

    # THEN, downward pass

    # top element
    # using be_mldivide!(..) instead of be_inv_from_lu!(..) because we want
    # to preallocate everything ourselves and prevent LAPACK from doing it on the fly
    be_mldivide!('N', Mout.M[1, 1], buffId.M[1, 1], buffM1.M[1, 1], td, cd)

    # middle elements
    for ix = 2:npl
        # upper diagonal of Mout
        be_gemm!('N', 'N', minusOneCmplx, Mout.M[ix-1, ix-1], buffM1.M[ix-1, ix], zeroCmplx, Mout.M[ix-1, ix], td, cd)

        # lower diagonal of Mout
        be_gemm!('N', 'N', minusOneCmplx, buffM1.M[ix, ix-1], Mout.M[ix-1, ix-1], zeroCmplx, Mout.M[ix, ix-1], td, cd)

        # diagonal of Mout
        be_mldivide!('N', Mout.M[ix, ix], buffId.M[ix, ix], buffM1.M[ix, ix], td, cd)
        be_gemm!('N', 'N', minusOneCmplx, buffM1.M[ix, ix-1], Mout.M[ix-1, ix], plusOneCmplx, Mout.M[ix, ix], td, cd)
    end

    # if auxData.buildFullInv = 1, then compute all the other missing blocks of the inverse
    if auxData.buildFullInv == 1
        if npl > 2
            for ix = 1:npl
                kx::Int = 1
                for jx = (ix+2):npl
                    # first, do the upper triangular part
                    be_gemm!('N', 'N', minusOneCmplx, Mout.M[ix, jx-1], buffM1.M[ix+kx, jx], zeroCmplx, Mout.M[ix, jx], td, cd)
                    # then, do the lower triangular
                    be_gemm!('N', 'N', minusOneCmplx, buffM1.M[jx, ix+kx], Mout.M[jx-1, ix], zeroCmplx, Mout.M[jx, ix], td, cd)

                    kx += 1
                end
            end
        end
    end
end