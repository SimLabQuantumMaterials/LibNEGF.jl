"""
	AuxDataDDRGF

Buffers used by RGF. The matrix `buffM` is used at the RGF level,
while `bIdM` is the identity in block 1-diagonal form whic his used
for explicit inversions via `getrs!(..)`.
"""
struct AuxDataDDRGF
    buffM::BlockMatrix
    bIdM::BlockMatrix
end

"""
	allocate_aux_data_DDRGF(M::BlockMatrix)

Based on the block-sparsity pattern of the input matrix `M`, allocate the
buffers in `AuxDataDDRGF`.

# Arguments
- `M::BlockMatrix`: the matrix used as reference.
"""
function allocate_aux_data_DDRGF(M::BlockMatrix)::AuxDataDDRGF
    npl = size(M.blockSizes)[1]

    # in general, these type of auxiliary block matrices will contain
    # Array-like object and not LU-like, as specified by the last param
    buffM = BlockMatrix(M.blockSizes, ArrayOrLU_(undef, npl, npl),
        M.ndiag, M.nrsType, 1)
    set_blocks_to_zero!(buffM)

    bIdM = BlockMatrix(M.blockSizes, ArrayOrLU_(undef, npl, npl),
        Dict("in" => 1, "out" => 1), M.nrsType, 0)
    set_blocks_to_identity!(bIdM)

    # the final struct with the buffers
    auxData = AuxDataDDRGF(buffM, bIdM)

    return auxData
end

"""
    bndiag_of_inv_ddrgf!(Mout::BlockMatrix, Min::BlockMatrix, auxData::AuxDataDDRGF, td::TimingData)

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
function bndiag_of_inv_ddrgf!(Mout::BlockMatrix, Min::BlockMatrix, auxData::AuxDataDDRGF, td::TimingData)
    # TODO : extend this code to n-diagonal, otherwise rename this function
    #        to keep it as the simple traditional RGF

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
    @timewrap td "_lu" be_lu!(buffM1.M[npl, npl], Min.M[npl, npl])

    # middle elements
    for ix = npl-1:-1:1
        # first run mrdivide, to make use of the mldivide data as a buffer for mrdivide

        # this is how we implement be_mrdivide!(..) via be_mldivide!(..)
        @timewrap td "_mrdivide" begin
            be_ctranspose!(buffM1.M[ix+1, ix], Min.M[ix, ix+1])
            be_mldivide!('C', buffM2.M[ix+1, ix], buffM1.M[ix+1, ix], buffM1.M[ix+1, ix+1])
            be_ctranspose!(buffM1.M[ix, ix+1], buffM2.M[ix+1, ix])
        end

        @timewrap td "_mldivide" be_mldivide!('N', buffM1.M[ix+1, ix], Min.M[ix+1, ix], buffM1.M[ix+1, ix+1])

        be_copy_in_hw!(buffM2.M[ix, ix], Min.M[ix, ix])
        @timewrap td "_gemm" be_gemm!('N', 'N', convert(Min.nrsType, -1.0), Min.M[ix, ix+1], buffM1.M[ix+1, ix],
            convert(Min.nrsType, 1.0), buffM2.M[ix, ix])
        @timewrap td "_lu" be_lu!(buffM1.M[ix, ix], buffM2.M[ix, ix])
    end

    # THEN, downward pass

    # top element
    # using be_mldivide!(..) instead of be_inv_from_lu!(..) because we want
    # to preallocate everything ourselves and avoid LAPACK from doing it on the fly
    @timewrap td "_mldivide" be_mldivide!('N', Mout.M[1, 1], buffId.M[1, 1], buffM1.M[1, 1])

    # # middle elements
    for ix = 2:npl
        # upper diagonal of Mout
        @timewrap td "_gemm" be_gemm!('N', 'N', convert(Min.nrsType, -1.0), Mout.M[ix-1, ix-1], buffM1.M[ix-1, ix],
            convert(Min.nrsType, 0.0), Mout.M[ix-1, ix])

        # lower diagonal of Mout
        @timewrap td "_gemm" be_gemm!('N', 'N', convert(Min.nrsType, -1.0), buffM1.M[ix, ix-1], Mout.M[ix-1, ix-1],
            convert(Min.nrsType, 0.0), Mout.M[ix, ix-1])

        # diagonal of Mout
        @timewrap td "_mldivide" be_mldivide!('N', Mout.M[ix, ix], buffId.M[ix, ix], buffM1.M[ix, ix])
        @timewrap td "_gemm" be_gemm!('N', 'N', convert(Min.nrsType, -1.0), buffM1.M[ix, ix-1], Mout.M[ix-1, ix],
            convert(Min.nrsType, 1.0), Mout.M[ix, ix])
    end
end