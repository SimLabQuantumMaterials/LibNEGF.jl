using TimerOutputs

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

struct AuxDataPDDRGF
    # the sequential data
    auxDataSeq::AuxDataDDRGF
    # the extra (parallel-related) params
    nrTasks::Int
    permVec::Vector{Int}
    permVecInv::Vector{Int}
    sizeDomains::Vector{Int}
    blockSizeD1::Int
    blockSizeD2::Int
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
    bm_blocks_define!(buffM, 1)

    bIdM = BlockMatrix(M.blockSizes, ArrayOrLU_(undef, npl, npl),
        Dict("in" => 1, "out" => 1), M.nrsType, 0)
    bm_blocks_define_identity!(bIdM)

    # the final struct with the buffers
    auxData = AuxDataDDRGF(buffM, bIdM)

    return auxData
end

"""
	allocate_aux_data_PDDRGF(M::BlockMatrix, auxDataSeq::AuxDataDDRGF)

Allocate some extra buffers in `AuxDataPDDRGF` useful for parallel RGF.

# Arguments
- `M::BlockMatrix`: the matrix used as reference.
- `auxDataSeq::AuxDataDDRGF`: reference to the data pre-allocated already for sequential RGF.
"""
function allocate_aux_data_PDDRGF(M::BlockMatrix, nrBlocksInPivots::Int, splitType::Bool,
    auxDataSeq::AuxDataDDRGF)::AuxDataPDDRGF
    nrTasks = Threads.nthreads()

    # this might change the number of threads to be used
    nrTasks, blockSizeD1, blockSizeD2 = bndiag_of_inv_pddrgf_check_nr_tasks(M, nrTasks, nrBlocksInPivots, splitType)
    if nrTasks == 1
        println("WARNING: nrTasks = 1, then calling sequential RGF.")
        return AuxDataPDDRGF(auxDataSeq, nrTasks, Vector{Int}(), Vector{Int}(), Vector{Int}(), 0, 0)
    end

    permVec, sizeDomains = bndiag_of_inv_pddrgf_create_permutation_vector(M, nrTasks, nrBlocksInPivots, splitType)

    permVecInv = bndiag_of_inv_pddrgf_transpose_permutation_vector(permVec)

    # the final struct with the buffers
    auxDataPar = AuxDataPDDRGF(auxDataSeq, nrTasks, permVec, permVecInv, sizeDomains, blockSizeD1, blockSizeD2)

    return auxDataPar
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
function bndiag_of_inv_ddrgf!(Mout::BlockMatrix, Min::BlockMatrix, auxData::AuxDataDDRGF, td::TimingData, cd::CountingData)
    # TODO : extend this code to n-diagonal, otherwise rename this function
    #        to keep it as the simple traditional RGF

    minusOneCmplx = convert(Min.nrsType, -1.0)
    plusOneCmplx = convert(Min.nrsType, 1.0)
    zeroCmplx = convert(Min.nrsType, 0.0)

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
    # to preallocate everything ourselves and avoid LAPACK from doing it on the fly
    be_mldivide!('N', Mout.M[1, 1], buffId.M[1, 1], buffM1.M[1, 1], td, cd)

    # # middle elements
    for ix = 2:npl
        # upper diagonal of Mout
        be_gemm!('N', 'N', minusOneCmplx, Mout.M[ix-1, ix-1], buffM1.M[ix-1, ix], zeroCmplx, Mout.M[ix-1, ix], td, cd)

        # lower diagonal of Mout
        be_gemm!('N', 'N', minusOneCmplx, buffM1.M[ix, ix-1], Mout.M[ix-1, ix-1], zeroCmplx, Mout.M[ix, ix-1], td, cd)

        # diagonal of Mout
        be_mldivide!('N', Mout.M[ix, ix], buffId.M[ix, ix], buffM1.M[ix, ix], td, cd)
        be_gemm!('N', 'N', minusOneCmplx, buffM1.M[ix, ix-1], Mout.M[ix-1, ix], plusOneCmplx, Mout.M[ix, ix], td, cd)
    end
end

"""
    bndiag_of_inv_pddrgf_check_nr_tasks(M::BlockMatrix, nrTasks::Int, nrBlocksInPivots::Int, splitType::Bool)

This function checks whether the desired number of tasks is possible, and if not, gives back
an alternative, let's call this nrTasksOut. Then, it checks whether the total number of Julia
threads times the total number of BLAS threads is divisible by nrTasksOut, and if not then returns
1 indicating that it is not possible to more forward with the desired parallelization (i.e. then
do sequential RGF).

# Arguments
- `M::BlockMatrix`: the matrix of the system.
- `nrTasks::Int`: number of subdomains in which domain 1 is to be divided.
- `nrBlocksInPivots::Int`: number of principal layers in each of the subdomains of domain 2.
- `splitType::Int`: whether we add an extra pivot at the very end or not.
"""
function bndiag_of_inv_pddrgf_check_nr_tasks(M::BlockMatrix, nrTasks::Int, nrBlocksInPivots::Int,
    splitType::Bool)::Tuple{Int, Int, Int}

    if nrTasks == 1
        # if nrTasks = 1, the other two values are irrelevant
        return nrTasks, 0, 0
    end

    npl = size(M.blockSizes)[1]

    if splitType == Bool(0)
        nrPivots = nrTasks
    else
        nrPivots = nrTasks + 1
    end

    blockSizeD2 = nrBlocksInPivots
    totalSizeD2 = blockSizeD2 * nrPivots
    totalSizeD1 = npl - totalSizeD2

    blockSizeD1 = Int(ceil(totalSizeD1 / nrTasks))
    ceilOfTotalSizeD1 = (nrTasks - 1) * blockSizeD1
    restOfTotalSizeD1 = totalSizeD1 - ceilOfTotalSizeD1

    if restOfTotalSizeD1 <= 0
        nrTasks, blockSizeD1, blockSizeD2 = bndiag_of_inv_pddrgf_check_nr_tasks(M, nrTasks-1, nrBlocksInPivots, splitType)
    end

    return nrTasks, blockSizeD1, blockSizeD2
end

"""
    bndiag_of_inv_pddrgf_create_permutation_vector(M::BlockMatrix, nrTasks::Int,
    nrBlocksInPivots::Int, splitType::Bool)

Creates the permutation vector for later parallel RGF computations.

# Arguments
- `M::BlockMatrix`: the matrix of the system.
- `nrTasks::Int`: number of subdomains in which domain 1 is to be divided.
- `nrBlocksInPivots::Int`: number of principal layers in each of the subdomains of domain 2.
- `splitType::Int`: whether we add an extra pivot at the very end or not.
"""
function bndiag_of_inv_pddrgf_create_permutation_vector(M::BlockMatrix, nrTasks::Int,
    nrBlocksInPivots::Int, splitType::Bool)::Tuple{Vector{Int}, Vector{Int}}
    npl = size(M.blockSizes)[1]
    nrSubdomains::Int = 0
    idSubdomain::Int = 0

    if splitType == Bool(0)
        nrPivots = nrTasks
    else
        nrPivots = nrTasks + 1
    end
    nrSubdomains += nrPivots
    nrSubdomains += nrTasks
    sizeDomains = Vector{Int}(undef, nrSubdomains)

    blockSizeD2 = nrBlocksInPivots
    totalSizeD2 = blockSizeD2 * nrPivots
    totalSizeD1 = npl - totalSizeD2
    blockSizeD1 = ceil(totalSizeD1 / nrTasks)
    lastSizeD1  = totalSizeD1 - blockSizeD1 * (nrTasks - 1)

    permVec = Vector{Int}(undef, npl)

    # first, gather all the indices of region 2 (i.e. the pivots)
    ixNew = 0
    ixOld = 0
    for ix = 1:nrPivots
        for jx = 1:blockSizeD2
            ixNew += 1
            ixOld += 1
            permVec[ixNew] = ixOld
        end
        idSubdomain += 1
        sizeDomains[idSubdomain] = blockSizeD2
        if ix == nrTasks
            ixOld += lastSizeD1
        else
            ixOld += blockSizeD1
        end
    end

    # then, gather all the indices of region 1
    ixOld = 0
    for ix = 1:nrTasks
        ixOld += blockSizeD2
        if ix == nrTasks
            bs = lastSizeD1
        else
            bs = blockSizeD1
        end
        for jx = 1:bs
            ixNew += 1
            ixOld += 1
            permVec[ixNew] = ixOld
        end
        idSubdomain += 1
        sizeDomains[idSubdomain] = bs
    end

    return permVec, sizeDomains
end

# this function returns the permutation vector corresponding to the transpose of the
# permutation matrix
function bndiag_of_inv_pddrgf_transpose_permutation_vector(permVec::Vector{Int})::Vector{Int}
    permVecInv = copy(permVec)

    for ix = 1:size(permVec)[1]
        permVecInv[permVec[ix]] = ix
    end

    return permVecInv
end

# this function applies the permutation, specified via permVec, on the input matrix M, but
# returning a matrix whose blocks are references to the blocks in M
function bndiag_of_inv_pddrgf_create_permuted_matrix(M::BlockMatrix, permVec::Vector{Int})::BlockMatrix
    npl = size(M.blockSizes)[1]
    ndiag = M.ndiag
    Mhat = BlockMatrix(M.blockSizes, ArrayOrLU_(undef, npl, npl), M.ndiag, M.nrsType, 0)
    pv = permVec

    # loop over the block sizes, conversely over the block rows
    for ix = 1:npl
        # now, copy the blocks within the ix-th row
        if ix > 1
            # left
            for jx = (ix-1):-1:max(1, ix - Int((ndiag["out"] - 1) / 2))
                # be_copy_in_hw!(Mout.M[ix, jx], Min.M[ix, jx])
                Mhat.M[pv[ix],pv[jx]] = M.M[ix,jx]
            end
        end
        # center
        # be_copy_in_hw!(Mout.M[ix, ix], Min.M[ix, ix])
        Mhat.M[pv[ix],pv[ix]] = M.M[ix,ix]
        if ix < npl
            # right
            for jx = (ix+1):1:min(npl, ix + Int((ndiag["out"] - 1) / 2))
                # be_copy_in_hw!(Mout.M[ix, jx], Min.M[ix, jx])
                Mhat.M[pv[ix],pv[jx]] = M.M[ix,jx]
            end
        end
    end

    return Mhat
end

"""
    bndiag_of_inv_pddrgf!(Mout::BlockMatrix, Min::BlockMatrix, auxData::AuxDataPDDRGF, td::TimingData, cd::CountingData)

For an input matrix `M`, possibly but not necessarily block n-diagonal,
where n is 3, 5, etc., compute the block n-diagonal part of the inverse
of `M`. This function uses the paralle RGF method (soon to be extended to DD-RGF).

# Arguments
- `Min::BlockMatrix`: the matrix to be inverted.
- `Mout::BlockMatrix`: the output matrix.
- `auxData`: auxiliary buffers.
- `td`: struct for fine-level (i.e. of the backend kernels) timing. The user can choose no timing,
in which case `td` is an empty `TimingData` struct.
"""
function bndiag_of_inv_pddrgf!(Mout::BlockMatrix, Min::BlockMatrix, auxData::AuxDataPDDRGF, td::TimingData, cd::CountingData)
    # TODO : extend this code to n-diagonal, otherwise rename this function
    #        to keep it as the simple traditional RGF

    # call sequential RGF if nrTasks = 1
    if auxData.nrTasks == 1
        bndiag_of_inv_ddrgf!(Mout, Min, auxData.auxDataSeq, td, cd)
        return
    end

    # tx = TimerOutput()
    # @timeit tx "permute" Mhat = bndiag_of_inv_pddrgf_create_permuted_matrix(Min, permVec)
    # println(tx)

    # println(Base.summarysize(Min))
    # println(Base.summarysize(Mhat))

    # the blocks in this matrix are references to the blocks in Min
    @time MinPerm = bndiag_of_inv_pddrgf_create_permuted_matrix(Min, auxData.permVec)

    # println(auxData.sizeDomains)
    # println(auxData.permVec)
    # println(auxData.permVecInv)

    # first, create permutation vector

    # minusOneCmplx = convert(Min.nrsType, -1.0)
    # plusOneCmplx = convert(Min.nrsType, 1.0)
    # zeroCmplx = convert(Min.nrsType, 0.0)

    # # IMPORTANT : we assume here that all of the blocks in Min and Mout argument
    # #             Array-like, and that those in the block-diagonal of auxData.rgfBuffs.buffM
    # #             are LU-like

    # npl = size(Mout.blockSizes)[1]
    # buffM1 = auxData.buffM
    # # Mout is used as a buffer in multiple places, this is just
    # # labeling for clarity of the implementation
    # buffM2 = Mout
    # buffId = auxData.bIdM

    # # TODO : we might not need a full block n-diagonal as a buffer. To see this,
    # #        go again over the algorithm, first simple RGF, and note that there
    # #        are more LAPACK in-place possibilities (namely, due to getrs! within
    # #        be_mldivide(..))

    # # FIRST, upward pass

    # # bottom element
    # be_lu!(buffM1.M[npl, npl], Min.M[npl, npl], td, cd)

    # # middle elements
    # for ix = npl-1:-1:1
    #     # first run mrdivide, to make use of the mldivide data as a buffer for mrdivide

    #     # this is how we implement be_mrdivide!(..) via be_mldivide!(..)
    #     begin
    #         be_ctranspose!(buffM1.M[ix+1, ix], Min.M[ix, ix+1], td, cd)
    #         be_mldivide!('C', buffM2.M[ix+1, ix], buffM1.M[ix+1, ix], buffM1.M[ix+1, ix+1], td, cd)
    #         be_ctranspose!(buffM1.M[ix, ix+1], buffM2.M[ix+1, ix], td, cd)
    #     end

    #     be_mldivide!('N', buffM1.M[ix+1, ix], Min.M[ix+1, ix], buffM1.M[ix+1, ix+1], td, cd)

    #     be_copy_in_hw!(buffM2.M[ix, ix], Min.M[ix, ix])
    #     be_gemm!('N', 'N', minusOneCmplx, Min.M[ix, ix+1], buffM1.M[ix+1, ix], plusOneCmplx, buffM2.M[ix, ix], td, cd)
    #     be_lu!(buffM1.M[ix, ix], buffM2.M[ix, ix], td, cd)
    # end

    # # THEN, downward pass

    # # top element
    # # using be_mldivide!(..) instead of be_inv_from_lu!(..) because we want
    # # to preallocate everything ourselves and avoid LAPACK from doing it on the fly
    # be_mldivide!('N', Mout.M[1, 1], buffId.M[1, 1], buffM1.M[1, 1], td, cd)

    # # # middle elements
    # for ix = 2:npl
    #     # upper diagonal of Mout
    #     be_gemm!('N', 'N', minusOneCmplx, Mout.M[ix-1, ix-1], buffM1.M[ix-1, ix], zeroCmplx, Mout.M[ix-1, ix], td, cd)

    #     # lower diagonal of Mout
    #     be_gemm!('N', 'N', minusOneCmplx, buffM1.M[ix, ix-1], Mout.M[ix-1, ix-1], zeroCmplx, Mout.M[ix, ix-1], td, cd)

    #     # diagonal of Mout
    #     be_mldivide!('N', Mout.M[ix, ix], buffId.M[ix, ix], buffM1.M[ix, ix], td, cd)
    #     be_gemm!('N', 'N', minusOneCmplx, buffM1.M[ix, ix-1], Mout.M[ix-1, ix], plusOneCmplx, Mout.M[ix, ix], td, cd)
    # end
end