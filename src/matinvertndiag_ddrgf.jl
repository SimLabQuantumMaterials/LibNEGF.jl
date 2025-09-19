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
    lastSizeD2::Int
    buffTHat::BlockMatrix
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
function allocate_aux_data_PDDRGF(M::BlockMatrix, nrBlocksInNonPivots::Int, splitType::Bool,
    auxDataSeq::AuxDataDDRGF)::AuxDataPDDRGF
    nrTasks = Threads.nthreads()

    if splitType != 0
        println("ERROR: the code is currently restricted to open-end only.")
        @code_location
        exit()
    end

    if nrBlocksInNonPivots > 3
        println("ERROR: the number of blocks in the 1-1 subdomains is restricted to <= 3 for now.")
        @code_location
        exit()
    end

    # this might change the number of threads to be used
    nrTasks, blockSizeD1, blockSizeD2, lastSizeD2 = bndiag_of_inv_pddrgf_check_nr_tasks(M, nrBlocksInNonPivots,
        nrTasks, splitType)
    if nrTasks == 1
        println("WARNING: nrTasks = 1, then calling sequential RGF.")
        return AuxDataPDDRGF(auxDataSeq, nrTasks, Vector{Int}(), Vector{Int}(), Vector{Int}(), 0, 0, 0, 0)
    end

    permVecInv, sizeDomains = bndiag_of_inv_pddrgf_create_permutation_vector(M, nrTasks, nrBlocksInNonPivots, splitType)

    permVec = bndiag_of_inv_pddrgf_transpose_permutation_vector(permVecInv)

    # pre-allocate the data for the inverse of \widehat{T}_{11}
    buffTHat = bm_copy(M)

    # the final struct with the buffers
    auxDataPar = AuxDataPDDRGF(auxDataSeq, nrTasks, permVec, permVecInv, sizeDomains, blockSizeD1,
        blockSizeD2, lastSizeD2, buffTHat)

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
function bndiag_of_inv_ddrgf!(Mout::BlockMatrix, Min::BlockMatrix, auxData::AuxDataDDRGF, td::TimingData,
    cd::CountingData)
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
function bndiag_of_inv_pddrgf_check_nr_tasks(M::BlockMatrix, nrBlocksInNonPivots::Int, nrTasks::Int,
    splitType::Bool)::Tuple{Int,Int,Int,Int}

    # println(nrTasks)

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
    nrNonPivots = nrTasks

    blockSizeD1 = nrBlocksInNonPivots
    totalSizeD1 = blockSizeD1 * nrNonPivots
    totalSizeD2 = npl - totalSizeD1

    blockSizeD2 = Int(ceil(totalSizeD2 / nrTasks))
    ceilOfTotalSizeD2 = (nrTasks - 1) * blockSizeD2
    restOfTotalSizeD2 = totalSizeD2 - ceilOfTotalSizeD2

    if restOfTotalSizeD2 <= 0
        nrTasks, blockSizeD1, blockSizeD2 = bndiag_of_inv_pddrgf_check_nr_tasks(M, nrBlocksInNonPivots,
            nrTasks - 1, splitType)
    end

    return nrTasks, blockSizeD1, blockSizeD2, restOfTotalSizeD2
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
    nrBlocksInNonPivots::Int, splitType::Bool)::Tuple{Vector{Int},Vector{Int}}
    npl = size(M.blockSizes)[1]
    nrSubdomains::Int = 0
    idSubdomain::Int = 0

    if splitType == Bool(0)
        nrPivots = nrTasks
    else
        nrPivots = nrTasks + 1
    end
    nrNonPivots = nrTasks
    # accounting for D1
    nrSubdomains += nrNonPivots
    # accounting for D2
    nrSubdomains += nrTasks
    sizeDomains = Vector{Int}(undef, nrSubdomains)

    blockSizeD1 = nrBlocksInNonPivots
    totalSizeD1 = blockSizeD1 * nrNonPivots
    totalSizeD2 = npl - totalSizeD1
    blockSizeD2 = ceil(totalSizeD2 / nrTasks)
    lastSizeD2 = totalSizeD2 - blockSizeD2 * (nrTasks - 1)

    permVecInv = Vector{Int}(undef, npl)

    # first, gather all the indices of region 2 (i.e. the pivots)
    ixNew = 0
    ixOld = 0
    for ix = 1:nrPivots
        if ix == nrPivots
            bs = lastSizeD2
        else
            bs = blockSizeD2
        end
        for jx = 1:bs
            ixNew += 1
            ixOld += 1
            permVecInv[ixNew] = ixOld
        end
        idSubdomain += 1
        sizeDomains[idSubdomain] = bs

        ixOld += blockSizeD1
    end

    # then, gather all the indices of region 1
    ixOld = 0
    for ix = 1:nrTasks
        if ix == nrTasks
            ixOld += lastSizeD2
        else
            ixOld += blockSizeD2
        end
        bs = blockSizeD1
        for jx = 1:bs
            ixNew += 1
            ixOld += 1
            permVecInv[ixNew] = ixOld
        end
        idSubdomain += 1
        sizeDomains[idSubdomain] = bs
    end

    return permVecInv, sizeDomains
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
                Mhat.M[pv[ix], pv[jx]] = M.M[ix, jx]
            end
        end
        # center
        Mhat.M[pv[ix], pv[ix]] = M.M[ix, ix]
        if ix < npl
            # right
            for jx = (ix+1):1:min(npl, ix + Int((ndiag["out"] - 1) / 2))
                Mhat.M[pv[ix], pv[jx]] = M.M[ix, jx]
            end
        end
    end

    # we also need to permute the block sizes
    for ix = 1:npl
        Mhat.blockSizes[pv[ix]] = M.blockSizes[ix]
    end

    return Mhat
end

"""
    bndiag_of_inv_pddrgf!(Mout::BlockMatrix, Min::BlockMatrix, auxData::AuxDataPDDRGF, td::TimingData,
    cd::CountingData)

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
function bndiag_of_inv_pddrgf!(Mout_::BlockMatrix, Min_::BlockMatrix, auxData::AuxDataPDDRGF, td::TimingData,
    cd::CountingData)
    # TODO : extend this code to n-diagonal, otherwise rename this function
    #        to keep it as the simple traditional RGF

    # call sequential RGF if nrTasks = 1
    if auxData.nrTasks == 1
        bndiag_of_inv_ddrgf!(Mout_, Min_, auxData.auxDataSeq, td, cd)
        return
    end

    # the blocks in the following matrices contain references to blocks
    # from sequential buffers
    buffM = bndiag_of_inv_pddrgf_create_permuted_matrix(auxData.auxDataSeq.buffM, auxData.permVec)
    bIdM = bndiag_of_inv_pddrgf_create_permuted_matrix(auxData.auxDataSeq.bIdM, auxData.permVec)
    # from parallel buffers
    Min = bndiag_of_inv_pddrgf_create_permuted_matrix(Min_, auxData.permVec)
    Mout = bndiag_of_inv_pddrgf_create_permuted_matrix(Mout_, auxData.permVec)
    buffTHat = bndiag_of_inv_pddrgf_create_permuted_matrix(auxData.buffTHat, auxData.permVec)

    # some relabelings, for clarity and general consistency
    buffM1 = buffM
    # Mout is used as a buffer in multiple places, this is just labeling for clarity of the implementation
    buffM2 = Mout
    # buffM3 will store the inverse of \widehat{T}_{11}
    buffM3 = buffTHat
    buffId = bIdM

    # PART (1,1)

    # first, compute the inverse of \widehat{T}_{11}
    begin
        # first, ensure pre-allocations
        i1 = auxData.nrTasks + 1
        jxStart = sum(auxData.sizeDomains[1:i1-1]) + 1
        jxEnd = sum(auxData.sizeDomains[1:i1])

        smallMViewIn = view(Min.M, jxStart:jxEnd, jxStart:jxEnd)
        smallMViewOut = view(buffM3.M, jxStart:jxEnd, jxStart:jxEnd)
        smallMViewBuffM1 = view(buffM1.M, jxStart:jxEnd, jxStart:jxEnd)
        smallMViewBuffId = view(buffId.M, jxStart:jxEnd, jxStart:jxEnd)

        smallBlockSizes = Min.blockSizes[jxStart:jxEnd]

        smallMbmIn = BlockMatrix(smallBlockSizes, ArrayOrLU_(undef, jxEnd - jxStart + 1, jxEnd - jxStart + 1),
            Min.ndiag, Min.nrsType, 0)
        bm_reference!(smallMbmIn, smallMViewIn)
        smallMbmOut = BlockMatrix(smallBlockSizes, ArrayOrLU_(undef, jxEnd - jxStart + 1, jxEnd - jxStart + 1),
            buffM3.ndiag, buffM3.nrsType, 0)
        bm_reference!(smallMbmOut, smallMViewOut)

        smallAuxDataSeq = AuxDataDDRGF(BlockMatrix(smallBlockSizes, ArrayOrLU_(undef, jxEnd - jxStart + 1, jxEnd - jxStart + 1),
                buffM1.ndiag, buffM1.nrsType, 0), BlockMatrix(smallBlockSizes, ArrayOrLU_(undef, jxEnd - jxStart + 1, jxEnd - jxStart + 1),
                buffId.ndiag, buffId.nrsType, 0))

        bm_reference!(smallAuxDataSeq.buffM, smallMViewBuffM1)
        bm_reference!(smallAuxDataSeq.bIdM, smallMViewBuffId)

        # then, loop over the sub-domains in the D1 domain
        for ix = auxData.nrTasks+1:2*auxData.nrTasks
            jxStart = sum(auxData.sizeDomains[1:ix-1]) + 1
            jxEnd = sum(auxData.sizeDomains[1:ix])

            smallMViewIn = view(Min.M, jxStart:jxEnd, jxStart:jxEnd)
            smallMViewOut = view(buffM3.M, jxStart:jxEnd, jxStart:jxEnd)
            smallMViewBuffM1 = view(buffM1.M, jxStart:jxEnd, jxStart:jxEnd)
            smallMViewBuffId = view(buffId.M, jxStart:jxEnd, jxStart:jxEnd)
            smallBlockSizes = Min.blockSizes[jxStart:jxEnd]

            copy!(smallMbmIn.blockSizes, smallBlockSizes)
            copy!(smallMbmOut.blockSizes, smallBlockSizes)
            copy!(smallAuxDataSeq.buffM.blockSizes, smallBlockSizes)
            copy!(smallAuxDataSeq.bIdM.blockSizes, smallBlockSizes)

            bm_reference!(smallMbmIn, smallMViewIn)
            bm_reference!(smallMbmOut, smallMViewOut)
            bm_reference!(smallAuxDataSeq.buffM, smallMViewBuffM1)
            bm_reference!(smallAuxDataSeq.bIdM, smallMViewBuffId)

            # smallMbmIn = BlockMatrix(smallBlockSizes, smallMViewIn, Min.ndiag, Min.nrsType, 0)
            # smallMbmOut = BlockMatrix(smallBlockSizes, smallMViewOut, Min.ndiag, Min.nrsType, 0)
            # # TODO : modify sequential RGF to give us the two little extra blocks in case of 3x3
            # # TODO : create (and pass) a reference to auxDataSeq that contains buffers as views of
            # #        small slices of the large buffer matrices
            bndiag_of_inv_ddrgf!(smallMbmOut, smallMbmIn, smallAuxDataSeq, TimingData(), CountingData())
        end
    end

    # offsetM = (auxData.nrTasks-1) * auxData.blockSizeD2 + auxData.lastSizeD2
    # smallMView = view(Min.M, 1:auxData.blockSizeD1, 1:auxData.blockSizeD1)
    # Mbfx = BlockMatrix(Min.blockSizes[1:auxData.blockSizeD1], MViewx, Dict("in" => 3, "out" => 3), Min.nrsType, 0)

    # MViewx = view(Min.M, 1:2, 1:2)
    # # Mbfx = BlockMatrix([32,32], ArrayOrLU_(undef, 2, 2),
    # #     Dict("in" => 3, "out" => 3), Min.nrsType, 0)

    # # Mview = view(Min.M, 1, 1)
    # # println(typeof(Mview[1,1]))
    # # println(typeof(Min.M[1,1]))

    # TODO

    # PART (2,2)

    # first, we compute \widetilde{T}_{11}^{-1}
    # TODO !!

    # println(Mout_.blockSizes)

    # TODO : the rest of the implementation
end