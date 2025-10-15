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
    buildFullInv::Bool
    nrBLASThreadsOuter::Int
    nrBLASThreadsInner::Int
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
    nrThreads::Int
    maxNrTasksPerThread::Int
    lastNrTasksPerThread::Int
    buffMPerm::BlockMatrix
    bIdMPerm::BlockMatrix
    buffTHatPerm::BlockMatrix
    smallBlockSizes11::Vector{Vector{Int}}
    smallAuxDataSeq11::Vector{AuxDataDDRGF}
    smallMbmIn11::Vector{BlockMatrix}
    smallMbmOut11::Vector{BlockMatrix}
    nrBLASThreadsOuter::Int
    nrBLASThreadsInner::Int
    smallBlockSizes22::Vector{Vector{Int}}
    smallMbmIn22::Vector{BlockMatrix}
    smallMbmBuffTHat22::Vector{BlockMatrix}
end

"""
	allocate_aux_data_DDRGF(M::BlockMatrix)

Based on the block-sparsity pattern of the input matrix `M`, allocate the
buffers in `AuxDataDDRGF`.

# Arguments
- `M::BlockMatrix`: the matrix used as reference.
"""
function allocate_aux_data_DDRGF(M::BlockMatrix, nrBLASThreadsOuter::Int, nrBLASThreadsInner::Int)::AuxDataDDRGF
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
    auxData = AuxDataDDRGF(buffM, bIdM, 0, nrBLASThreadsOuter, nrBLASThreadsInner)

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
    auxDataSeq::AuxDataDDRGF, nrTasksBare::Int, nrBLASThreadsOuter::Int, nrBLASThreadsInner::Int)::AuxDataPDDRGF
    nrTasks = nrTasksBare

    if splitType != 0
        println("ERROR: the code is currently restricted to open-end only.")
        @code_location
        exit()
    end

    # this might change the number of threads to be used
    nrTasks, blockSizeD1, blockSizeD2, lastSizeD2 = bndiag_of_inv_pddrgf_check_nr_tasks(M, nrBlocksInNonPivots,
        nrTasks, splitType)
    println("Actual number of tasks: " * string(nrTasks))
    if nrTasks == 1
        println("WARNING: nrTasks = 1, then calling sequential RGF.")
        # FIXME : the following call to the constructor AuxDataPDDRGF(..) is not really correct. Change and call/test
        return AuxDataPDDRGF(auxDataSeq, nrTasks, Vector{Int}(), Vector{Int}(), Vector{Int}(),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    end

    permVecInv, sizeDomains = bndiag_of_inv_pddrgf_create_permutation_vector(M, nrTasks, nrBlocksInNonPivots, splitType)

    permVec = bndiag_of_inv_pddrgf_transpose_permutation_vector(permVecInv)

    # pre-allocate the data for the inverse of \widehat{T}_{11}
    buffTHat = bm_copy(M)

    # pre-allocate full sub-domains for D1 in buffTHat, because it will contain
    # the inverse of \widehat{T}_{11}
    if blockSizeD1 > 2
        jx::Int = 0
        for ix = 1:nrTasks
            if ix == 1
                jx += blockSizeD2 + 1
            elseif ix < nrTasks
                jx += blockSizeD1 + blockSizeD2
            else
                jx += blockSizeD1 + lastSizeD2
            end
            # start and end local indices
            jxStart = jx
            jxEnd = jx + blockSizeD1 - 1
            # slice the sub-matrix with views
            smallMViewBuffTHat = view(buffTHat.M, jxStart:jxEnd, jxStart:jxEnd)
            # build a small BlockMatrix to pass to the defining function
            smallBlockSizes = buffTHat.blockSizes[jxStart:jxEnd]
            nrDiags::Int = blockSizeD1 + (blockSizeD1 - 1)
            smallMbmBuffTHat = BlockMatrix(smallBlockSizes, ArrayOrLU_(undef, jxEnd - jxStart + 1, jxEnd - jxStart + 1),
                Dict("in" => 3, "out" => nrDiags), buffTHat.nrsType, 0)
            bm_blocks_define_complement11!(smallMbmBuffTHat, smallMViewBuffTHat, 2)
        end
    end

    # the appropriate number of threads for good load balance and not wasting energy
    nrThreads, maxNrTasksPerThread, lastNrTasksPerThread = bndiag_of_inv_pddrgf_check_nr_threads(Threads.nthreads(), nrTasks)
    println("Actual number of threads: " * string(nrThreads))

    buffMPerm = bndiag_of_inv_pddrgf_create_permuted_matrix(auxDataSeq.buffM, permVec)
    bIdMPerm = bndiag_of_inv_pddrgf_create_permuted_matrix(auxDataSeq.bIdM, permVec)
    buffTHatPerm = bndiag_of_inv_pddrgf_create_permuted_matrix(buffTHat, permVec)

    smallBlockSizes11 = Vector{Vector{Int}}()
    smallAuxDataSeq11 = Vector{AuxDataDDRGF}()
    smallMbmIn11 = Vector{BlockMatrix}()
    smallMbmOut11 = Vector{BlockMatrix}()
    smallBlockSizes22 = Vector{Vector{Int}}()
    smallMbmIn22 = Vector{BlockMatrix}()
    smallMbmBuffTHat22 = Vector{BlockMatrix}()
    for ix = 1:nrThreads
        push!(smallBlockSizes11, M.blockSizes[1:blockSizeD1])
        push!(smallAuxDataSeq11, AuxDataDDRGF(BlockMatrix(smallBlockSizes11[ix], ArrayOrLU_(undef, blockSizeD1, blockSizeD1),
                buffMPerm.ndiag, buffMPerm.nrsType, 0), BlockMatrix(smallBlockSizes11[ix], ArrayOrLU_(undef, blockSizeD1, blockSizeD1),
                bIdMPerm.ndiag, bIdMPerm.nrsType, 0), 1, auxDataSeq.nrBLASThreadsOuter, auxDataSeq.nrBLASThreadsInner))
        push!(smallMbmIn11, BlockMatrix(smallBlockSizes11[ix], ArrayOrLU_(undef, blockSizeD1, blockSizeD1),
            M.ndiag, M.nrsType, 0))
        push!(smallMbmOut11, BlockMatrix(smallBlockSizes11[ix], ArrayOrLU_(undef, blockSizeD1, blockSizeD1),
            buffTHatPerm.ndiag, buffTHatPerm.nrsType, 0))
        push!(smallBlockSizes22, M.blockSizes[1:blockSizeD2])
        push!(smallMbmIn22, BlockMatrix(smallBlockSizes22[ix], ArrayOrLU_(undef, blockSizeD2, blockSizeD2),
            M.ndiag, M.nrsType, 0))
        push!(smallMbmBuffTHat22, BlockMatrix(smallBlockSizes22[ix], ArrayOrLU_(undef, blockSizeD2, blockSizeD2),
            buffTHatPerm.ndiag, buffTHatPerm.nrsType, 0))
    end

    # the final struct with the buffers
    auxDataPar = AuxDataPDDRGF(auxDataSeq, nrTasks, permVec, permVecInv, sizeDomains, blockSizeD1,
        blockSizeD2, lastSizeD2, buffTHat, nrThreads, maxNrTasksPerThread, lastNrTasksPerThread, buffMPerm,
        bIdMPerm, buffTHatPerm, smallBlockSizes11, smallAuxDataSeq11, smallMbmIn11, smallMbmOut11, nrBLASThreadsOuter,
        nrBLASThreadsInner, smallBlockSizes22, smallMbmIn22, smallMbmBuffTHat22)

    # add extra allocations for buffTHat, for those little blocks of the Schur
    # complement that make it non embarrasingly parallel
    bm_blocks_define_complement22!(auxDataPar.buffTHat, auxDataPar, 2)
    # and we need those little blocks for the buffM buffer as well
    bm_blocks_define_complement22!(auxDataPar.auxDataSeq.buffM, auxDataPar, 2)
    # and extra allocations for THat_{11}^{-1} * THat_{12} and THat_{21} * THat_{11}^{-1}
    bm_blocks_define_complement12!(auxDataPar, 2)
    bm_blocks_define_complement21!(auxDataPar, 2)

    bndiag_of_inv_pddrgf_add_block_refs_to_permuted_matrix22!(auxDataPar.buffMPerm, auxDataPar.auxDataSeq.buffM, auxDataPar)
    # add block references, in buffTHat, for the D1 regions
    bndiag_of_inv_pddrgf_add_block_refs_to_permuted_matrix11!(auxDataPar.buffTHatPerm, auxDataPar)
    # add references to extra Schur complement blocks, those that make it non embarrasingly parallel
    bndiag_of_inv_pddrgf_add_block_refs_to_permuted_matrix22!(auxDataPar.buffTHatPerm, auxDataPar.buffTHat, auxDataPar)
    # add references to extra blocks related to hopping terms interactions, in particular
    # the computation of THat_{11}^{-1} * THat_{12} and THat_{21} * THat_{11}^{-1}
    bndiag_of_inv_pddrgf_add_block_refs_to_permuted_matrix12!(auxDataPar.buffTHatPerm, auxDataPar)
    bndiag_of_inv_pddrgf_add_block_refs_to_permuted_matrix21!(auxDataPar.buffTHatPerm, auxDataPar)

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
function bndiag_of_inv_ddrgf_local!(Mout::BlockMatrix, Min::BlockMatrix, auxData::AuxDataDDRGF, td::TimingData,
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

function bndiag_of_inv_ddrgf_global!(Mout::BlockMatrix, Min::BlockMatrix, auxData::AuxDataDDRGF, td::TimingData,
    cd::CountingData)
    # set the number of chosen BLAS threads
    LinearAlgebra.BLAS.set_num_threads(auxData.nrBLASThreadsOuter * auxData.nrBLASThreadsInner)

    bndiag_of_inv_ddrgf_local!(Mout, Min, auxData, td, cd)

    # restore the number of BLAS threads to 1
    LinearAlgebra.BLAS.set_num_threads(1)
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
        nrTasks, blockSizeD1, blockSizeD2, restOfTotalSizeD2 = bndiag_of_inv_pddrgf_check_nr_tasks(M, nrBlocksInNonPivots,
            nrTasks - 1, splitType)
    end

    return nrTasks, blockSizeD1, blockSizeD2, restOfTotalSizeD2
end

function bndiag_of_inv_pddrgf_check_nr_threads(nrThreads::Int, nrTasks::Int)::Tuple{Int,Int,Int}
    maxNrTasksPerThread = Int(ceil(nrTasks / nrThreads))
    ceilOfTotalNrTasksPerThread = (nrThreads - 1) * maxNrTasksPerThread
    restOfTotalNrTasksPerThread = nrTasks - ceilOfTotalNrTasksPerThread
    if restOfTotalNrTasksPerThread <= 0
        nrThreads, maxNrTasksPerThread, restOfTotalNrTasksPerThread = bndiag_of_inv_pddrgf_check_nr_threads(nrThreads - 1, nrTasks)
    end

    return nrThreads, maxNrTasksPerThread, restOfTotalNrTasksPerThread
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

# these are references to the extra blocks in the sub-domains in D1, because there we
# need to compute full inverses and not only block tridiagonals
function bndiag_of_inv_pddrgf_add_block_refs_to_permuted_matrix11!(M::BlockMatrix, auxData::AuxDataPDDRGF)
    if auxData.blockSizeD1 > 2
        jx1::Int = (auxData.nrTasks - 1) * auxData.blockSizeD2 + auxData.lastSizeD2
        jx2::Int = 0
        for ix = 1:auxData.nrTasks
            if ix == 1
                jx2 += auxData.blockSizeD2
            elseif ix < auxData.nrTasks
                jx2 += auxData.blockSizeD1 + auxData.blockSizeD2
            else
                jx2 += auxData.blockSizeD1 + auxData.lastSizeD2
            end

            for ix_ = 1:auxData.blockSizeD1
                for jx_ = (ix_-2):-1:1
                    M.M[jx1+ix_, jx1+jx_] = auxData.buffTHat.M[jx2+ix_, jx2+jx_]
                end
                for jx_ = (ix_+2):1:auxData.blockSizeD1
                    M.M[jx1+ix_, jx1+jx_] = auxData.buffTHat.M[jx2+ix_, jx2+jx_]
                end
            end

            jx1 += auxData.blockSizeD1
        end
    end
end

function bndiag_of_inv_pddrgf_add_block_refs_to_permuted_matrix22!(Mout::BlockMatrix, Min::BlockMatrix, auxData::AuxDataPDDRGF)
    blockSizeD2 = auxData.blockSizeD2
    nrTasks = auxData.nrTasks
    permVecInv = auxData.permVecInv

    for ix = 1:nrTasks-1
        # first, the upper block
        ixLperm = blockSizeD2 * ix
        jxLperm = ixLperm + 1
        ixL = permVecInv[ixLperm]
        jxL = permVecInv[jxLperm]

        # M.M[ixLperm, jxLperm] = auxData.buffTHat.M[ixL, jxL]
        Mout.M[ixLperm, jxLperm] = Min.M[ixL, jxL]

        # then, the lower block
        jxLperm = blockSizeD2 * ix
        ixLperm = jxLperm + 1
        ixL = permVecInv[ixLperm]
        jxL = permVecInv[jxLperm]

        # M.M[ixLperm, jxLperm] = auxData.buffTHat.M[ixL, jxL]
        Mout.M[ixLperm, jxLperm] = Min.M[ixL, jxL]
    end

end

function bndiag_of_inv_pddrgf_add_block_refs_to_permuted_matrix12!(M::BlockMatrix, auxData::AuxDataPDDRGF)
    blockSizeD1 = auxData.blockSizeD1
    permVec = auxData.permVec
    permVecInv = auxData.permVecInv
    nrTasks = auxData.nrTasks
    sizeDomains = auxData.sizeDomains
    sizeDomains22 = sizeDomains[1:nrTasks]
    sizeDomains11 = sizeDomains[nrTasks+1:2*nrTasks]

    for ix_ = 1:nrTasks
        ixLpermOffset = sum(sizeDomains22) + sum(sizeDomains11[1:ix_-1])

        # first, the central sub-domain
        jxLperm = sum(sizeDomains22[1:ix_])
        for ix = 2:blockSizeD1
            ixLperm = ixLpermOffset + ix

            ixL = permVecInv[ixLperm]
            jxL = permVecInv[jxLperm]

            M.M[ixLperm, jxLperm] = auxData.buffTHat.M[ixL, jxL]
        end

        if ix_ < nrTasks
            # then, the right sub-domain
            jxLperm = sum(sizeDomains22[1:ix_]) + 1
            for ix = 1:blockSizeD1-1
                ixLperm = ixLpermOffset + ix

                ixL = permVecInv[ixLperm]
                jxL = permVecInv[jxLperm]

                M.M[ixLperm, jxLperm] = auxData.buffTHat.M[ixL, jxL]
            end
        end
    end
end

function bndiag_of_inv_pddrgf_add_block_refs_to_permuted_matrix21!(M::BlockMatrix, auxData::AuxDataPDDRGF)
    blockSizeD1 = auxData.blockSizeD1
    permVecInv = auxData.permVecInv
    nrTasks = auxData.nrTasks
    sizeDomains = auxData.sizeDomains
    sizeDomains22 = sizeDomains[1:nrTasks]
    sizeDomains11 = sizeDomains[nrTasks+1:2*nrTasks]

    for jx_ = 1:nrTasks
        jxLpermOffset = sum(sizeDomains22) + sum(sizeDomains11[1:jx_-1])

        # first, the central sub-domain
        ixLperm = sum(sizeDomains22[1:jx_])
        for jx = 2:blockSizeD1
            jxLperm = jxLpermOffset + jx

            ixL = permVecInv[ixLperm]
            jxL = permVecInv[jxLperm]

            M.M[ixLperm, jxLperm] = auxData.buffTHat.M[ixL, jxL]
        end

        if jx_ < nrTasks
            # then, the right sub-domain
            ixLperm = sum(sizeDomains22[1:jx_]) + 1
            for jx = 1:blockSizeD1-1
                jxLperm = jxLpermOffset + jx

                ixL = permVecInv[ixLperm]
                jxL = permVecInv[jxLperm]

                M.M[ixLperm, jxLperm] = auxData.buffTHat.M[ixL, jxL]
            end
        end
    end
end

function bndiag_of_inv_pddrgf_compute_THat11Inv_x_THat12!(buffTHat::BlockMatrix, Min::BlockMatrix, auxData::AuxDataPDDRGF,
    td::TimingData, cd::CountingData)
    LinearAlgebra.BLAS.set_num_threads(auxData.nrBLASThreadsInner)

    blockSizeD1 = auxData.blockSizeD1
    nrTasks = auxData.nrTasks
    sizeDomains = auxData.sizeDomains
    sizeDomains22 = sizeDomains[1:nrTasks]
    sizeDomains11 = sizeDomains[nrTasks+1:2*nrTasks]

    plusOneCmplx = convert(Min.nrsType, 1.0)
    zeroCmplx = convert(Min.nrsType, 0.0)

    Threads.@threads for ixo = 1:auxData.nrThreads
        if ixo < auxData.nrThreads
            nrTasksPerThread = auxData.maxNrTasksPerThread
        else
            nrTasksPerThread = auxData.lastNrTasksPerThread
        end

        for ixi = 1:nrTasksPerThread
            # index of each individual task
            ix_ = (ixo - 1) * auxData.maxNrTasksPerThread + ixi

            ixLpermOffset = sum(sizeDomains22) + sum(sizeDomains11[1:ix_-1])

            # first, the central sub-domain
            jxLperm = sum(sizeDomains22[1:ix_])
            for ix = 1:blockSizeD1
                ixLperm = ixLpermOffset + ix

                be_gemm!('N', 'N', plusOneCmplx, buffTHat.M[ixLperm, ixLpermOffset+1], Min.M[ixLpermOffset+1, jxLperm],
                    zeroCmplx, buffTHat.M[ixLperm, jxLperm], td, cd)
            end

            if ix_ < nrTasks
                # then, the right sub-domain
                jxLperm = sum(sizeDomains22[1:ix_]) + 1
                for ix = 1:blockSizeD1
                    ixLperm = ixLpermOffset + ix

                    be_gemm!('N', 'N', plusOneCmplx, buffTHat.M[ixLperm, ixLpermOffset+blockSizeD1], Min.M[ixLpermOffset+blockSizeD1, jxLperm],
                        zeroCmplx, buffTHat.M[ixLperm, jxLperm], td, cd)
                end
            end
        end
    end

    LinearAlgebra.BLAS.set_num_threads(1)
end

function bndiag_of_inv_pddrgf_compute_THat21_x_THat11Inv!(buffTHat::BlockMatrix, Min::BlockMatrix, auxData::AuxDataPDDRGF,
    td::TimingData, cd::CountingData)
    LinearAlgebra.BLAS.set_num_threads(auxData.nrBLASThreadsInner)

    blockSizeD1 = auxData.blockSizeD1
    nrTasks = auxData.nrTasks
    sizeDomains = auxData.sizeDomains
    sizeDomains22 = sizeDomains[1:nrTasks]
    sizeDomains11 = sizeDomains[nrTasks+1:2*nrTasks]

    plusOneCmplx = convert(Min.nrsType, 1.0)
    zeroCmplx = convert(Min.nrsType, 0.0)

    Threads.@threads for jxo = 1:auxData.nrThreads
        if jxo < auxData.nrThreads
            nrTasksPerThread = auxData.maxNrTasksPerThread
        else
            nrTasksPerThread = auxData.lastNrTasksPerThread
        end

        for jxi = 1:nrTasksPerThread
            # index of each individual task
            jx_ = (jxo - 1) * auxData.maxNrTasksPerThread + jxi

            jxLpermOffset = sum(sizeDomains22) + sum(sizeDomains11[1:jx_-1])

            # first, the central sub-domain
            ixLperm = sum(sizeDomains22[1:jx_])
            for jx = 1:blockSizeD1
                jxLperm = jxLpermOffset + jx

                be_gemm!('N', 'N', plusOneCmplx, Min.M[ixLperm, jxLpermOffset+1], buffTHat.M[jxLpermOffset+1, jxLperm],
                    zeroCmplx, buffTHat.M[ixLperm, jxLperm], td, cd)
            end

            if jx_ < nrTasks
                # then, the right sub-domain
                ixLperm = sum(sizeDomains22[1:jx_]) + 1
                for jx = 1:blockSizeD1
                    jxLperm = jxLpermOffset + jx

                    be_gemm!('N', 'N', plusOneCmplx, Min.M[ixLperm, jxLpermOffset+blockSizeD1], buffTHat.M[jxLpermOffset+blockSizeD1, jxLperm],
                        zeroCmplx, buffTHat.M[ixLperm, jxLperm], td, cd)
                end
            end
        end
    end

    LinearAlgebra.BLAS.set_num_threads(1)
end

function bndiag_of_inv_pddrgf_inv_of_T11!(Min_::BlockMatrix, auxData::AuxDataPDDRGF, td::TimingData,
    cd::CountingData)
    LinearAlgebra.BLAS.set_num_threads(auxData.nrBLASThreadsInner)

    # the blocks in the following matrices contain references to blocks
    buffM1 = auxData.buffMPerm
    buffId = auxData.bIdMPerm
    # buffM3 will store the inverse of \widehat{T}_{11}
    buffM3 = auxData.buffTHatPerm

    # Min_ is assumed to be permuted already
    Min = Min_

    # then, loop over the sub-domains in the D1 domain

    # to parallelize the following loop, append to its beginning : Threads.@threads
    Threads.@threads for ixo = 1:auxData.nrThreads
        if ixo < auxData.nrThreads
            nrTasksPerThread = auxData.maxNrTasksPerThread
        else
            nrTasksPerThread = auxData.lastNrTasksPerThread
        end

        # per-thread pre-allocations
        smallBlockSizes = auxData.smallBlockSizes11[ixo]
        smallAuxDataSeq = auxData.smallAuxDataSeq11[ixo]
        smallMbmIn = auxData.smallMbmIn11[ixo]
        smallMbmOut = auxData.smallMbmOut11[ixo]

        for ixi = 1:nrTasksPerThread
            # index of each individual task
            ix = auxData.nrTasks + (ixo - 1) * auxData.maxNrTasksPerThread + ixi

            jxStart = sum(auxData.sizeDomains[1:ix-1]) + 1
            jxEnd = sum(auxData.sizeDomains[1:ix])

            copy!(smallBlockSizes, Min.blockSizes[jxStart:jxEnd])

            # making sure we have the correct sizes of the blocks within the domain
            copy!(smallAuxDataSeq.buffM.blockSizes, smallBlockSizes)
            copy!(smallAuxDataSeq.bIdM.blockSizes, smallBlockSizes)
            copy!(smallMbmIn.blockSizes, smallBlockSizes)
            copy!(smallMbmOut.blockSizes, smallBlockSizes)

            # 'pointing' to the appropriate blocks
            bm_reference!(smallMbmIn, Min.M, jxStart - 1, jxStart - 1)
            bm_reference_full!(smallMbmOut, buffM3.M, jxStart - 1, jxStart - 1)
            bm_reference!(smallAuxDataSeq.buffM, buffM1.M, jxStart - 1, jxStart - 1)
            bm_reference!(smallAuxDataSeq.bIdM, buffId.M, jxStart - 1, jxStart - 1)

            # note that RGF has been modified to give us the little extra blocks in the beyond-2x2 cases
            # (i.e., for the number of layers within each sub-domain in D1)
            bndiag_of_inv_ddrgf_local!(smallMbmOut, smallMbmIn, smallAuxDataSeq, td, cd)

        end
    end

    LinearAlgebra.BLAS.set_num_threads(1)
end

function bndiag_of_inv_pddrgf_error_inv_of_T11(Min_::BlockMatrix, Mout_::BlockMatrix,
    auxData::AuxDataPDDRGF, td::TimingData, cd::CountingData)::Float64
    plusOneCmplx = convert(Min_.nrsType, 1.0)
    zeroCmplx = convert(Min_.nrsType, 0.0)
    # 'multiply' the D1 part of Min_ and auxData.buffTHat
    # IMPORTANT : this section of rough code assumes all the layers have
    # the same size
    # TODO : the following assignment of accBlk needs to be changed when the blocks are
    #        not all of the same size
    accBlk = Mout_.M[1, 1]
    Min = bndiag_of_inv_pddrgf_create_permuted_matrix(Min_, auxData.permVec)
    buffTHat = bndiag_of_inv_pddrgf_create_permuted_matrix(auxData.buffTHat, auxData.permVec)
    bndiag_of_inv_pddrgf_add_block_refs_to_permuted_matrix11!(buffTHat, auxData)
    jx::Int = (auxData.nrTasks - 1) * auxData.blockSizeD2 + auxData.lastSizeD2
    numErr::Float64 = 0.0
    denErr::Float64 = 0.0
    for ix = 1:auxData.nrTasks
        for ix_ = 1:auxData.blockSizeD1
            for jx_ = 1:auxData.blockSizeD1
                be_fill!(accBlk, 0)
                # left
                if ix_ > 1
                    kx_ = ix_ - 1
                    be_gemm!('N', 'N', plusOneCmplx, Min.M[jx+ix_, jx+kx_], buffTHat.M[jx+kx_, jx+jx_], zeroCmplx, accBlk, td, cd)
                end
                # center
                kx_ = ix_
                be_gemm!('N', 'N', plusOneCmplx, Min.M[jx+ix_, jx+kx_], buffTHat.M[jx+kx_, jx+jx_], plusOneCmplx, accBlk, td, cd)
                # right
                if ix_ < auxData.blockSizeD1
                    kx_ = ix_ + 1
                    be_gemm!('N', 'N', plusOneCmplx, Min.M[jx+ix_, jx+kx_], buffTHat.M[jx+kx_, jx+jx_], plusOneCmplx, accBlk, td, cd)
                end

                if ix_ == jx_
                    # subtract the identity
                    blkId = be_identity(Min.nrsType, size(accBlk)[1])
                    accBlk -= blkId
                    denErr += convert(Float64, size(accBlk)[1])
                end
                locFrobNorm = LinearAlgebra.norm(accBlk, 2)
                numErr += locFrobNorm * locFrobNorm
            end
        end
        jx += auxData.blockSizeD1
    end

    return sqrt(numErr / denErr)
end

function bndiag_of_inv_pddrgf_build_Schur_compl!(Min_::BlockMatrix, auxData::AuxDataPDDRGF, td::TimingData,
    cd::CountingData)
    LinearAlgebra.BLAS.set_num_threads(auxData.nrBLASThreadsInner)

    minusOneCmplx = convert(Min_.nrsType, -1.0)
    plusOneCmplx = convert(Min_.nrsType, 1.0)
    zeroCmplx = convert(Min_.nrsType, 0.0)

    # the blocks in the following matrices contain references to blocks
    buffTHat = auxData.buffTHatPerm
    Min = Min_

    Threads.@threads for ixo = 1:auxData.nrThreads
        if ixo < auxData.nrThreads
            nrTasksPerThread = auxData.maxNrTasksPerThread
        else
            nrTasksPerThread = auxData.lastNrTasksPerThread
        end

        for ixi = 1:nrTasksPerThread
            # index of each individual task
            ix = (ixo - 1) * auxData.maxNrTasksPerThread + ixi

            # Threads.@threads for ix in 1:Threads.nthreads()
            jx2Start = sum(auxData.sizeDomains[1:ix-1]) + 1
            jx2End = sum(auxData.sizeDomains[1:ix])

            # copy the D2 part of Min into buffTHat

            if ix < auxData.nrTasks
                smallBlockSizes = auxData.smallBlockSizes22[ixo]
                smallMbmIn = auxData.smallMbmIn22[ixo]
                smallMbmBuffTHat = auxData.smallMbmBuffTHat22[ixo]
            else
                smallBlockSizes = Min.blockSizes[jx2Start:jx2End]
                smallMbmIn = BlockMatrix(smallBlockSizes, ArrayOrLU_(undef, jx2End - jx2Start + 1, jx2End - jx2Start + 1),
                    Min.ndiag, Min.nrsType, 0)
                smallMbmBuffTHat = BlockMatrix(smallBlockSizes, ArrayOrLU_(undef, jx2End - jx2Start + 1, jx2End - jx2Start + 1),
                    buffTHat.ndiag, buffTHat.nrsType, 0)
            end

            bm_reference!(smallMbmIn, Min.M, jx2Start - 1, jx2Start - 1)
            bm_reference!(smallMbmBuffTHat, buffTHat.M, jx2Start - 1, jx2Start - 1)

            # copy THat22^k2 into THatS^k2
            bm_copy!(smallMbmBuffTHat, smallMbmIn)

            begin
                jx1Start = sum(auxData.sizeDomains[1:auxData.nrTasks+ix-1]) + 1

                # # in the notation of the paper:
                # # THatS^k2
                # THatS_k2 = smallMbmBuffTHat.M
                # # THat12_k2k2
                # THat12_k2k2 = view(Min.M, jx1Start:jx1End, jx2Start:jx2End)
                # # buffer for the product of THat21_k2k2 times ((THat_11)^-1)^k2
                # THat21_k2k2_buff = view(buffTHat.M, jx2Start:jx2End, jx1Start:jx1End)
                # be_gemm!('N', 'N', minusOneCmplx, THat21_k2k2_buff[auxData.sizeDomains[ix], 1], THat12_k2k2[1, auxData.sizeDomains[ix]],
                #     plusOneCmplx, THatS_k2[auxData.sizeDomains[ix], auxData.sizeDomains[ix]], td, cd)

                be_gemm!('N', 'N', minusOneCmplx,
                    buffTHat.M[jx2Start-1+auxData.sizeDomains[ix], jx1Start-1+1],
                    Min.M[jx1Start-1+1, jx2Start-1+auxData.sizeDomains[ix]],
                    plusOneCmplx,
                    smallMbmBuffTHat.M[auxData.sizeDomains[ix], auxData.sizeDomains[ix]],
                    td, cd)
            end

            if ix > 1
                # running on index 2
                ixm1 = ix - 1
                # running on index 1
                ixm1_s = auxData.nrTasks + ixm1

                jx1Start = sum(auxData.sizeDomains[1:auxData.nrTasks+ixm1-1]) + 1

                # # in the notation of the paper:
                # # THatS^k2
                # THatS_k2 = smallMbmBuffTHat.M
                # # THat12_k2k2
                # THat12_k2k2 = view(Min.M, jx1Start:jx1End, jx2Start:jx2End)
                # # buffer for the product of THat21_k2k2 times ((THat_11)^-1)^k2
                # THat21_k2k2_buff = view(buffTHat.M, jx2Start:jx2End, jx1Start:jx1End)
                # be_gemm!('N', 'N', minusOneCmplx, THat21_k2k2_buff[1, auxData.sizeDomains[ixm1_s]], THat12_k2k2[auxData.sizeDomains[ixm1_s], 1],
                #     plusOneCmplx, THatS_k2[1, 1], td, cd)

                be_gemm!('N', 'N', minusOneCmplx,
                    buffTHat.M[jx2Start-1+1, jx1Start-1+auxData.sizeDomains[ixm1_s]],
                    Min.M[jx1Start-1+auxData.sizeDomains[ixm1_s], jx2Start-1+1],
                    plusOneCmplx,
                    smallMbmBuffTHat.M[1, 1],
                    td, cd)
            end

            if ix < auxData.nrTasks
                # compute here those blocks that make the Schur complement non embarrasingly parallel. take into
                # account the pre-computed THat_{11}^{-1} * THat_{12} and THat_{21} * THat_{11}^{-1}

                jx2p1Start = sum(auxData.sizeDomains[1:ix+1-1]) + 1
                jx1Start = sum(auxData.sizeDomains[1:auxData.nrTasks+ix-1]) + 1

                ix_s = auxData.nrTasks + ix

                # the right block
                begin
                    # # in the notation of the paper:
                    # # THatS^k2k2p1
                    # THatS_k2k2p1 = view(buffTHat.M, jx2Start:jx2End, jx2p1Start:jx2p1End)
                    # # THat12_k2k2p1
                    # THat12_k2k2p1 = view(Min.M, jx1Start:jx1End, jx2p1Start:jx2p1End)
                    # # buffer for the product of THat21_k2k2 times ((THat_11)^-1)^k2
                    # THat21_k2k2_buff = view(buffTHat.M, jx2Start:jx2End, jx1Start:jx1End)
                    # be_gemm!('N', 'N', minusOneCmplx, THat21_k2k2_buff[auxData.sizeDomains[ix], auxData.sizeDomains[ix_s]],
                    #     THat12_k2k2p1[auxData.sizeDomains[ix_s], 1],
                    #     zeroCmplx, THatS_k2k2p1[auxData.sizeDomains[ix], 1], td, cd)

                    be_gemm!('N', 'N', minusOneCmplx,
                        buffTHat.M[jx2Start-1+auxData.sizeDomains[ix], jx1Start-1+auxData.sizeDomains[ix_s]],
                        Min.M[jx1Start-1+auxData.sizeDomains[ix_s], jx2p1Start-1+1],
                        zeroCmplx,
                        buffTHat.M[jx2Start-1+auxData.sizeDomains[ix], jx2p1Start-1+1],
                        td, cd)
                end

                # the left block
                begin
                    # # in the notation of the paper:
                    # # THatS^k2k2p1
                    # THatS_k2p1k2 = view(buffTHat.M, jx2p1Start:jx2p1End, jx2Start:jx2End)
                    # # THat21k2p1k2
                    # THat21_k2p1k2 = view(Min.M, jx2p1Start:jx2p1End, jx1Start:jx1End)
                    # # THat12_k2k2_buff
                    # THat12_k2k2_buff = view(buffTHat.M, jx1Start:jx1End, jx2Start:jx2End)
                    # be_gemm!('N', 'N', minusOneCmplx, THat21_k2p1k2[1, auxData.sizeDomains[ix_s]],
                    #     THat12_k2k2_buff[auxData.sizeDomains[ix_s], auxData.sizeDomains[ix]],
                    #     zeroCmplx, THatS_k2p1k2[1, auxData.sizeDomains[ix]], td, cd)

                    be_gemm!('N', 'N', minusOneCmplx,
                        Min.M[jx2p1Start-1+1, jx1Start-1+auxData.sizeDomains[ix_s]],
                        buffTHat.M[jx1Start-1+auxData.sizeDomains[ix_s], jx2Start-1+auxData.sizeDomains[ix]],
                        zeroCmplx,
                        buffTHat.M[jx2p1Start-1+1, jx2Start-1+auxData.sizeDomains[ix]],
                        td, cd)
                end
            end
        end

    end

    LinearAlgebra.BLAS.set_num_threads(1)
end

function bndiag_of_inv_pddrgf_inv_Schur_compl!(Mout_::BlockMatrix, auxData::AuxDataPDDRGF, td::TimingData,
    cd::CountingData)
    # the blocks in the following matrices contain references to blocks
    buffM1 = auxData.buffMPerm
    buffId = auxData.bIdMPerm
    buffTHat = auxData.buffTHatPerm
    buffM2 = Mout_

    # TODO : for the whole code in this function, change the code to make
    #        use of memory pre-allocations (as in the T11 inverse function)

    nrLayersSchurCompl = sum(auxData.sizeDomains[1:auxData.nrTasks])
    blockSizesSchurCompl = buffTHat.blockSizes[1:nrLayersSchurCompl]

    buffTHat22 = BlockMatrix(blockSizesSchurCompl, ArrayOrLU_(undef, nrLayersSchurCompl, nrLayersSchurCompl),
        buffTHat.ndiag, buffTHat.nrsType, 0)
    bm_reference!(buffTHat22, buffTHat.M, 0, 0)

    auxDataSeq22 = AuxDataDDRGF(BlockMatrix(blockSizesSchurCompl, ArrayOrLU_(undef, nrLayersSchurCompl, nrLayersSchurCompl),
            buffM1.ndiag, buffM1.nrsType, 0), BlockMatrix(blockSizesSchurCompl, ArrayOrLU_(undef, nrLayersSchurCompl, nrLayersSchurCompl),
            buffId.ndiag, buffId.nrsType, 0), 0, auxData.nrBLASThreadsOuter, auxData.nrBLASThreadsInner)
    bm_reference!(auxDataSeq22.buffM, buffM1.M, 0, 0)
    bm_reference!(auxDataSeq22.bIdM, buffId.M, 0, 0)

    buffM222 = BlockMatrix(blockSizesSchurCompl, ArrayOrLU_(undef, nrLayersSchurCompl, nrLayersSchurCompl),
        buffM2.ndiag, buffM2.nrsType, 0)
    bm_reference!(buffM222, buffM2.M, 0, 0)

    @time bndiag_of_inv_ddrgf_global!(buffM222, buffTHat22, auxDataSeq22, td, cd)
end

function bndiag_of_inv_pddrgf_inv_of_Schur_compl!(Mout_::BlockMatrix, Min_::BlockMatrix, auxData::AuxDataPDDRGF, td::TimingData,
    cd::CountingData)

    # the blocks in the following matrices contain references to blocks
    buffTHat = auxData.buffTHatPerm
    Min = Min_

    # compute the nonzero blocks in THat_{11}^{-1} * THat_{12}, saving the output to the 12 and 21 parts of buffTHat
    @time bndiag_of_inv_pddrgf_compute_THat11Inv_x_THat12!(buffTHat, Min, auxData, td, cd)
    # and then those of THat_{21} * THat_{11}^{-1}
    @time bndiag_of_inv_pddrgf_compute_THat21_x_THat11Inv!(buffTHat, Min, auxData, td, cd)

    # the D2 part of buffTHat contains the (approximated) Schur complement
    bndiag_of_inv_pddrgf_build_Schur_compl!(Min, auxData, td, cd)

    # call sequential RGF to compute the inverse of the Schur complement
    bndiag_of_inv_pddrgf_inv_Schur_compl!(Mout_, auxData, td, cd)
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
        bndiag_of_inv_ddrgf_global!(Mout_, Min_, auxData.auxDataSeq, td, cd)
        return
    end

    # PART (1,1)

    Min = bndiag_of_inv_pddrgf_create_permuted_matrix(Min_, auxData.permVec)
    Mout = bndiag_of_inv_pddrgf_create_permuted_matrix(Mout_, auxData.permVec)
    bndiag_of_inv_pddrgf_add_block_refs_to_permuted_matrix22!(Mout, Mout_, auxData)

    # first, compute the inverse of \widehat{T}_{11}, storing it in the D1 part of auxData.buffTHat
    bndiag_of_inv_pddrgf_inv_of_T11!(Min, auxData, td, cd)

    # with the inverse of \widehat{T}_{11} at hand, construct the Schur complement, and
    # then invert it, stored in the D2 part of Mout_
    bndiag_of_inv_pddrgf_inv_of_Schur_compl!(Mout, Min, auxData, td, cd)

    # TODO : the rest of the implementation
end