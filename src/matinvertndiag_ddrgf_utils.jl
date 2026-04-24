function check_if_enough_mem_ddrgf(npl::Int, blockSize::Int, precx::DataType)
    # total system memory in MB
    totalMem = Sys.total_memory() / 2^20
    requiredMem::Float64 = 0.0

    # RGF-wise

    # +1 for a buffer identity
    rgfN1diag = 1
    # +1 for the sparse original matrix, +1 for the conversion of that
    # original matrix to the input block tridiagonal matrix, +1 for a buffer
    # block tridiagonal matrix in RGF
    rgfN3diag = 3

    # for DDRGF itself

    # +1 buffTHat, +1 buffMout
    ddrgfN3diag = 2
    # IMPORTANT : in the DDRGF case, there isn't really a block diagonal
    # buffer, but as estimating the memory requirement is a bit convoluted to
    # integrate with the already-existing workflow of allocations (see the
    # function allocate_aux_data_DDRGF(...)), then we're adding here an extra
    # block diagonal memory to roughly take into account all of those extra
    # allocations involved in DDRGF
    ddrgfN1diag = 1

    # before allocating, check whether there is enough memory
    # to allocate all the needed buffers
    requiredMem += required_mem_non_symm(npl, precx, rgfN1diag, rgfN3diag, blockSize)
    requiredMem += required_mem_non_symm(npl, precx, ddrgfN1diag, ddrgfN3diag, blockSize)
    #if requiredMem > 0.8*totalMem
    #    error("The required memory exceeds 80% of the total memory")
    #end
end

function get_rMLDIV(M::BlockMatrix, td::TimingData, cd::CountingData)::Vector{Float64}
    # we take the block size to be the average of the block sizes
    avgBlockSize = Int(floor((sum(M.blockSizes) / size(M.blockSizes)[1])))
    # this is just a heuristic, based on experimentation, this perhaps should
    # be formalized a bit better
    nrSamples::Int = floor(1.0E4 * 1.0E5 / (avgBlockSize^3)) + 10

    # we obtain rLU for that average block size

    plusOneCmplx = convert(FieldType, 1.0)

    # create three block-diagonal matrices
    blockSizes = repeat([avgBlockSize], nrSamples)
    A = bm_empty(blockSizes, nrSamples, 1, false, false)
    bm_blocks_define!(A, 2)
    B = bm_empty(blockSizes, nrSamples, 1, false, false)
    bm_blocks_define!(B, 2)
    C = bm_empty(blockSizes, nrSamples, 1, false, false)
    bm_blocks_define!(C, 2)

    # let's pre-run one GEMM, to avoid setup-ish times being accounted for
    be_gemm!('N', 'N', plusOneCmplx, A.M[1, 1], B.M[1, 1], plusOneCmplx, C.M[1, 1], td, cd)

    # let's pre-run one MLDIV, to avoid setup-ish times being accounted for
    Alu = be_zero_lu(avgBlockSize)
    be_lu!(Alu, A.M[1, 1], td, cd)
    be_mldivide!('N', B.M[1, 1], A.M[1, 1], Alu, td, cd)

    timesGEMM::Vector{Float64} = zeros(Float64, nrSamples)
    timesMLDIV::Vector{Float64} = zeros(Float64, nrSamples)

    # get the GEMM times
    for ix = 1:nrSamples
        t1GEMM = time()
        be_gemm!('N', 'N', plusOneCmplx, A.M[ix, ix], B.M[ix, ix], plusOneCmplx, C.M[ix, ix], td, cd)
        t2GEMM = time()
        timesGEMM[ix] = t2GEMM - t1GEMM
    end

    # get the LU times
    for ix = 1:nrSamples
        be_lu!(Alu, A.M[ix, ix], td, cd)
        t1MLDIV = time()
        be_mldivide!('N', C.M[ix, ix], B.M[ix, ix], Alu, td, cd)
        t2MLDIV = time()
        timesMLDIV[ix] = t2MLDIV - t1MLDIV
    end

    # all the ratios MLDIV to GEMM
    rMLDIVs::Vector{Float64} = timesMLDIV ./ timesGEMM

    avgrMLDIV::Float64 = sum(rMLDIVs) / nrSamples
    # standard deviation
    stdrMLDIV::Float64 = sqrt(sum((rMLDIVs .- avgrMLDIV) .^ 2) / nrSamples)

    # suggest cleanup to the garbage collector
    A = nothing
    B = nothing
    C = nothing
    GC.gc()

    return [avgrMLDIV, stdrMLDIV]
end

function get_rLU(M::BlockMatrix, td::TimingData, cd::CountingData)::Vector{Float64}
    # we take the block size to be the average of the block sizes
    avgBlockSize = Int(floor((sum(M.blockSizes) / size(M.blockSizes)[1])))
    # this is just a heuristic, based on experimentation, this perhaps should
    # be formalized a bit better
    nrSamples::Int = floor(1.0E4 * 1.0E5 / (avgBlockSize^3)) + 10

    # we obtain rLU for that average block size

    plusOneCmplx = convert(FieldType, 1.0)

    # create three block-diagonal matrices
    blockSizes = repeat([avgBlockSize], nrSamples)
    A = bm_empty(blockSizes, nrSamples, 1, false, false)
    bm_blocks_define!(A, 2)
    B = bm_empty(blockSizes, nrSamples, 1, false, false)
    bm_blocks_define!(B, 2)
    C = bm_empty(blockSizes, nrSamples, 1, false, false)
    bm_blocks_define!(C, 2)

    # let's pre-run one GEMM, to avoid setup-ish times being accounted for
    be_gemm!('N', 'N', plusOneCmplx, A.M[1, 1], B.M[1, 1], plusOneCmplx, C.M[1, 1], td, cd)

    # let's pre-run one LU, to avoid setup-ish times being accounted for
    Alu = be_zero_lu(avgBlockSize)
    be_lu!(Alu, A.M[1, 1], td, cd)

    timesGEMM::Vector{Float64} = zeros(Float64, nrSamples)
    timesLU::Vector{Float64} = zeros(Float64, nrSamples)

    # get the GEMM times
    for ix = 1:nrSamples
        t1GEMM = time()
        be_gemm!('N', 'N', plusOneCmplx, A.M[ix, ix], B.M[ix, ix], plusOneCmplx, C.M[ix, ix], td, cd)
        t2GEMM = time()
        timesGEMM[ix] = t2GEMM - t1GEMM
    end

    # get the LU times
    for ix = 1:nrSamples
        t1LU = time()
        be_lu!(Alu, A.M[ix, ix], td, cd)
        t2LU = time()
        timesLU[ix] = t2LU - t1LU
    end

    # all the ratios LU to GEMM
    rLUs::Vector{Float64} = timesLU ./ timesGEMM

    avgrLU::Float64 = sum(rLUs) / nrSamples
    # standard deviation
    stdrLU::Float64 = sqrt(sum((rLUs .- avgrLU) .^ 2) / nrSamples)

    # suggest cleanup to the garbage collector
    A = nothing
    B = nothing
    C = nothing
    GC.gc()

    return [avgrLU, stdrLU]
end

function cost_rgf(L::Int, rLU::Float64, rMLDIV::Float64, doFull::Bool)::Float64
    # counts for the three kernels
    nGEMMfull = 0

    if doFull && L > 4
        error("Something is not OK: a block tridiagonal matrix with L>4 is being \
               fully inverted with RGF")
    elseif doFull && L == 3
        nGEMMfull += 1
    elseif doFull && L == 4
        nGEMMfull += 3
    end

    if doFull
        if L == 3
            nGEMMfull += 1 * 2
        elseif L == 4
            nGEMMfull += 2 * 3
        end
    end

    nGEMM = (L - 1) + 3 * (L - 1) + nGEMMfull
    nMLDIV = 2 * (L - 1) + 1 + (L - 1)
    nLU = L

    rRGF::Float64 = nGEMM + rLU * nLU + rMLDIV * nMLDIV
    return rRGF
end

function cost_ddrgf(rLU::Float64, rMLDIV::Float64, nrTasks::Int, nrThreads::Int,
    blockSizeD1::Int)::Float64
    rDDRGF::Float64 = 0.0

    # r11inv term
    rDDRGF += (1.0 / nrThreads) * nrTasks * cost_rgf(blockSizeD1, rLU, rMLDIV, true)

    # r22(12) and r22(21) terms
    rDDRGF += 2.0 * (1.0 / nrThreads) * 2 * nrTasks * blockSizeD1

    # r22S term
    rDDRGF += (1.0 / nrThreads) * 4 * nrTasks

    # r12 term
    rDDRGF += (1.0 / nrThreads) * 4 * nrTasks * blockSizeD1

    # r21 term
    rDDRGF += (1.0 / nrThreads) * 4 * nrTasks

    # r11 term
    rDDRGF += (1.0 / nrThreads) * nrTasks * (4 * (blockSizeD1 - 1) + 2 * blockSizeD1)

    # those are all GEMMs
    rDDRGF *= 1.0

    return rDDRGF
end

function bndiag_of_inv_ddrgf_get_nr_tasks(npl, blockSizeD1::Int,
    blockSizeD2::Int)::Vector{Int}
    # check that blockSizeD2 has been set to 1
    if blockSizeD2 != 1
        error("The value of blockSizeD2 should be set to 1")
    end

    # number of principal layers per task, except possibly the last chunk
    nplPerTask = blockSizeD1 + blockSizeD2

    # get the number of tasks and the (possible) last chunk
    nrTasks = Int(floor(npl / nplPerTask)) + 1
    nplLeftover = npl - (nrTasks - 1) * nplPerTask

    # the leftover gets a 1 removed from the sub-domain 2 within
    blockSizeD1Leftover = nplLeftover - 1

    # possible scenarios at this point:

    # 1. the last chunk is void, which is perfect
    if nplLeftover == 0
        return [nrTasks - 1, 0]
    else
        if blockSizeD1Leftover == 0
            # 2. the last chunk is not void, but the sub-domain 2 within is void.
            # This situation is not possible to handle here, as we're doing open-ended
            # distributions to have a proper load balance of the tasks. We return a -1
            # to indicate that this case doesn't work
            return [-1, 0]
        else
            # 3. this case is ok
            return [nrTasks, blockSizeD1Leftover]
        end
    end
end

function bndiag_of_inv_ddrgf_get_nr_tasks(M::BlockMatrix, blockSizeD1::Int,
    blockSizeD2::Int)::Vector{Int}
    # check that blockSizeD2 has been set to 1
    if blockSizeD2 != 1
        error("The value of blockSizeD2 should be set to 1")
    end

    npl::Int = size(M.blockSizes)[1]

    # number of principal layers per task, except possibly the last chunk
    nplPerTask = blockSizeD1 + blockSizeD2

    # get the number of tasks and the (possible) last chunk
    nrTasks = Int(floor(npl / nplPerTask)) + 1
    nplLeftover = npl - (nrTasks - 1) * nplPerTask

    # the leftover gets a 1 removed from the sub-domain 2 within
    blockSizeD1Leftover = nplLeftover - 1

    # possible scenarios at this point:

    # 1. the last chunk is void, which is perfect
    if nplLeftover == 0
        return [nrTasks - 1, 0]
    else
        if blockSizeD1Leftover == 0
            # 2. the last chunk is not void, but the sub-domain 2 within is void.
            # This situation is not possible to handle here, as we're doing open-ended
            # distributions to have a proper load balance of the tasks. We return a -1
            # to indicate that this case doesn't work
            return [-1, 0]
        else
            # 3. this case is ok
            return [nrTasks, blockSizeD1Leftover]
        end
    end
end

# returns :
# nrLevels : scalar
# nrTasks  : array
# totCost  : scalar
function opt_params(Min::BlockMatrix, rLU::Float64, rMLDIV::Float64)::Tuple{Int,Vector{Int},Vector{Int},Float64}
    # we fix blockSizeD2 = 1, in the paper it's explained why
    blockSizeD2 = 1
    nrThreadsBare = Threads.nthreads()

    nrTasksList = Vector{Int}()
    blockSizeD1List = Vector{Int}()

    npl = size(Min.blockSizes)[1]

    optCostOld::Float64 = Inf
    optCostNew::Float64 = 0.0
    coarseCost::Float64 = 0.0

    nrLevels::Int = 0
    nrTasksPrev::Int = npl

    # this is a loop increasing the number of levels one by one, and will continue looping
    # as long as the cost continues to go down as we increase the number of levels
    while optCostNew < optCostOld
        nrLevels += 1
        if optCostNew != 0
            optCostOld = optCostNew
        end
        optCostNew -= coarseCost

        # we would rather have blockSizeD2 = 4, but it might not always be possible
        for blockSizeD1 = 4:-1:1
            nrTasks, blockSizeD1Leftover = bndiag_of_inv_ddrgf_get_nr_tasks(nrTasksPrev, blockSizeD1, blockSizeD2)
            if nrTasks == -1
                continue
            end

            nrTasksPrev = nrTasks

            nrThreads, maxNrTasksPerThread, lastNrTasksPerThread = bndiag_of_inv_ddrgf_check_nr_threads(nrThreadsBare, nrTasks)
            # the tasks map directy to the coarse grid
            nplCoarse = nrTasks

            # update cost
            optCostNew += cost_ddrgf(rLU, rMLDIV, nrTasks, nrThreads, blockSizeD1)
            coarseCost = cost_rgf(nplCoarse, rLU, rMLDIV, false)

            optCostNew += coarseCost

            push!(nrTasksList, nrTasks)
            push!(blockSizeD1List, blockSizeD1)

            # if we have reached this point, it means that this value of blockSizeD1 is
            # possible, hence we don't need to continue looping over smaller values of it
            break
        end

        # don't let the number of levels grow too much, let's put a cap
        if nrLevels == 6
            break
        end
    end

    return (nrLevels, nrTasksList, blockSizeD1List, optCostNew)
end

function bndiag_of_inv_ddrgf_check_nr_threads(nrThreads::Int, nrTasks::Int)::Tuple{Int,Int,Int}
    maxNrTasksPerThread = Int(ceil(nrTasks / nrThreads))
    ceilOfTotalNrTasksPerThread = (nrThreads - 1) * maxNrTasksPerThread
    restOfTotalNrTasksPerThread = nrTasks - ceilOfTotalNrTasksPerThread
    if restOfTotalNrTasksPerThread <= 0
        nrThreads, maxNrTasksPerThread, restOfTotalNrTasksPerThread = bndiag_of_inv_ddrgf_check_nr_threads(nrThreads - 1, nrTasks)
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
function bndiag_of_inv_ddrgf_create_permutation_vector(M::BlockMatrix,
    nrBlocksInNonPivots::Int)::Tuple{Vector{Int},Vector{Int}}
    npl = size(M.blockSizes)[1]
    nrSubdomains::Int = 0
    idSubdomain::Int = 0

    # if splitType == Bool(0)
    #     nrPivots = nrTasks
    # else
    #     nrPivots = nrTasks + 1
    # end

    blockSizeD2 = 1
    blockSizeD1 = nrBlocksInNonPivots

    nrTasks, blockSizeD1Leftover = bndiag_of_inv_ddrgf_get_nr_tasks(M, blockSizeD1, blockSizeD2)
    lastSizeD1 = blockSizeD1Leftover
    lastSizeD2 = 1

    nrPivots = nrTasks
    nrNonPivots = nrTasks

    # accounting for D1
    nrSubdomains += nrNonPivots
    # accounting for D2
    nrSubdomains += nrTasks

    sizeDomains = Vector{Int}(undef, nrSubdomains)

    totalSizeD1 = 0
    if lastSizeD1 == 0
        totalSizeD1 = nrNonPivots * blockSizeD1
    else
        totalSizeD1 = (nrNonPivots - 1) * blockSizeD1 + lastSizeD1
    end
    totalSizeD2 = npl - totalSizeD1

    # blockSizeD2 = ceil(totalSizeD2 / nrTasks)
    # lastSizeD2 = totalSizeD2 - blockSizeD2 * (nrTasks - 1)

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

        if ix == nrPivots
            if lastSizeD1 == 0
                ixOld += blockSizeD1
            else
                ixOld += lastSizeD1
            end
        else
            ixOld += blockSizeD1
        end
    end

    # then, gather all the indices of region 1
    ixOld = 0
    for ix = 1:nrTasks
        if ix == nrTasks
            ixOld += lastSizeD2
        else
            ixOld += blockSizeD2
        end
        if ix == nrTasks
            if lastSizeD1 == 0
                bs = blockSizeD1
            else
                bs = lastSizeD1
            end
        else
            bs = blockSizeD1
        end
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
function bndiag_of_inv_ddrgf_transpose_permutation_vector(permVec::Vector{Int})::Vector{Int}
    permVecInv = copy(permVec)

    for ix = 1:size(permVec)[1]
        permVecInv[permVec[ix]] = ix
    end

    return permVecInv
end

# this function applies the permutation, specified via permVec, on the input matrix M, but
# returning a matrix whose blocks are references to the blocks in M
function bndiag_of_inv_ddrgf_create_permuted_matrix(M::BlockMatrix, permVec::Vector{Int})::BlockMatrix
    npl = size(M.blockSizes)[1]
    ndiag = M.ndiag
    Mhat = BlockMatrix(copy(M.blockSizes), ArrayOrLU_(undef, npl, npl), M.ndiag, 0, M.isHermitian)
    pv = permVec

    # loop over the block sizes, conversely over the block rows
    for ix = 1:npl
        # now, copy the blocks within the ix-th row
        if !M.isHermitian
            if ix > 1
                # left
                for jx = (ix-1):-1:max(1, ix - Int((ndiag["out"] - 1) / 2))
                    Mhat.M[pv[ix], pv[jx]] = M.M[ix, jx]
                end
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
function bndiag_of_inv_ddrgf_add_block_refs_to_permuted_matrix11!(M::BlockMatrix, auxData::AuxDataDDRGF)
    if auxData.blockSizeD1 > 2
        buffBlockSizeD1 = auxData.blockSizeD1

        # jx1::Int = (auxData.nrTasks - 1) * auxData.blockSizeD2 + auxData.lastSizeD2
        jx1::Int = (auxData.nrTasks - 1) * auxData.blockSizeD2 + 1
        jx2::Int = 0
        for ix = 1:auxData.nrTasks
            if (auxData.lastSizeD1 != 0) && (ix == auxData.nrTasks)
                buffBlockSizeD1 = auxData.lastSizeD1
            end

            if ix == 1
                jx2 += auxData.blockSizeD2
            elseif ix < auxData.nrTasks
                jx2 += buffBlockSizeD1 + auxData.blockSizeD2
            else
                # jx2 += buffBlockSizeD1 + auxData.lastSizeD2
                jx2 += buffBlockSizeD1 + 1
            end

            for ix_ = 1:buffBlockSizeD1
                for jx_ = (ix_-2):-1:1
                    M.M[jx1+ix_, jx1+jx_] = auxData.buffTHat.M[jx2+ix_, jx2+jx_]
                end
                for jx_ = (ix_+2):1:buffBlockSizeD1
                    M.M[jx1+ix_, jx1+jx_] = auxData.buffTHat.M[jx2+ix_, jx2+jx_]
                end
            end

            jx1 += buffBlockSizeD1
        end
    end
end

function bndiag_of_inv_ddrgf_add_block_refs_to_permuted_matrix22!(Mout::BlockMatrix, Min::BlockMatrix, auxData::AuxDataDDRGF)
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

function bndiag_of_inv_ddrgf_add_block_refs_to_permuted_matrix12!(M1::BlockMatrix, M2::BlockMatrix, auxData::AuxDataDDRGF)
    blockSizeD1 = auxData.blockSizeD1
    permVecInv = auxData.permVecInv
    nrTasks = auxData.nrTasks
    sizeDomains = auxData.sizeDomains
    sizeDomains22 = sizeDomains[1:nrTasks]
    sizeDomains11 = sizeDomains[nrTasks+1:2*nrTasks]

    buffBlockSizeD1 = blockSizeD1

    for ix_ = 1:nrTasks
        ixLpermOffset = sum(sizeDomains22) + sum(sizeDomains11[1:ix_-1])

        if (auxData.lastSizeD1 != 0) && (ix_ == nrTasks)
            buffBlockSizeD1 = auxData.lastSizeD1
        end

        # first, the central sub-domain
        jxLperm = sum(sizeDomains22[1:ix_])
        for ix = 2:buffBlockSizeD1
            ixLperm = ixLpermOffset + ix

            ixL = permVecInv[ixLperm]
            jxL = permVecInv[jxLperm]

            M1.M[ixLperm, jxLperm] = M2.M[ixL, jxL]
        end

        if ix_ < nrTasks
            # then, the right sub-domain
            jxLperm = sum(sizeDomains22[1:ix_]) + 1
            for ix = 1:buffBlockSizeD1-1
                ixLperm = ixLpermOffset + ix

                ixL = permVecInv[ixLperm]
                jxL = permVecInv[jxLperm]

                M1.M[ixLperm, jxLperm] = M2.M[ixL, jxL]
            end
        end
    end
end

function bndiag_of_inv_ddrgf_add_block_refs_to_permuted_matrix21!(M1::BlockMatrix, M2::BlockMatrix, auxData::AuxDataDDRGF)
    blockSizeD1 = auxData.blockSizeD1
    permVecInv = auxData.permVecInv
    nrTasks = auxData.nrTasks
    sizeDomains = auxData.sizeDomains
    sizeDomains22 = sizeDomains[1:nrTasks]
    sizeDomains11 = sizeDomains[nrTasks+1:2*nrTasks]

    buffBlockSizeD1 = blockSizeD1

    for jx_ = 1:nrTasks
        jxLpermOffset = sum(sizeDomains22) + sum(sizeDomains11[1:jx_-1])

        if (auxData.lastSizeD1 != 0) && (jx_ == nrTasks)
            buffBlockSizeD1 = auxData.lastSizeD1
        end

        # first, the central sub-domain
        ixLperm = sum(sizeDomains22[1:jx_])
        for jx = 2:buffBlockSizeD1
            jxLperm = jxLpermOffset + jx

            ixL = permVecInv[ixLperm]
            jxL = permVecInv[jxLperm]

            M1.M[ixLperm, jxLperm] = M2.M[ixL, jxL]
        end

        if jx_ < nrTasks
            # then, the right sub-domain
            ixLperm = sum(sizeDomains22[1:jx_]) + 1
            for jx = 1:buffBlockSizeD1-1
                jxLperm = jxLpermOffset + jx

                ixL = permVecInv[ixLperm]
                jxL = permVecInv[jxLperm]

                M1.M[ixLperm, jxLperm] = M2.M[ixL, jxL]
            end
        end
    end
end

function bndiag_of_inv_ddrgf_error_inv_of_T11(Min_::BlockMatrix, Mout_::BlockMatrix,
    auxData::AuxDataDDRGF, td::TimingData, cd::CountingData)::Float64
    plusOneCmplx = convert(FieldType, 1.0)
    zeroCmplx = convert(FieldType, 0.0)
    # 'multiply' the D1 part of Min_ and auxData.buffTHat
    # IMPORTANT : this section of rough code assumes all the layers have
    # the same size
    # TODO : the following assignment of accBlk needs to be changed when the blocks are
    #        not all of the same size
    accBlk = Mout_.M[1, 1]
    Min = bndiag_of_inv_ddrgf_create_permuted_matrix(Min_, auxData.permVec)
    buffTHat = bndiag_of_inv_ddrgf_create_permuted_matrix(auxData.buffTHat, auxData.permVec)
    bndiag_of_inv_ddrgf_add_block_refs_to_permuted_matrix11!(buffTHat, auxData)
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
                    blkId = be_identity(size(accBlk)[1])
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