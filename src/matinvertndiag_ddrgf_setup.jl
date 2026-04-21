# function allocate_aux_data_DDRGF(Min::BlockMatrix, splitType::Bool, nrJuliaThreads::Int,
#     nrBLASThreads::Int, td::TimingData, cd::CountingData)::Vector{AuxDataDDRGF}

function allocate_aux_data_DDRGF(Min::BlockMatrix,
    td::TimingData, cd::CountingData)::Vector{AuxDataDDRGF}

    # before anything else, find the optimal parameters for DDRGF

    # IMPORTANT : below, nrLevels=1 means that we have a two-level DDRGF,
    #             i.e., one domain decomposition application and then the
    #             Schur complement inverse sequentially. This implies no
    #             recursive call of DDRGF to itself

    # to do this, first obtain rMLDIV and rLU, and the threaded BLAS factor
    rLUavg::Float64, rLUstd::Float64 = get_rLU(Min, td, cd)

    rMLDIVavg::Float64, rMLDIVstd::Float64 = get_rMLDIV(Min, td, cd)

    # we fix blockSizeD2 = 1, in the paper it's explained why
    blockSizeD2 = 1

    # nrLevels : scalar
    # nrTasks  : array
    # totCost  : scalar
    nrLevels::Int, nrTasksList::Vector{Int}, blockSizeD1List::Vector{Int}, optCost::Float64 =
        opt_params(Min, rLUavg, rMLDIVavg)

    nrBlocksInNonPivotsList = blockSizeD1List

    # allocation of auxiliary data for DDRGF
    listOfAuxDataPar = Vector{AuxDataDDRGF}()

    println(Core.stdout, "\nDDRGF allocations:")

    # fine grid
    auxDataSeq = allocate_aux_data_RGF(Min)
    auxDataPar = allocate_aux_data_DDRGF_single_level(Min, nrBlocksInNonPivotsList[1], auxDataSeq,
        nrTasksList[1])
    push!(listOfAuxDataPar, auxDataPar)

    # coarse grids
    nrDDRGFLevels = nrLevels - 1
    for ix = 1:nrDDRGFLevels-1
        auxDataSeq2 = allocate_aux_data_RGF(listOfAuxDataPar[ix].buffTHat22inv)
        auxDataPar2 = allocate_aux_data_DDRGF_single_level(listOfAuxDataPar[ix].buffTHat22inv,
            nrBlocksInNonPivotsList[ix+1], auxDataSeq2, nrTasksList[ix+1])
        push!(listOfAuxDataPar, auxDataPar2)
    end

    return listOfAuxDataPar
end

"""
	allocate_aux_data_DDRGF_single_level(M::BlockMatrix, nrBlocksInNonPivots::Int, splitType::Bool,
        auxDataSeq::AuxDataRGF, nrTasksBare::Int, nrBLASThreadsOuter::Int, nrBLASThreadsInner::Int)

Allocate some extra buffers in `AuxDataDDRGF` useful for parallel RGF.

# Arguments
- `M::BlockMatrix`: the matrix used as reference.
- `auxDataSeq::AuxDataRGF`: reference to the data pre-allocated already for sequential RGF.
"""
function allocate_aux_data_DDRGF_single_level(M::BlockMatrix, nrBlocksInNonPivots::Int,
    auxDataSeq::AuxDataRGF, nrTasksBare::Int)::AuxDataDDRGF
    nrTasks = nrTasksBare

    # we fix blockSizeD2 = 1, in the paper it's explained why
    blockSizeD2 = 1

    blockSizeD1 = nrBlocksInNonPivots
    nrTasks, blockSizeD1Leftover = bndiag_of_inv_ddrgf_get_nr_tasks(M, blockSizeD1, blockSizeD2)

    lastSizeD1 = blockSizeD1Leftover
    # all the sub-domains in D2 have been forced to have npl=1
    lastSizeD2 = 1

    # nrTasks, blockSizeD1, blockSizeD2, lastSizeD2 = bndiag_of_inv_ddrgf_check_nr_tasks(M, nrBlocksInNonPivots,
    #     nrTasks, splitType)
    # if nrTasks == 1
    #     println("WARNING: nrTasks = 1, then calling sequential RGF.")
    #     # FIXME : the following call to the constructor AuxDataDDRGF(..) is not really correct. Change and call/test
    #     return AuxDataDDRGF(auxDataSeq, nrTasks, Vector{Int}(), Vector{Int}(), Vector{Int}(),
    #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # end

    permVecInv, sizeDomains = bndiag_of_inv_ddrgf_create_permutation_vector(M, nrBlocksInNonPivots)

    permVec = bndiag_of_inv_ddrgf_transpose_permutation_vector(permVecInv)

    # pre-allocate the data for the inverse of \widehat{T}_{11}
    buffTHat = bm_copy(M)

    # pre-allocate full sub-domains for D1 in buffTHat, because it will contain
    # the inverse of \widehat{T}_{11}
    if blockSizeD1 > 2
        jx::Int = 0
        buffBlockSizeD1 = blockSizeD1
        for ix = 1:nrTasks
            if (lastSizeD1 != 0) && (ix == nrTasks)
                buffBlockSizeD1 = lastSizeD1
            end

            if ix == 1
                jx += blockSizeD2 + 1
            elseif ix < nrTasks
                jx += buffBlockSizeD1 + blockSizeD2
            else
                jx += buffBlockSizeD1 + lastSizeD2
            end
            # start and end local indices
            jxStart = jx
            jxEnd = jx + buffBlockSizeD1 - 1
            # slice the sub-matrix with views
            smallMViewBuffTHat = view(buffTHat.M, jxStart:jxEnd, jxStart:jxEnd)
            # build a small BlockMatrix to pass to the defining function
            smallBlockSizes = copy(buffTHat.blockSizes[jxStart:jxEnd])
            nrDiags::Int = buffBlockSizeD1 + (buffBlockSizeD1 - 1)
            smallMbmBuffTHat = BlockMatrix(copy(smallBlockSizes), ArrayOrLU_(undef, jxEnd - jxStart + 1, jxEnd - jxStart + 1),
                Dict("in" => 3, "out" => nrDiags), 0, false)
            bm_blocks_define_complement11!(smallMbmBuffTHat, smallMViewBuffTHat, 2)
        end
    end

    # the appropriate number of threads for good load balance and not wasting energy
    nrThreads, maxNrTasksPerThread, lastNrTasksPerThread = bndiag_of_inv_ddrgf_check_nr_threads(Threads.nthreads(), nrTasks)

    # IMPORTANT : these prints allow us double-checking the DDRGF recursive construction
    println(Core.stdout, "* New level in DDRGF:")
    println(Core.stdout, "\tNumber of threads = " * string(nrThreads))
    println(Core.stdout, "\tNumber of tasks = " * string(nrTasks))
    println(Core.stdout, "\tNumber of layers = " * string(size(M.blockSizes)[1]))
    println(Core.stdout, "\tBlock size D1 = " * string(blockSizeD1))
    println(Core.stdout, "\tBlock size D2 = " * string(blockSizeD2))

    buffMPerm = bndiag_of_inv_ddrgf_create_permuted_matrix(auxDataSeq.buffM, permVec)
    bIdMPerm = bndiag_of_inv_ddrgf_create_permuted_matrix(auxDataSeq.bIdM, permVec)
    buffTHatPerm = bndiag_of_inv_ddrgf_create_permuted_matrix(buffTHat, permVec)

    smallBlockSizes11 = Vector{Vector{Int}}()
    smallAuxDataSeq11 = Vector{AuxDataRGF}()
    smallMbmIn11 = Vector{BlockMatrix}()
    smallMbmOut11 = Vector{BlockMatrix}()
    smallBlockSizes22 = Vector{Vector{Int}}()
    smallMbmIn22 = Vector{BlockMatrix}()
    smallMbmBuffTHat22 = Vector{BlockMatrix}()

    for ix = 1:nrThreads
        push!(smallBlockSizes11, copy(M.blockSizes[1:blockSizeD1]))
        push!(smallAuxDataSeq11, AuxDataRGF(BlockMatrix(copy(smallBlockSizes11[ix]), ArrayOrLU_(undef, blockSizeD1, blockSizeD1),
                buffMPerm.ndiag, 0, false), BlockMatrix(copy(smallBlockSizes11[ix]), ArrayOrLU_(undef, blockSizeD1, blockSizeD1),
                bIdMPerm.ndiag, 0, false), true))
        push!(smallMbmIn11, BlockMatrix(copy(smallBlockSizes11[ix]), ArrayOrLU_(undef, blockSizeD1, blockSizeD1),
            M.ndiag, 0, false))
        push!(smallMbmOut11, BlockMatrix(copy(smallBlockSizes11[ix]), ArrayOrLU_(undef, blockSizeD1, blockSizeD1),
            buffTHatPerm.ndiag, 0, false))
        push!(smallBlockSizes22, copy(M.blockSizes[1:blockSizeD2]))
        push!(smallMbmIn22, BlockMatrix(copy(smallBlockSizes22[ix]), ArrayOrLU_(undef, blockSizeD2, blockSizeD2),
            M.ndiag, 0, false))
        push!(smallMbmBuffTHat22, BlockMatrix(copy(smallBlockSizes22[ix]), ArrayOrLU_(undef, blockSizeD2, blockSizeD2),
            buffTHatPerm.ndiag, 0, false))
    end

    # pre-allocations needed for the inverse of the Schur complement
    nrLayersSchurCompl = sum(sizeDomains[1:nrTasks])
    blockSizesSchurCompl = copy(buffTHatPerm.blockSizes[1:nrLayersSchurCompl])
    buffTHat22inv = BlockMatrix(copy(blockSizesSchurCompl), ArrayOrLU_(undef, nrLayersSchurCompl, nrLayersSchurCompl),
        buffTHatPerm.ndiag, 0, false)
    auxDataSeq22inv = AuxDataRGF(BlockMatrix(copy(blockSizesSchurCompl), ArrayOrLU_(undef, nrLayersSchurCompl, nrLayersSchurCompl),
            buffMPerm.ndiag, 0, false), BlockMatrix(copy(blockSizesSchurCompl), ArrayOrLU_(undef, nrLayersSchurCompl, nrLayersSchurCompl),
            bIdMPerm.ndiag, 0, false), 0)
    buffM222inv = BlockMatrix(copy(blockSizesSchurCompl), ArrayOrLU_(undef, nrLayersSchurCompl, nrLayersSchurCompl),
        buffTHatPerm.ndiag, 0, false)

    # buffer for the output, (independently) available at each level of DDRGF
    buffMout = bm_copy(buffTHat)

    # the final struct with the buffers
    auxDataPar = AuxDataDDRGF(auxDataSeq, nrTasks, permVec, permVecInv, sizeDomains, blockSizeD1,
        blockSizeD2, lastSizeD1, buffTHat, nrThreads, maxNrTasksPerThread, lastNrTasksPerThread, buffMPerm,
        bIdMPerm, buffTHatPerm, smallBlockSizes11, smallAuxDataSeq11, smallMbmIn11, smallMbmOut11, smallBlockSizes22,
        smallMbmIn22, smallMbmBuffTHat22, buffTHat22inv, auxDataSeq22inv, buffM222inv,
        buffMout)

    # add extra allocations for buffTHat, for those little blocks of the Schur
    # complement that make it non embarrasingly parallel
    bm_blocks_define_complement22_non_recurs!(auxDataPar.buffTHat, auxDataPar, 2)
    # and we need those little blocks for the buffM buffer as well
    bm_blocks_define_complement22_non_recurs!(auxDataPar.auxDataSeq.buffM, auxDataPar, 2)

    # and extra allocations for THat_{11}^{-1} * THat_{12} and THat_{21} * THat_{11}^{-1}
    bm_blocks_define_complement12!(auxDataPar.buffTHat, auxDataPar, 2)
    bm_blocks_define_complement21!(auxDataPar.buffTHat, auxDataPar, 2)

    bndiag_of_inv_ddrgf_add_block_refs_to_permuted_matrix22!(auxDataPar.buffMPerm, auxDataPar.auxDataSeq.buffM, auxDataPar)

    # add block references, in buffTHat, for the D1 regions
    bndiag_of_inv_ddrgf_add_block_refs_to_permuted_matrix11!(auxDataPar.buffTHatPerm, auxDataPar)

    # add references to extra Schur complement blocks, those that make it non embarrasingly parallel
    bndiag_of_inv_ddrgf_add_block_refs_to_permuted_matrix22!(auxDataPar.buffTHatPerm, auxDataPar.buffTHat, auxDataPar)

    # add references to extra blocks related to hopping terms interactions, in particular
    # the computation of THat_{11}^{-1} * THat_{12} and THat_{21} * THat_{11}^{-1}
    bndiag_of_inv_ddrgf_add_block_refs_to_permuted_matrix12!(auxDataPar.buffTHatPerm, auxDataPar.buffTHat, auxDataPar)
    bndiag_of_inv_ddrgf_add_block_refs_to_permuted_matrix21!(auxDataPar.buffTHatPerm, auxDataPar.buffTHat, auxDataPar)

    bm_reference!(auxDataPar.buffTHat22inv, buffTHatPerm.M, 0, 0)
    bm_reference!(auxDataPar.auxDataSeq22inv.buffM, buffMPerm.M, 0, 0)
    bm_reference!(auxDataPar.auxDataSeq22inv.bIdM, bIdMPerm.M, 0, 0)

    # the following is needed for storing -1 * THat11Inv * THat12 * THatSInv, but what is needed to
    # compute ( THat11Inv * THat12 * THatSInv ) * THat21 * THat11Inv
    bm_blocks_define_complement12!(auxDataPar.auxDataSeq.buffM, auxDataPar, 2)
    bndiag_of_inv_ddrgf_add_block_refs_to_permuted_matrix12!(auxDataPar.buffMPerm, auxDataPar.auxDataSeq.buffM, auxDataPar)

    # buffer for the output, (independently) available at each level of DDRGF
    bm_blocks_define_complement22_non_recurs!(auxDataPar.buffMout, auxDataPar, 2)

    return auxDataPar
end