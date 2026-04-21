struct AuxDataDDRGF
    # the sequential data
    auxDataSeq::AuxDataRGF
    # the extra (parallel-related) params
    nrTasks::Int
    permVec::Vector{Int}
    permVecInv::Vector{Int}
    sizeDomains::Vector{Int}
    blockSizeD1::Int
    blockSizeD2::Int
    lastSizeD1::Int
    buffTHat::BlockMatrix
    nrThreads::Int
    maxNrTasksPerThread::Int
    lastNrTasksPerThread::Int
    buffMPerm::BlockMatrix
    bIdMPerm::BlockMatrix
    buffTHatPerm::BlockMatrix
    smallBlockSizes11::Vector{Vector{Int}}
    smallAuxDataSeq11::Vector{AuxDataRGF}
    smallMbmIn11::Vector{BlockMatrix}
    smallMbmOut11::Vector{BlockMatrix}
    smallBlockSizes22::Vector{Vector{Int}}
    smallMbmIn22::Vector{BlockMatrix}
    smallMbmBuffTHat22::Vector{BlockMatrix}
    buffTHat22inv::BlockMatrix
    auxDataSeq22inv::AuxDataRGF
    buffM222inv::BlockMatrix
    buffMout::BlockMatrix
end

function bndiag_of_inv_rgf_global!(Mout::BlockMatrix, Min::BlockMatrix, auxData::AuxDataRGF, td::TimingData,
    cd::CountingData)

    # we call this function from within DDRGF, at the "coarsest" level

    LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())

    bndiag_of_inv_rgf_local!(Mout, Min, auxData, td, cd)

    # restore the number of BLAS threads to 1
    #if auxData.nrBLASThreadsOuter * auxData.nrBLASThreadsInner > 1
    #    LinearAlgebra.BLAS.set_num_threads(1)
    #end
    LinearAlgebra.BLAS.set_num_threads(1)
end

function bndiag_of_inv_ddrgf_compute_THat11Inv_x_THat12!(Min::BlockMatrix, auxData::AuxDataDDRGF,
    td::TimingData, cd::CountingData)

    blockSizeD1 = auxData.blockSizeD1
    nrTasks = auxData.nrTasks
    sizeDomains = auxData.sizeDomains
    sizeDomains22 = sizeDomains[1:nrTasks]
    sizeDomains11 = sizeDomains[nrTasks+1:2*nrTasks]

    buffTHat = auxData.buffTHatPerm

    plusOneCmplx = convert(FieldType, 1.0)
    zeroCmplx = convert(FieldType, 0.0)

    iOffset = sum(sizeDomains22)

    Threads.@threads for ixo = 1:auxData.nrThreads
        buffBlockSizeD1 = blockSizeD1

        if ixo < auxData.nrThreads
            nrTasksPerThread = auxData.maxNrTasksPerThread
        else
            nrTasksPerThread = auxData.lastNrTasksPerThread
        end

        iAccum = sum(sizeDomains11[1:(ixo-1)*auxData.maxNrTasksPerThread])
        jAccum = sum(sizeDomains22[1:(ixo-1)*auxData.maxNrTasksPerThread])
        for ixi = 1:nrTasksPerThread
            # index of each individual task
            ix_ = (ixo - 1) * auxData.maxNrTasksPerThread + ixi

            if (auxData.lastSizeD1 != 0) && (ix_ == auxData.nrTasks)
                # if ixo == 1
                #     println("this is weird")
                # end
                buffBlockSizeD1 = auxData.lastSizeD1
            end

            if ixi > 1
                iAccum += sizeDomains11[ix_-1]
            end
            ixLpermOffset = iOffset + iAccum

            # first, the central sub-domain
            jAccum += sizeDomains22[ix_]
            jxLperm = jAccum
            for ix = 1:buffBlockSizeD1
                ixLperm = ixLpermOffset + ix

                be_gemm!('N', 'N', plusOneCmplx, buffTHat.M[ixLperm, ixLpermOffset+1], Min.M[ixLpermOffset+1, jxLperm],
                    zeroCmplx, buffTHat.M[ixLperm, jxLperm], td, cd)
            end

            if ix_ < nrTasks
                # then, the right sub-domain
                jxLperm += 1
                for ix = 1:buffBlockSizeD1
                    ixLperm = ixLpermOffset + ix

                    be_gemm!('N', 'N', plusOneCmplx, buffTHat.M[ixLperm, ixLpermOffset+buffBlockSizeD1],
                        Min.M[ixLpermOffset+buffBlockSizeD1, jxLperm],
                        zeroCmplx, buffTHat.M[ixLperm, jxLperm], td, cd)
                end
            end
        end
    end
end

function bndiag_of_inv_ddrgf_compute_minus_THat11Inv_x_THat12_x_THatSInv!(Mout::BlockMatrix, auxData::AuxDataDDRGF,
    td::TimingData, cd::CountingData)

    blockSizeD1 = auxData.blockSizeD1
    nrTasks = auxData.nrTasks
    sizeDomains = auxData.sizeDomains
    sizeDomains22 = sizeDomains[1:nrTasks]
    sizeDomains11 = sizeDomains[nrTasks+1:2*nrTasks]

    buffTHat = auxData.buffTHatPerm
    buffM = auxData.buffMPerm

    plusOneCmplx = convert(FieldType, 1.0)
    minusOneCmplx = convert(FieldType, -1.0)
    zeroCmplx = convert(FieldType, 0.0)

    iOffset = sum(sizeDomains22)

    Threads.@threads for ixo = 1:auxData.nrThreads
        buffBlockSizeD1 = blockSizeD1

        if ixo < auxData.nrThreads
            nrTasksPerThread = auxData.maxNrTasksPerThread
        else
            nrTasksPerThread = auxData.lastNrTasksPerThread
        end

        iAccum = sum(sizeDomains11[1:(ixo-1)*auxData.maxNrTasksPerThread])
        jAccum = sum(sizeDomains22[1:(ixo-1)*auxData.maxNrTasksPerThread])
        for ixi = 1:nrTasksPerThread
            # index of each individual task
            ix_ = (ixo - 1) * auxData.maxNrTasksPerThread + ixi

            if (auxData.lastSizeD1 != 0) && (ix_ == auxData.nrTasks)
                buffBlockSizeD1 = auxData.lastSizeD1
            end

            if ixi > 1
                iAccum += sizeDomains11[ix_-1]
            end
            ixLpermOffset = iOffset + iAccum

            # first, the central sub-domain
            jAccum += sizeDomains22[ix_]
            jxLperm = jAccum
            for ix = 1:buffBlockSizeD1
                ixLperm = ixLpermOffset + ix

                # buffM.M[ixLperm, jxLperm] = - buffTHat.M[ixLperm, jxLperm] * buffTHat.M[jxLperm, jxLperm]
                # buffM.M[ixLperm, jxLperm] -= buffTHat.M[ixLperm, jxLperm+1] * buffTHat.M[jxLperm+1, jxLperm]

                be_gemm!('N', 'N', minusOneCmplx, buffTHat.M[ixLperm, jxLperm], Mout.M[jxLperm, jxLperm],
                    zeroCmplx, buffM.M[ixLperm, jxLperm], td, cd)
                if ix_ < nrTasks
                    be_gemm!('N', 'N', minusOneCmplx, buffTHat.M[ixLperm, jxLperm+1], Mout.M[jxLperm+1, jxLperm],
                        plusOneCmplx, buffM.M[ixLperm, jxLperm], td, cd)
                end

                # copy the element that goes directly to the output
                if ix == 1
                    be_copy_in_hw!(Mout.M[ixLperm, jxLperm], buffM.M[ixLperm, jxLperm])
                end
            end

            if ix_ < nrTasks
                # then, the right sub-domain
                jxLperm += 1
                for ix = 1:buffBlockSizeD1
                    ixLperm = ixLpermOffset + ix

                    # buffM.M[ixLperm, jxLperm] = - buffTHat[ixLperm, jxLperm-1] * buffTHat[jxLperm-1, jxLperm]
                    # buffM.M[ixLperm, jxLperm] -= buffTHat[ixLperm, jxLperm] * buffTHat[jxLperm, jxLperm]

                    be_gemm!('N', 'N', minusOneCmplx, buffTHat.M[ixLperm, jxLperm-1], Mout.M[jxLperm-1, jxLperm],
                        zeroCmplx, buffM.M[ixLperm, jxLperm], td, cd)
                    be_gemm!('N', 'N', minusOneCmplx, buffTHat.M[ixLperm, jxLperm], Mout.M[jxLperm, jxLperm],
                        plusOneCmplx, buffM.M[ixLperm, jxLperm], td, cd)

                    if ix == buffBlockSizeD1
                        be_copy_in_hw!(Mout.M[ixLperm, jxLperm], buffM.M[ixLperm, jxLperm])
                    end
                end
            end
        end
    end
end

function bndiag_of_inv_ddrgf_compute_THat21_x_THat11Inv!(Min::BlockMatrix, auxData::AuxDataDDRGF,
    td::TimingData, cd::CountingData)

    blockSizeD1 = auxData.blockSizeD1
    nrTasks = auxData.nrTasks
    sizeDomains = auxData.sizeDomains
    sizeDomains22 = sizeDomains[1:nrTasks]
    sizeDomains11 = sizeDomains[nrTasks+1:2*nrTasks]

    buffTHat = auxData.buffTHatPerm

    plusOneCmplx = convert(FieldType, 1.0)
    zeroCmplx = convert(FieldType, 0.0)

    jOffset = sum(sizeDomains22)

    Threads.@threads for jxo = 1:auxData.nrThreads
        buffBlockSizeD1 = blockSizeD1

        if jxo < auxData.nrThreads
            nrTasksPerThread = auxData.maxNrTasksPerThread
        else
            nrTasksPerThread = auxData.lastNrTasksPerThread
        end

        jAccum = sum(sizeDomains11[1:(jxo-1)*auxData.maxNrTasksPerThread])
        iAccum = sum(sizeDomains22[1:(jxo-1)*auxData.maxNrTasksPerThread])
        for jxi = 1:nrTasksPerThread
            # index of each individual task
            jx_ = (jxo - 1) * auxData.maxNrTasksPerThread + jxi

            if (auxData.lastSizeD1 != 0) && (jx_ == auxData.nrTasks)
                buffBlockSizeD1 = auxData.lastSizeD1
            end

            if jxi > 1
                jAccum += sizeDomains11[jx_-1]
            end
            jxLpermOffset = jOffset + jAccum

            # first, the central sub-domain
            iAccum += sizeDomains22[jx_]
            ixLperm = iAccum
            for jx = 1:buffBlockSizeD1
                jxLperm = jxLpermOffset + jx

                be_gemm!('N', 'N', plusOneCmplx, Min.M[ixLperm, jxLpermOffset+1], buffTHat.M[jxLpermOffset+1, jxLperm],
                    zeroCmplx, buffTHat.M[ixLperm, jxLperm], td, cd)
            end

            if jx_ < nrTasks
                # then, the right sub-domain
                ixLperm += 1
                for jx = 1:buffBlockSizeD1
                    jxLperm = jxLpermOffset + jx

                    be_gemm!('N', 'N', plusOneCmplx, Min.M[ixLperm, jxLpermOffset+buffBlockSizeD1],
                        buffTHat.M[jxLpermOffset+buffBlockSizeD1, jxLperm],
                        zeroCmplx, buffTHat.M[ixLperm, jxLperm], td, cd)
                end
            end
        end
    end
end

function bndiag_of_inv_ddrgf_compute_minus_x_THatSInv_THat21_x_THat11Inv!(Mout::BlockMatrix, auxData::AuxDataDDRGF,
    td::TimingData, cd::CountingData)

    # TODO : based on analyzing this function a bit more, can we reduce the cost of the
    #        function bndiag_of_inv_ddrgf_compute_THat21_x_THat11Inv!(...) ?

    blockSizeD1 = auxData.blockSizeD1
    nrTasks = auxData.nrTasks
    sizeDomains = auxData.sizeDomains
    sizeDomains22 = sizeDomains[1:nrTasks]
    sizeDomains11 = sizeDomains[nrTasks+1:2*nrTasks]

    buffTHat = auxData.buffTHatPerm

    plusOneCmplx = convert(FieldType, 1.0)
    zeroCmplx = convert(FieldType, 0.0)
    minusOneCmplx = convert(FieldType, -1.0)

    jOffset = sum(sizeDomains22)

    Threads.@threads for jxo = 1:auxData.nrThreads
        buffBlockSizeD1 = blockSizeD1

        if jxo < auxData.nrThreads
            nrTasksPerThread = auxData.maxNrTasksPerThread
        else
            nrTasksPerThread = auxData.lastNrTasksPerThread
        end

        jAccum = sum(sizeDomains11[1:(jxo-1)*auxData.maxNrTasksPerThread])
        iAccum = sum(sizeDomains22[1:(jxo-1)*auxData.maxNrTasksPerThread])
        for jxi = 1:nrTasksPerThread
            # index of each individual task
            jx_ = (jxo - 1) * auxData.maxNrTasksPerThread + jxi

            if auxData.lastSizeD1 != 0 && jx_ == auxData.nrTasks
                buffBlockSizeD1 = auxData.lastSizeD1
            end

            if jxi > 1
                jAccum += sizeDomains11[jx_-1]
            end
            jxLpermOffset = jOffset + jAccum

            # first, the central sub-domain
            iAccum += sizeDomains22[jx_]
            ixLperm = iAccum
            for jx = 1:buffBlockSizeD1
                jxLperm = jxLpermOffset + jx

                if jx == 1
                    be_gemm!('N', 'N', minusOneCmplx, Mout.M[ixLperm, ixLperm], buffTHat.M[ixLperm, jxLperm],
                        zeroCmplx, Mout.M[ixLperm, jxLperm], td, cd)
                    if jx_ < nrTasks
                        be_gemm!('N', 'N', minusOneCmplx, Mout.M[ixLperm, ixLperm+1], buffTHat.M[ixLperm+1, jxLperm],
                            plusOneCmplx, Mout.M[ixLperm, jxLperm], td, cd)
                    end
                end
            end

            if jx_ < nrTasks
                # then, the bottom sub-domain
                ixLperm += 1
                for jx = 1:buffBlockSizeD1
                    jxLperm = jxLpermOffset + jx

                    if jx == buffBlockSizeD1
                        be_gemm!('N', 'N', minusOneCmplx, Mout.M[ixLperm, ixLperm-1], buffTHat.M[ixLperm-1, jxLperm],
                            zeroCmplx, Mout.M[ixLperm, jxLperm], td, cd)
                        be_gemm!('N', 'N', minusOneCmplx, Mout.M[ixLperm, ixLperm], buffTHat.M[ixLperm, jxLperm],
                            plusOneCmplx, Mout.M[ixLperm, jxLperm], td, cd)
                    end
                end
            end
        end
    end
end

function bndiag_of_inv_ddrgf_compute_11_part!(Mout::BlockMatrix, auxData::AuxDataDDRGF,
    td::TimingData, cd::CountingData)

    blockSizeD1 = auxData.blockSizeD1
    nrTasks = auxData.nrTasks
    sizeDomains = auxData.sizeDomains
    sizeDomains22 = sizeDomains[1:nrTasks]
    sizeDomains11 = sizeDomains[nrTasks+1:2*nrTasks]

    buffTHat = auxData.buffTHatPerm
    buffM = auxData.buffMPerm

    plusOneCmplx = convert(FieldType, 1.0)
    minusOneCmplx = convert(FieldType, -1.0)
    zeroCmplx = convert(FieldType, 0.0)

    iOffset = sum(sizeDomains22)

    Threads.@threads for ixo = 1:auxData.nrThreads
        buffBlockSizeD1 = blockSizeD1

        if ixo < auxData.nrThreads
            nrTasksPerThread = auxData.maxNrTasksPerThread
        else
            nrTasksPerThread = auxData.lastNrTasksPerThread
        end

        iAccum = sum(sizeDomains11[1:(ixo-1)*auxData.maxNrTasksPerThread])
        jAccum = sum(sizeDomains22[1:(ixo-1)*auxData.maxNrTasksPerThread])
        for ixi = 1:nrTasksPerThread
            # index of each individual task
            ix_ = (ixo - 1) * auxData.maxNrTasksPerThread + ixi

            if (auxData.lastSizeD1 != 0) && (ix_ == auxData.nrTasks)
                buffBlockSizeD1 = auxData.lastSizeD1
            end

            if ixi > 1
                iAccum += sizeDomains11[ix_-1]
            end
            ixLpermOffset = iOffset + iAccum

            jAccum += sizeDomains22[ix_]
            jxLperm = jAccum
            for ix = 1:buffBlockSizeD1
                ixLperm = ixLpermOffset + ix

                # FIRST : left term : [ixLperm, ixLperm-1]
                if ix > 1
                    be_copy_in_hw!(Mout.M[ixLperm, ixLperm-1], buffTHat.M[ixLperm, ixLperm-1])
                    # Mout.M[ixLperm, ixLperm-1] =  buffM.M[ixLperm, jxLperm] * buffTHat.M[jxLperm, ixLperm-1]
                    # Mout.M[ixLperm, ixLperm-1] += buffM.M[ixLperm, jxLperm+1] * buffTHat.M[jxLperm+1, ixLperm-1]
                    be_gemm!('N', 'N', minusOneCmplx, buffM.M[ixLperm, jxLperm], buffTHat.M[jxLperm, ixLperm-1],
                        plusOneCmplx, Mout.M[ixLperm, ixLperm-1], td, cd)
                    if ix_ < nrTasks
                        be_gemm!('N', 'N', minusOneCmplx, buffM.M[ixLperm, jxLperm+1], buffTHat.M[jxLperm+1, ixLperm-1],
                            plusOneCmplx, Mout.M[ixLperm, ixLperm-1], td, cd)
                    end
                    # IMPORTANT : using the following line would lead to an incorrect result as it would
                    #             change the reference the block element is pointing to
                    # Mout.M[ixLperm, ixLperm-1] = Mout.M[ixLperm, ixLperm-1] + buffTHat.M[ixLperm, ixLperm-1]
                end

                # SECOND : central term : [ixLperm, ixLperm]
                be_copy_in_hw!(Mout.M[ixLperm, ixLperm], buffTHat.M[ixLperm, ixLperm])
                # Mout.M[ixLperm, ixLperm] =  buffM.M[ixLperm, jxLperm] * buffTHat.M[jxLperm, ixLperm]
                # Mout.M[ixLperm, ixLperm] += buffM.M[ixLperm, jxLperm+1] * buffTHat.M[jxLperm+1, ixLperm]
                be_gemm!('N', 'N', minusOneCmplx, buffM.M[ixLperm, jxLperm], buffTHat.M[jxLperm, ixLperm],
                    plusOneCmplx, Mout.M[ixLperm, ixLperm], td, cd)
                if ix_ < nrTasks
                    be_gemm!('N', 'N', minusOneCmplx, buffM.M[ixLperm, jxLperm+1], buffTHat.M[jxLperm+1, ixLperm],
                        plusOneCmplx, Mout.M[ixLperm, ixLperm], td, cd)
                end

                # THIRD : right term : [ixLperm, ixLperm+1]
                if ix < buffBlockSizeD1
                    be_copy_in_hw!(Mout.M[ixLperm, ixLperm+1], buffTHat.M[ixLperm, ixLperm+1])
                    # Mout.M[ixLperm, ixLperm+1] =  buffM.M[ixLperm, jxLperm] * buffTHat.M[jxLperm, ixLperm+1]
                    # Mout.M[ixLperm, ixLperm+1] += buffM.M[ixLperm, jxLperm+1] * buffTHat.M[jxLperm+1, ixLperm+1]
                    be_gemm!('N', 'N', minusOneCmplx, buffM.M[ixLperm, jxLperm], buffTHat.M[jxLperm, ixLperm+1],
                        plusOneCmplx, Mout.M[ixLperm, ixLperm+1], td, cd)
                    if ix_ < nrTasks
                        be_gemm!('N', 'N', minusOneCmplx, buffM.M[ixLperm, jxLperm+1], buffTHat.M[jxLperm+1, ixLperm+1],
                            plusOneCmplx, Mout.M[ixLperm, ixLperm+1], td, cd)
                    end
                end
            end
        end
    end
end

function bndiag_of_inv_ddrgf_inv_of_T11!(Min_::BlockMatrix, auxData::AuxDataDDRGF, td::TimingData,
    cd::CountingData)

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

        jxStartAccum = sum(auxData.sizeDomains[1:auxData.nrTasks+(ixo-1)*auxData.maxNrTasksPerThread])
        jxEndAccum = jxStartAccum
        for ixi = 1:nrTasksPerThread
            # index of each individual task
            ix = auxData.nrTasks + (ixo - 1) * auxData.maxNrTasksPerThread + ixi

            if ixi > 1
                jxStartAccum += auxData.sizeDomains[ix-1]
            end
            jxStart = jxStartAccum + 1
            jxEndAccum += auxData.sizeDomains[ix]
            jxEnd = jxEndAccum

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
            bndiag_of_inv_rgf_local!(smallMbmOut, smallMbmIn, smallAuxDataSeq, td, cd)

        end
    end
end

function bndiag_of_inv_ddrgf_build_Schur_compl!(Min_::BlockMatrix, auxData::AuxDataDDRGF, td::TimingData,
    cd::CountingData)

    minusOneCmplx = convert(FieldType, -1.0)
    plusOneCmplx = convert(FieldType, 1.0)
    zeroCmplx = convert(FieldType, 0.0)

    # the blocks in the following matrices contain references to blocks
    buffTHat = auxData.buffTHatPerm
    Min = Min_

    Threads.@threads for ixo = 1:auxData.nrThreads
        if ixo < auxData.nrThreads
            nrTasksPerThread = auxData.maxNrTasksPerThread
        else
            nrTasksPerThread = auxData.lastNrTasksPerThread
        end

        jx2StartAccum = sum(auxData.sizeDomains[1:(ixo-1)*auxData.maxNrTasksPerThread])
        jx2EndAccum = jx2StartAccum
        for ixi = 1:nrTasksPerThread
            # index of each individual task
            ix = (ixo - 1) * auxData.maxNrTasksPerThread + ixi

            if ixi > 1
                jx2StartAccum += auxData.sizeDomains[ix-1]
            end
            jx2Start = jx2StartAccum + 1
            jx2EndAccum += auxData.sizeDomains[ix]
            jx2End = jx2EndAccum

            # copy the D2 part of Min into buffTHat

            if ix < auxData.nrTasks
                smallBlockSizes = auxData.smallBlockSizes22[ixo]
                smallMbmIn = auxData.smallMbmIn22[ixo]
                smallMbmBuffTHat = auxData.smallMbmBuffTHat22[ixo]
            else
                smallBlockSizes = Min.blockSizes[jx2Start:jx2End]
                smallMbmIn = BlockMatrix(copy(smallBlockSizes), ArrayOrLU_(undef, jx2End - jx2Start + 1, jx2End - jx2Start + 1),
                    Min.ndiag, 0, false)
                smallMbmBuffTHat = BlockMatrix(copy(smallBlockSizes), ArrayOrLU_(undef, jx2End - jx2Start + 1, jx2End - jx2Start + 1),
                    buffTHat.ndiag, 0, false)
            end

            # TODO : move setting these references to a 'setup' stage (then wrap with an
            #        if statement here for the case ix = auxData.nrTasks)
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
end

function bndiag_of_inv_ddrgf_inv_Schur_compl!(Mout_::BlockMatrix, listOfAuxData::Vector{AuxDataDDRGF}, td::TimingData,
    cd::CountingData, depth::Int)
    # the blocks in the following matrices contain references to blocks
    buffM2 = Mout_

    auxData = listOfAuxData[depth]
    nrLevels = size(listOfAuxData)[1]

    buffTHat22 = auxData.buffTHat22inv
    # auxData.buffM222inv is just a 'shell' to reference to another block matrix
    buffM222 = auxData.buffM222inv

    # this should not be moved to a 'setup' stage, as Mout might change
    # from inversion to inversion (is this true, though, now that we are
    # using auxData.buffMout?)
    bm_reference!(buffM222, buffM2.M, 0, 0)

    if depth == nrLevels
        auxDataSeq22 = auxData.auxDataSeq22inv
        @timewrap td "_SeqInv" bndiag_of_inv_rgf_global!(buffM222, buffTHat22, auxDataSeq22, td, cd)
    else
        bndiag_of_inv_ddrgf!(buffTHat22, listOfAuxData, td, cd, depth + 1)
        bm_copy!(buffM222, listOfAuxData[depth+1].buffMout)
    end
end

function bndiag_of_inv_ddrgf_inv_of_Schur_compl!(Mout_::BlockMatrix, Min_::BlockMatrix, listOfAuxData::Vector{AuxDataDDRGF}, td::TimingData,
    cd::CountingData, depth::Int)

    auxData = listOfAuxData[depth]

    # the blocks in the following matrices contain references to blocks
    Min = Min_
    Mout = Mout_

    # compute the nonzero blocks in THat_{11}^{-1} * THat_{12}, saving the output to the 12 part of buffTHat
    bndiag_of_inv_ddrgf_compute_THat11Inv_x_THat12!(Min, auxData, td, cd)

    # and then those of THat_{21} * THat_{11}^{-1}, saving the output to the 21 part of buffTHat
    bndiag_of_inv_ddrgf_compute_THat21_x_THat11Inv!(Min, auxData, td, cd)

    # the D2 part of buffTHat contains the (approximated) Schur complement
    bndiag_of_inv_ddrgf_build_Schur_compl!(Min, auxData, td, cd)

    # call sequential RGF to compute the inverse of the Schur complement
    bndiag_of_inv_ddrgf_inv_Schur_compl!(Mout, listOfAuxData, td, cd, depth)
end

"""
    bndiag_of_inv_ddrgf!(Mout_::BlockMatrix, Min_::BlockMatrix, auxData::AuxDataDDRGF, td::TimingData,
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
function bndiag_of_inv_ddrgf!(Min_::BlockMatrix, listOfAuxData::Vector{AuxDataDDRGF}, td::TimingData,
    cd::CountingData, depth::Int)
    # TODO : extend this code to n-diagonal, otherwise rename this function
    #        to keep it as the simple traditional RGF

    auxData = listOfAuxData[depth]

    # call sequential RGF if nrTasks = 1
    if auxData.nrTasks == 1
        bndiag_of_inv_rgf_global!(Mout_, Min_, auxData.auxDataSeq, td, cd)
        return
    end

    Min = bndiag_of_inv_ddrgf_create_permuted_matrix(Min_, auxData.permVec)
    # Mout = bndiag_of_inv_ddrgf_create_permuted_matrix(Mout_, auxData.permVec)
    Mout = bndiag_of_inv_ddrgf_create_permuted_matrix(auxData.buffMout, auxData.permVec)
    bndiag_of_inv_ddrgf_add_block_refs_to_permuted_matrix22!(Mout, auxData.buffMout, auxData)

    # first, compute the inverse of \widehat{T}_{11}, storing it in the D1 part of auxData.buffTHat
    # (this is used in both the (1,1) and (2,2) parts)
    @timewrap td "_T11inv" bndiag_of_inv_ddrgf_inv_of_T11!(Min, auxData, td, cd)

    # with the inverse of \widehat{T}_{11} at hand, construct the Schur complement, and
    # then invert it, stored in the D2 part of Mout_
    # (this is the (2,2) part)
    @timewrap td "_SCinv" bndiag_of_inv_ddrgf_inv_of_Schur_compl!(Mout, Min, listOfAuxData, td, cd, depth)

    # compute the (1,2) and (2,1) parts of the global result

    # first, -1 * THat11Inv * THat12 * THatSInv
    # (for this we have auxData.buffMPerm, where we put the whole needed object
    # for THat11Inv * THat12 * THatSInv * THat21 * THat11Inv but also immediately
    # copy what should go to the output Mout)
    @timewrap td "_Hopp12" bndiag_of_inv_ddrgf_compute_minus_THat11Inv_x_THat12_x_THatSInv!(Mout, auxData, td, cd)
    # then, -1 * THatSInv * THat11Inv * THat21
    @timewrap td "_Hopp21" bndiag_of_inv_ddrgf_compute_minus_x_THatSInv_THat21_x_THat11Inv!(Mout, auxData, td, cd)

    # compute the (1,1) part of the global result
    @timewrap td "_11" bndiag_of_inv_ddrgf_compute_11_part!(Mout, auxData, td, cd)
end