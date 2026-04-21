# computing the block n-diagonal of the inverse of T by the DD-RGF method

include("common_to_test.jl")

using SparseArrays, Random

# fix seed to have reproducible tests
Random.seed!(1234)

# 1 from disk, 2 is random
whereFrom = 2
# values for the synthetic matrix
npl = 128
blockSize = 64

# this factor relaxes the required relative tolerance
accFctrBare = 2.5E5
accFctr = 0.0

for k in kpoints
    for precx in [ComplexF64]
        # check_if_enough_mem_ddrgf(npl, blockSize, precx)

        if precx == ComplexF64
            global accFctr = accFctrBare
        else
            # extra relaxation in lower precision
            global accFctr = 3.0 * accFctrBare
        end

        # if whereFrom == 1
        #     # load matrices and build M
        #     listMatsToLoad = ["H", "S", "Sc"]
        #     loadedMats, blockSizes = load_matrices(systemx, E, k,
        #         listMatsToLoad, precx, whereFrom)
        #     H = loadedMats[1]
        #     S = loadedMats[2]
        #     Se = loadedMats[3]
        #     M = build_M_from_HS(H, S, Se, energVals[E])

        #     # we don't really need Gr, we do checks for DDRGF against RGF
        #     # # load Gr
        #     # listMatsToLoad = ["Gr"]
        #     # loadedMats, blockSizes = load_matrices(systemx, E, k,
        #     #     listMatsToLoad, precx, whereFrom)
        #     # Gr = loadedMats[1]

        #     # loading blockSizes only - this is redundant, but illustrates
        #     # that this can be done without any matrix loading
        #     listMatsToLoad = Vector{String}()
        #     loadedMats, blockSizes = load_matrices(systemx, E, k,
        #         listMatsToLoad, precx, whereFrom)

        #     # convert to BlockMatrix
        #     MbmFromData = bm_convert(M, blockSizes, Dict("in" => 3, "out" => 3), false)

        #     # crate synthetic matrix
        #     MbmSynth = bm_create_synthetic(MbmFromData, npl, blockSize)
        # else
        #     MbmSynth = bm_create_synthetic_random(npl, blockSize, precx, false)
        # end

        MbmSynth = nothing
        try
            MbmSynth = bm_create_synthetic_random(npl, blockSize, precx, false)
        catch
            if e isa OutOfMemoryError
                # TODO : handle this better, with perhaps a suggestion in params change
                error("The application tried to allocate beyond the available system memory")
            else
                rethrow(e)
            end
        end

        # -----------------------------

        # first, some minor checks, mostly related to permutations

        # reference to the block matrix coming from data
        MbmSeq = MbmSynth

        begin
            MbmSeqSpPerm = nothing
            MbmSeqPermSp = nothing
            try
                listOfAuxDataPar = allocate_aux_data_DDRGF(MbmSeq, TimingData(), CountingData())
                auxDataPar = listOfAuxDataPar[1]

                # the blocks in the following matrices are references to the blocks in Min
                MbmSeqPerm = bndiag_of_inv_ddrgf_create_permuted_matrix(MbmSeq, auxDataPar.permVec)

                # covert MbmSeq to sparse
                MbmSeqSp = bm_convert(MbmSeq)
                # permute that sparse matrix
                PermMat = bndiag_of_inv_ddrgf_create_sparse_permutator(auxDataPar.permVec, MbmSeq.blockSizes)
                MbmSeqSpPerm = PermMat * (MbmSeqSp * PermMat')
                # convert MbmSeqPerm to sparse
                MbmSeqPermSp = bm_convert(MbmSeqPerm, auxDataPar.permVec)
            catch
                if e isa OutOfMemoryError
                    # TODO : handle this better, with perhaps a suggestion in params change
                    error("The application tried to allocate beyond the available system memory")
                else
                    rethrow(e)
                end
            end

            # compare both
            relErr = LinearAlgebra.norm(Array(MbmSeqSpPerm - MbmSeqPermSp), 2) / LinearAlgebra.norm(Array(MbmSeqSpPerm), 2)
            @test relErr < roundoffs[precx]

            # freeing some data
            auxDataSeq = nothing
            auxDataPar = nothing
            GC.gc()
        end

        # -----------------------------

        # then, the main operations:

        begin
            # FIRST, sequential

            # reference to the block matrix coming from data
            MbmSeq = MbmSynth
            # pre-allocate the output matrix
            MbmInvNdiagSeq = bm_similar(MbmSeq, 1)

            auxDataSeq = nothing
            try
                # pre-allocate buffer data for sequential RGF
                auxDataSeq = allocate_aux_data_RGF(MbmSeq)
            catch
                if e isa OutOfMemoryError
                    # TODO : handle this better, with perhaps a suggestion in params change
                    error("The application tried to allocate beyond the available system memory")
                else
                    rethrow(e)
                end
            end

            # call sequential RGF
            bndiag_of_inv_rgf_global!(MbmInvNdiagSeq, MbmSeq, auxDataSeq, TimingData(), CountingData())

            GC.gc()

            # SECOND, parallel (use the data already allocated for the sequential case)

            # reference to the block matrix coming from data
            MbmPar = MbmSynth
            # pre-allocate the output matrix
            MbmInvNdiagPar = bm_similar(MbmPar, 1)

            listOfAuxDataPar = nothing
            try
                # pre-allocate buffer data for parallel RGF
                listOfAuxDataPar = allocate_aux_data_DDRGF(MbmPar, TimingData(), CountingData())
            catch
                if e isa OutOfMemoryError
                    # TODO : handle this better, with perhaps a suggestion in params change
                    error("The application tried to allocate beyond the available system memory")
                else
                    rethrow(e)
                end
            end

            # relErr::Float64 = bndiag_of_inv_ddrgf_error_inv_of_T11(MbmPar, MbmInvNdiagPar, auxDataPar, TimingData(), CountingData())
            # @test relErr < roundoffs[precx] * 1.0E6

            # call the DDRGF inversion
            begin
                bndiag_of_inv_ddrgf!(MbmPar, listOfAuxDataPar, TimingData(), CountingData(), 1)
                bm_copy!(MbmInvNdiagPar, listOfAuxDataPar[1].buffMout)
            end

            # check that the Schur complement construction is correct

            auxDataPar = listOfAuxDataPar[1]

            MbmPar_reord = bndiag_of_inv_ddrgf_create_permuted_matrix(MbmPar, auxDataPar.permVec)
            nb2 = sum(auxDataPar.sizeDomains[1:auxDataPar.nrTasks])
            nb1 = sum(auxDataPar.sizeDomains[auxDataPar.nrTasks+1:2*auxDataPar.nrTasks])
            nx = sum(MbmPar_reord.blockSizes[1:nb2])
            ny = sum(MbmPar_reord.blockSizes[nb2+1:nb2+nb1])
            PermMat = bndiag_of_inv_ddrgf_create_sparse_permutator(auxDataPar.permVec, MbmSeq.blockSizes)

            MPar = bm_convert(MbmPar)
            MPar_perm = PermMat * (MPar * PermMat')
            MPar_perm11 = MPar_perm[nx+1:nx+ny, nx+1:nx+ny]
            sizeDomains22 = auxDataPar.sizeDomains[1:auxDataPar.nrTasks]
            sizeDomains11 = auxDataPar.sizeDomains[auxDataPar.nrTasks+1:2*auxDataPar.nrTasks]
            offset22 = sum(sizeDomains22)
            blockSizes11 = MbmPar_reord.blockSizes[offset22+1:sum(auxDataPar.sizeDomains)]
            MPar_perm11Inv = copy(MPar_perm11)
            for ix = 1:auxDataPar.nrTasks
                ixDStart = sum(sizeDomains11[1:ix-1]) + 1
                ixDEnd = sum(sizeDomains11[1:ix])
                ixLStart = sum(blockSizes11[1:ixDStart-1]) + 1
                ixLEnd = sum(blockSizes11[1:ixDEnd])
                MPar_perm11Inv[ixLStart:ixLEnd, ixLStart:ixLEnd] =
                    SparseArrays.SparseMatrixCSC(LinearAlgebra.inv(Array(MPar_perm11[ixLStart:ixLEnd, ixLStart:ixLEnd])))
            end

            MPar_perm22 = MPar_perm[1:nx, 1:nx]
            MPar_perm12 = MPar_perm[nx+1:nx+ny, 1:nx]
            MPar_perm21 = MPar_perm[1:nx, nx+1:nx+ny]

            exactSC = MPar_perm22 - MPar_perm21 * (MPar_perm11Inv * MPar_perm12)

            # check that the Schur complement has been built correctly, at the D2-level sub-matrices. This
            # also serves as an indirect check of the inverse of \widehat{T}_{11}

            buffTHat = bm_convert(auxDataPar.buffTHat)
            buffTHat_perm = PermMat * (buffTHat * PermMat')
            buffTHat_perm22 = buffTHat_perm[1:nx, 1:nx]
            approSC = buffTHat_perm22
            for ix = 1:auxDataPar.nrTasks
                d1 = sum(auxDataPar.sizeDomains[1:ix-1]) + 1
                d2 = sum(auxDataPar.sizeDomains[1:ix])
                r1 = sum(MbmPar_reord.blockSizes[1:d1-1]) + 1
                r2 = sum(MbmPar_reord.blockSizes[1:d2])
                relErr = LinearAlgebra.norm(Array(approSC[r1:r2, r1:r2] - exactSC[r1:r2, r1:r2]), 2) /
                         LinearAlgebra.norm(Array(exactSC[r1:r2, r1:r2]), 2)
                @test relErr < roundoffs[precx] * accFctr
            end

            # with the exact Schur complement at hand, check whether it was constructed correctly within
            # the function bndiag_of_inv_ddrgf_inv_of_Schur_compl!(...)

            buffTHat = bndiag_of_inv_ddrgf_create_permuted_matrix(auxDataPar.buffTHat, auxDataPar.permVec)
            bndiag_of_inv_ddrgf_add_block_refs_to_permuted_matrix22!(buffTHat, auxDataPar.buffTHat, auxDataPar)
            nrLayersSchurCompl = sum(auxDataPar.sizeDomains[1:auxDataPar.nrTasks])
            blockSizesSchurCompl = buffTHat.blockSizes[1:nrLayersSchurCompl]
            buffTHat22 = bm_empty(blockSizesSchurCompl, nrLayersSchurCompl, buffTHat.ndiag["in"],
                buffTHat.isArrayOrLU, false)
            buffTHatMView = view(buffTHat.M, 1:nrLayersSchurCompl, 1:nrLayersSchurCompl)
            bm_reference!(buffTHat22, buffTHatMView)
            buffTHatM22 = bm_convert(buffTHat22)
            relErr = LinearAlgebra.norm(Array(buffTHatM22 - exactSC), 2) / LinearAlgebra.norm(Array(exactSC), 2)
            @test relErr < roundoffs[precx] * accFctr

            # check the correctness of the inverse of the Schur complement

            MinvNdiagSeq = bm_convert(MbmInvNdiagSeq)
            MinvNdiagPar = bm_convert(MbmInvNdiagPar)

            MinvNdiagSeq_perm = PermMat * (MinvNdiagSeq * PermMat')
            MinvNdiagPar_perm = PermMat * (MinvNdiagPar * PermMat')

            MinvNdiagPar_perm22 = MinvNdiagPar_perm[1:nx, 1:nx]
            MinvNdiagSeq_perm22 = MinvNdiagSeq_perm[1:nx, 1:nx]

            for ix = 1:auxDataPar.nrTasks
                d1 = sum(auxDataPar.sizeDomains[1:ix-1]) + 1
                d2 = sum(auxDataPar.sizeDomains[1:ix])
                r1 = sum(MbmPar_reord.blockSizes[1:d1-1]) + 1
                r2 = sum(MbmPar_reord.blockSizes[1:d2])
                relErr = LinearAlgebra.norm(Array(MinvNdiagSeq_perm22[r1:r2, r1:r2] - MinvNdiagPar_perm22[r1:r2, r1:r2]), 2) /
                         LinearAlgebra.norm(Array(MinvNdiagSeq_perm22[r1:r2, r1:r2]), 2)
                @test relErr < roundoffs[precx] * accFctr
            end

            # check the correctness of the (1,2) part of the output

            MinvNdiagPar_perm12 = MinvNdiagPar_perm[nx+1:nx+ny, 1:nx]
            MinvNdiagSeq_perm12 = MinvNdiagSeq_perm[nx+1:nx+ny, 1:nx]
            relErr = LinearAlgebra.norm(Array(MinvNdiagSeq_perm12 - MinvNdiagPar_perm12), 2) /
                     LinearAlgebra.norm(Array(MinvNdiagSeq_perm12), 2)
            @test relErr < roundoffs[precx] * accFctr

            # check the correctness of the (2,1) part of the output

            MinvNdiagPar_perm21 = MinvNdiagPar_perm[1:nx, nx+1:nx+ny]
            MinvNdiagSeq_perm21 = MinvNdiagSeq_perm[1:nx, nx+1:nx+ny]
            relErr = LinearAlgebra.norm(Array(MinvNdiagSeq_perm21 - MinvNdiagPar_perm21), 2) /
                     LinearAlgebra.norm(Array(MinvNdiagSeq_perm21), 2)
            @test relErr < roundoffs[precx] * accFctr

            # check the correctness of the (1,1) part of the output

            MinvNdiagPar_perm11 = MinvNdiagPar_perm[nx+1:nx+ny, nx+1:nx+ny]
            MinvNdiagSeq_perm11 = MinvNdiagSeq_perm[nx+1:nx+ny, nx+1:nx+ny]
            for ix = 1:auxDataPar.nrTasks
                ixDStart = sum(sizeDomains11[1:ix-1]) + 1
                ixDEnd = sum(sizeDomains11[1:ix])
                ixLStart = sum(blockSizes11[1:ixDStart-1]) + 1
                ixLEnd = sum(blockSizes11[1:ixDEnd])
                relErr = LinearAlgebra.norm(Array(MinvNdiagSeq_perm11[ixLStart:ixLEnd, ixLStart:ixLEnd] -
                                                  MinvNdiagPar_perm11[ixLStart:ixLEnd, ixLStart:ixLEnd]), 2) /
                         LinearAlgebra.norm(Array(MinvNdiagSeq_perm11[ixLStart:ixLEnd, ixLStart:ixLEnd]), 2)
                @test relErr < roundoffs[precx] * accFctr
            end
        end
    end
end