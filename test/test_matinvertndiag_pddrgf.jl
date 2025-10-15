# computing the block n-diagonal of the inverse of T
# by the DD-RGF method

include("common_to_test.jl")

# TODO : remove after some temporary dirty tests in here
# import BenchmarkTools

using SparseArrays

for systemx in systemNames
    for E in [Epoints[1]]
        for k in [kpoints[1]]
            # pre-compute the condition number in double precision
            # list of matrices to load
            listMatsToLoad = ["H", "S", "Sc"]
            loadedMats, blockSizes = load_matrices(systemx, E, k,
                listMatsToLoad, ComplexF64)
            H = loadedMats[1]
            S = loadedMats[2]
            Se = loadedMats[3]
            M = build_M_from_HS(H, S, Se, energVals[E])

            for precx in [precs[2]]
                # load matrices and build M
                listMatsToLoad = ["H", "S", "Sc"]
                loadedMats, blockSizes = load_matrices(systemx, E, k,
                    listMatsToLoad, precx)
                H = loadedMats[1]
                S = loadedMats[2]
                Se = loadedMats[3]
                M = build_M_from_HS(H, S, Se, energVals[E])

                # load Gr
                listMatsToLoad = ["Gr"]
                loadedMats, blockSizes = load_matrices(systemx, E, k,
                    listMatsToLoad, precx)
                Gr = loadedMats[1]

                # loading blockSizes only - this is redundant, but illustrates
                # that this can be done without any matrix loading
                listMatsToLoad = Vector{String}()
                loadedMats, blockSizes = load_matrices(systemx, E, k,
                    listMatsToLoad, precx)

                # convert to BlockMatrix
                MbmFromData = bm_convert(M, blockSizes, Dict("in" => 3, "out" => 3))

                # crate synthetic matrix with more principal layers and smaller block size
                npl = 160
                blockSize = 128
                MbmSynth = bm_create_synthetic(MbmFromData, npl, blockSize)
                # IMPORTANT : the recommended value for nrBlocksInNonPivots is 2, to reduce fill up
                #             as much as possible
                nrBlocksInNonPivots = 2

                # -----------------------------

                # first, some minor checks, mostly related to permutations

                begin

                    # reference to the block matrix coming from data
                    MbmSeq = MbmSynth
                    # pre-allocate buffer data for sequential RGF
                    auxDataSeq = allocate_aux_data_DDRGF(MbmSeq, parse(Int, ARGS[3]), parse(Int, ARGS[4]))
                    # pre-allocate buffer data for parallel RGF
                    # TODO : move the following param inside the check_nr_tasks function,
                    #        and with this decide based on the criteria explained in the paper
                    #        (throw an error in the code if the last else is not being caught)
                    # 0 is open-end, 1 is closed-end
                    splitType::Bool = 0
                    auxDataPar = allocate_aux_data_PDDRGF(MbmSeq, nrBlocksInNonPivots, splitType, auxDataSeq,
                        parse(Int, ARGS[2]), parse(Int, ARGS[3]), parse(Int, ARGS[4]))

                    # the blocks in the following matrices are references to the blocks in Min
                    MbmSeqPerm = bndiag_of_inv_pddrgf_create_permuted_matrix(MbmSeq, auxDataPar.permVec)

                    # covert MbmSeq to sparse
                    MbmSeqSp = bm_convert(MbmSeq)
                    # permute that sparse matrix
                    PermMat = bndiag_of_inv_pddrgf_create_sparse_permutator(auxDataPar.permVec, MbmSeq.blockSizes, MbmSeq.nrsType)
                    MbmSeqSpPerm = PermMat * (MbmSeqSp * PermMat')
                    # convert MbmSeqPerm to sparse
                    MbmSeqPermSp = bm_convert(MbmSeqPerm, auxDataPar.permVec)
                    # compare both
                    relErr = LinearAlgebra.norm(Array(MbmSeqSpPerm - MbmSeqPermSp), 2) / LinearAlgebra.norm(Array(MbmSeqSpPerm), 2)
                    @test relErr < roundoffs[precx]

                    # freeing some data
                    auxDataSeq = 0
                    auxDataPar = 0
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
                    # pre-allocate buffer data for sequential RGF
                    auxDataSeq = allocate_aux_data_DDRGF(MbmSeq, parse(Int, ARGS[3]), parse(Int, ARGS[4]))

                    # call sequential RGF
                    @time bndiag_of_inv_ddrgf_global!(MbmInvNdiagSeq, MbmSeq, auxDataSeq, TimingData(), CountingData())
                    @time bndiag_of_inv_ddrgf_global!(MbmInvNdiagSeq, MbmSeq, auxDataSeq, TimingData(), CountingData())
                    @time bndiag_of_inv_ddrgf_global!(MbmInvNdiagSeq, MbmSeq, auxDataSeq, TimingData(), CountingData())

                    GC.gc()

                    # SECOND, parallel (use the data already allocated for the sequential case)

                    # reference to the block matrix coming from data
                    MbmPar = MbmSynth
                    # pre-allocate the output matrix
                    MbmInvNdiagPar = bm_similar(MbmPar, 1)
                    # pre-allocate buffer data for parallel RGF
                    # TODO : move the following param inside the check_nr_tasks function,
                    #        and with this decide based on the criteria explained in the paper
                    #        (throw an error in the code if the last else is not being caught)
                    # 0 is open-end, 1 is closed-end
                    splitType = 0
                    auxDataPar = allocate_aux_data_PDDRGF(MbmPar, nrBlocksInNonPivots, splitType, auxDataSeq,
                        parse(Int, ARGS[2]), parse(Int, ARGS[3]), parse(Int, ARGS[4]))
                    bm_blocks_define_complement22!(MbmInvNdiagPar, auxDataPar, 2)

                    # # get the block n-diagonal of M^-1 via RGF
                    # println("Measurements for running parallel RGF")
                    # @time bndiag_of_inv_pddrgf!(MbmInvNdiagPar, MbmPar, auxDataPar, TimingData(), CountingData())

                    MbmParPerm = bndiag_of_inv_pddrgf_create_permuted_matrix(MbmPar, auxDataPar.permVec)
                    MbmInvNdiagParPerm = bndiag_of_inv_pddrgf_create_permuted_matrix(MbmInvNdiagPar, auxDataPar.permVec)
                    bndiag_of_inv_pddrgf_add_block_refs_to_permuted_matrix22!(MbmInvNdiagParPerm, MbmInvNdiagPar, auxDataPar)

                    # compute \widehat{T}_{11} (saved @ the D1 part of auxDataPar.buffTHat) and check its correctness
                    @time bndiag_of_inv_pddrgf_inv_of_T11!(MbmParPerm, auxDataPar, TimingData(), CountingData())
                    @time bndiag_of_inv_pddrgf_inv_of_T11!(MbmParPerm, auxDataPar, TimingData(), CountingData())
                    @time bndiag_of_inv_pddrgf_inv_of_T11!(MbmParPerm, auxDataPar, TimingData(), CountingData())
                    relErr::Float64 = bndiag_of_inv_pddrgf_error_inv_of_T11(MbmPar, MbmInvNdiagPar, auxDataPar, TimingData(), CountingData())
                    @test relErr < roundoffs[precx] * 1.0E6

                    # check that the Schur complement construction is correct

                    # compute the inverse of the Schur complement. The Schur complement is stored
                    # in the D2 part of auxData.buffTHat, and its inverse in the D2 part of MbmInvNdiagPar
                    println("")
                    @time bndiag_of_inv_pddrgf_inv_of_Schur_compl!(MbmInvNdiagParPerm, MbmParPerm, auxDataPar, TimingData(), CountingData())
                    println("")
                    @time bndiag_of_inv_pddrgf_inv_of_Schur_compl!(MbmInvNdiagParPerm, MbmParPerm, auxDataPar, TimingData(), CountingData())
                    println("")
                    @time bndiag_of_inv_pddrgf_inv_of_Schur_compl!(MbmInvNdiagParPerm, MbmParPerm, auxDataPar, TimingData(), CountingData())
                    println("")

                    MbmPar_reord = bndiag_of_inv_pddrgf_create_permuted_matrix(MbmPar, auxDataPar.permVec)
                    nb2 = sum(auxDataPar.sizeDomains[1:auxDataPar.nrTasks])
                    nb1 = sum(auxDataPar.sizeDomains[auxDataPar.nrTasks+1:2*auxDataPar.nrTasks])
                    nx = sum(MbmPar_reord.blockSizes[1:nb2])
                    ny = sum(MbmPar_reord.blockSizes[nb2+1:nb2+nb1])
                    PermMat = bndiag_of_inv_pddrgf_create_sparse_permutator(auxDataPar.permVec, MbmSeq.blockSizes, MbmSeq.nrsType)

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
                        @test relErr < roundoffs[precx] * 1.E6
                    end

                    # with the exact Schur complement at hand, check whether it was constructed correctly within
                    # the function bndiag_of_inv_pddrgf_inv_of_Schur_compl!(...)
                    buffTHat = bndiag_of_inv_pddrgf_create_permuted_matrix(auxDataPar.buffTHat, auxDataPar.permVec)
                    bndiag_of_inv_pddrgf_add_block_refs_to_permuted_matrix22!(buffTHat, auxDataPar.buffTHat, auxDataPar)
                    nrLayersSchurCompl = sum(auxDataPar.sizeDomains[1:auxDataPar.nrTasks])
                    blockSizesSchurCompl = buffTHat.blockSizes[1:nrLayersSchurCompl]
                    buffTHat22 = bm_empty(blockSizesSchurCompl, nrLayersSchurCompl, buffTHat.ndiag["in"],
                        buffTHat.isArrayOrLU, buffTHat.nrsType)
                    buffTHatMView = view(buffTHat.M, 1:nrLayersSchurCompl, 1:nrLayersSchurCompl)
                    bm_reference!(buffTHat22, buffTHatMView)
                    buffTHatM22 = bm_convert(buffTHat22)
                    relErr = LinearAlgebra.norm(Array(buffTHatM22 - exactSC), 2) / LinearAlgebra.norm(Array(exactSC), 2)
                    @test relErr < roundoffs[precx] * 1.E6

                    # check the correctness of the inverse of the Schur complement

                    MinvNdiagSeq = bm_convert(MbmInvNdiagSeq)
                    MinvNdiagPar = bm_convert(MbmInvNdiagPar)

                    MinvNdiagSeq_perm = PermMat * (MinvNdiagSeq * PermMat')
                    MinvNdiagPar_perm = PermMat * (MinvNdiagPar * PermMat')

                    MinvNdiagPar_perm = MinvNdiagPar_perm[1:nx, 1:nx]
                    MinvNdiagSeq_perm = MinvNdiagSeq_perm[1:nx, 1:nx]

                    for ix = 1:auxDataPar.nrTasks
                        d1 = sum(auxDataPar.sizeDomains[1:ix-1]) + 1
                        d2 = sum(auxDataPar.sizeDomains[1:ix])
                        r1 = sum(MbmPar_reord.blockSizes[1:d1-1]) + 1
                        r2 = sum(MbmPar_reord.blockSizes[1:d2])
                        relErr = LinearAlgebra.norm(Array(MinvNdiagSeq_perm[r1:r2, r1:r2] - MinvNdiagPar_perm[r1:r2, r1:r2]), 2) /
                                 LinearAlgebra.norm(Array(MinvNdiagSeq_perm[r1:r2, r1:r2]), 2)
                        @test relErr < roundoffs[precx] * 1.E6
                    end

                end
            end
        end
    end
end