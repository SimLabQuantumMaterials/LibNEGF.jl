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
                npl = 135
                blockSize = 128
                MbmSynth = bm_create_synthetic(MbmFromData, npl, blockSize)
                # IMPORTANT : the recommended value for nrBlocksInNonPivots is four or less
                nrBlocksInNonPivots = 4

                # -----------------------------

                # first, some minor checks, mostly related to permutations

                begin

                    # reference to the block matrix coming from data
                    MbmSeq = MbmSynth
                    # pre-allocate buffer data for sequential RGF
                    auxDataSeq = allocate_aux_data_DDRGF(MbmSeq)
                    # pre-allocate buffer data for parallel RGF
                    # TODO : move the following param inside the check_nr_tasks function,
                    #        and with this decide based on the criteria explained in the paper
                    #        (throw an error in the code if the last else is not being caught)
                    # 0 is open-end, 1 is closed-end
                    splitType::Bool = 0
                    auxDataPar = allocate_aux_data_PDDRGF(MbmSeq, nrBlocksInNonPivots, splitType, auxDataSeq)

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
                    auxDataSeq = allocate_aux_data_DDRGF(MbmSeq)

                    # call sequential RGF
                    bndiag_of_inv_ddrgf!(MbmInvNdiagSeq, MbmSeq, auxDataSeq, TimingData(), CountingData())

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
                    auxDataPar = allocate_aux_data_PDDRGF(MbmPar, nrBlocksInNonPivots, splitType, auxDataSeq)

                    # # get the block n-diagonal of M^-1 via RGF
                    # println("Measurements for running parallel RGF")
                    # @time bndiag_of_inv_pddrgf!(MbmInvNdiagPar, MbmPar, auxDataPar, TimingData(), CountingData())

                    # compute \widehat{T}_{11} (saved @ the D1 part of auxDataPar.buffTHat) and check its correctness
                    bndiag_of_inv_pddrgf_inv_of_T11!(MbmPar, auxDataPar, TimingData(), CountingData())
                    relErr::Float64 = bndiag_of_inv_pddrgf_error_inv_of_T11(MbmPar, MbmInvNdiagPar, auxDataPar, TimingData(), CountingData())
                    @test relErr < roundoffs[precx] * 1.0E5

                    # ----------

                    # # check that the Schur complement construction is correct

                    # MbmPar_reord = bndiag_of_inv_pddrgf_create_permuted_matrix(MbmPar, auxDataPar.permVec)
                    # println(auxDataPar.sizeDomains)
                    # nb2 = sum(auxDataPar.sizeDomains[1:auxDataPar.nrTasks])
                    # nb1 = sum(auxDataPar.sizeDomains[auxDataPar.nrTasks+1:2*auxDataPar.nrTasks])
                    # nx = sum(MbmPar_reord.blockSizes[1:nb2])
                    # ny = sum(MbmPar_reord.blockSizes[nb2+1:nb2+nb1])
                    # PermMat = bndiag_of_inv_pddrgf_create_sparse_permutator(auxDataPar.permVec, MbmSeq.blockSizes, MbmSeq.nrsType)

                    # MPar = bm_convert(MbmPar)
                    # MPar_perm = PermMat * (MPar * PermMat')
                    # MPar_perm11 = MPar_perm[nx+1:nx+ny,nx+1:nx+ny]
                    # MPar_perm11Inv = SparseArrays.SparseMatrixCSC(LinearAlgebra.inv(Array(MPar_perm11)))

                    # # compute the inverse of the Schur complement
                    # bndiag_of_inv_pddrgf_inv_of_Schur_compl!(MbmInvNdiagPar, MbmPar, auxDataPar, TimingData(), CountingData())

                    # buffTHat = bm_convert(auxDataPar.buffTHat)
                    # buffTHat_perm = PermMat * (buffTHat * PermMat')
                    # buffTHat_perm11 = buffTHat_perm[nx+1:nx+ny,nx+1:nx+ny]

                    # # ll1 = 129:256
                    # # ll2 = 1:128
                    # # relErr = LinearAlgebra.norm(Array(MPar_perm11Inv[ll1,ll2] - buffTHat_perm11[1:dsx,1:dsx][ll1,ll2]), 2) / LinearAlgebra.norm(Array(MPar_perm11Inv[ll1,ll2]), 2)
                    # # println(relErr)

                    # # check first that the Schur complement was built properly

                    # MPar_perm22 = MPar_perm[1:nx,1:nx]
                    # MPar_perm12 = MPar_perm[nx+1:nx+ny,1:nx]
                    # MPar_perm21 = MPar_perm[1:nx,nx+1:nx+ny]

                    # buffTHat_perm22 = buffTHat_perm[1:nx,1:nx]

                    # exactSC = MPar_perm22 - MPar_perm21 * (MPar_perm11Inv * MPar_perm12)
                    # approSC = buffTHat_perm22

                    # # MbmSynthSp = bm_convert(MbmSynth)
                    # # tx = size(MbmSynthSp)[1]
                    # # display(xlims!(ylims!(spy!(sparse(abs.(MbmSynthSp))), (1,tx)), (1,tx)))

                    # # # MbmFromDataSp = bm_convert(MbmFromData)
                    # # # tx = size(MbmFromDataSp)[1]
                    # # # display(xlims!(ylims!(spy!(sparse(abs.(MbmFromDataSp))), (1,tx)), (1,tx)))

                    # # sleep(30)

                    # # # # xxi = 1
                    # # # # xx1 = sum(auxDataPar.sizeDomains[xxi-1:xxi-1])+1
                    # # # # xx2 = sum(auxDataPar.sizeDomains[xxi:xxi])
                    # # # # dsx = sum(MbmPar_reord.blockSizes[xx1:xx2])
                    # # dsx = sum(MbmPar_reord.blockSizes[1:sum(auxDataPar.sizeDomains[1:1])])
                    # # println(dsx)
                    # # r1 = 1:dsx
                    # # r2 = r1
                    # # relErr = LinearAlgebra.norm(Array(approSC[r1,r2] - exactSC[r1,r2]), 2) / LinearAlgebra.norm(Array(exactSC[r1,r2]), 2)
                    # # println(relErr)

                    # # # intrvl = 1:896
                    # # # intrvl = 897:2*896
                    # # intrvl = 1:2*896
                    # Ex = approSC - exactSC
                    # relErr = LinearAlgebra.norm(Array(Ex), 2) / LinearAlgebra.norm(Array(exactSC), 2)
                    # println("relErr = "*string(relErr))
                    # # relErr = LinearAlgebra.norm(Array(Ex[intrvl,intrvl]), 2) / LinearAlgebra.norm(Array(exactSC[intrvl,intrvl]), 2)
                    # # println("relErr = "*string(relErr))

                    # ET0 = approSC * ( Ex * approSC )
                    # ET0 = ET0 * ( Ex * approSC )
                    # # et0Norm = LinearAlgebra.opnorm(Array(ET0), 2)
                    # # println("et0Norm = "*string(et0Norm))
                    # # evalsx = LinearAlgebra.eigvals(Array(ET0))
                    # # evalsx = abs.(evalsx)
                    # # println(sort(evalsx))

                    # # IMPORTANT : to use the following spy lines, ones needs to do Pkg.add("Plots")
                    # # using Plots
                    # # sx = size(Ex)[1]
                    # # println(size(Ex))
                    # # display(xlims!(ylims!(spy!(sparse(abs.(ET0))), (1,sx)), (1,sx)))
                    # # # display(xlims!(ylims!(spy!(sparse(abs.(exactSC))), (1,sx)), (1,sx)))
                    # # sleep(60)

                    # # check that the Schur complement has been built correctly, at loadedMats
                    # # at the D2-level sub-matrices
                    # for ix=1:auxDataPar.nrTasks
                    #     d1 = sum(auxDataPar.sizeDomains[1:ix-1]) + 1
                    #     d2 = sum(auxDataPar.sizeDomains[1:ix])
                    #     r1 = sum(MbmPar_reord.blockSizes[1:d1-1]) + 1
                    #     r2 = sum(MbmPar_reord.blockSizes[1:d2])
                    #     relErr = LinearAlgebra.norm(Array(approSC[r1:r2,r1:r2] - exactSC[r1:r2,r1:r2]), 2) / LinearAlgebra.norm(Array(exactSC[r1:r2,r1:r2]), 2)
                    #     println(relErr)
                    # end

                    # # # check the correctness of the inverse of the Schur complement

                    # MinvNdiagSeq = bm_convert(MbmInvNdiagSeq)
                    # MinvNdiagPar = bm_convert(MbmInvNdiagPar)

                    # MinvNdiagSeq_perm = PermMat * (MinvNdiagSeq * PermMat')
                    # MinvNdiagPar_perm = PermMat * (MinvNdiagPar * PermMat')

                    # # NOTE : 
                    # # MinvNdiagPar_perm = MinvNdiagPar_perm[1:nx,1:nx]
                    # Mx_perm = copy(MinvNdiagSeq_perm)
                    # Mx_perm[1:nx,1:nx] = sparse(LinearAlgebra.inv(Array(approSC)))
                    # Mx = PermMat' * (Mx_perm * PermMat)
                    # Mbmx = bm_convert(Mx, MbmSynth.blockSizes, Dict("in" => 3, "out" => 3))
                    # Mx = bm_convert(Mbmx)
                    # Mx_perm = PermMat * (Mx * PermMat')
                    # MinvNdiagPar_perm = Mx_perm[1:nx,1:nx]

                    # MinvNdiagSeq_perm = MinvNdiagSeq_perm[1:nx,1:nx]

                    # for ix=1:auxDataPar.nrTasks
                    #     d1 = sum(auxDataPar.sizeDomains[1:ix-1]) + 1
                    #     d2 = sum(auxDataPar.sizeDomains[1:ix])
                    #     r1 = sum(MbmPar_reord.blockSizes[1:d1-1]) + 1
                    #     r2 = sum(MbmPar_reord.blockSizes[1:d2])
                    #     println(r1)
                    #     println(r2)
                    #     relErr = LinearAlgebra.norm(Array(MinvNdiagSeq_perm[r1:r2,r1:r2] - MinvNdiagPar_perm[r1:r2,r1:r2]), 2) / LinearAlgebra.norm(Array(MinvNdiagSeq_perm[r1:r2,r1:r2]), 2)
                    #     println(relErr)
                    # end

                    # # MinvNdiagSeq_perm_sl = MinvNdiagSeq_perm[1:nx,1:nx]
                    # # MinvNdiagPar_perm_sl = MinvNdiagPar_perm[1:nx,1:nx]
                    # # # E = approSC - exactSC

                    # # println(nx+ny)
                    # # println(nx)
                    # # println(size(MinvNdiagPar))

                    # # # MinvNdiagPar_perm_sl = MinvNdiagPar_perm_sl + MinvNdiagPar_perm_sl*(E*MinvNdiagPar_perm_sl)
                    # # # relErr = LinearAlgebra.norm(Array(MinvNdiagSeq_perm_sl[1:128] - MinvNdiagPar_perm_sl[1:128]), 2) / LinearAlgebra.norm(Array(MinvNdiagSeq_perm_sl[1:128]), 2)

                    # # relErr = LinearAlgebra.norm(Array(MinvNdiagSeq_perm_sl[1:128,1:128] - MinvNdiagPar_perm_sl[1:128,1:128]), 2) / LinearAlgebra.norm(Array(MinvNdiagSeq_perm_sl[1:128,1:128]), 2)
                    # # println(relErr)
                    # # # println(nx)
                    # # # println(size(MinvNdiagSeq_perm_sl))
                    # # # # MinvNdiagSeq_D2 = 

                    # # # convert back to sparse
                    # # MinvSp = bm_convert(MbmInvNdiag)

                    # # relErr = LinearAlgebra.norm(Array(MinvSp - Gr), 2) / LinearAlgebra.norm(Array(Gr), 2)
                    # # # making a rough assumption on backward stability. The additional
                    # # # 1.0E1 is because we see a loss in 1 digit in some cases
                    # # @test relErr < roundoffs[precx] * 1.0E4

                    # # MbmFromData = 0
                    # # Mbm = 0
                    # # auxData = 0
                    # # MbmInvNdiag = 0
                    # # GC.gc()

                end
            end
        end
    end
end