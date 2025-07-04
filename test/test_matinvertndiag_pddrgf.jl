# computing the block n-diagonal of the inverse of T
# by the DD-RGF method

include("common_to_test.jl")

# TODO : remove after some temporary dirty tests in here
# import BenchmarkTools

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

            for precx in [precs[1]]
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
                npl = 136
                blockSize = 128
                MbmSynth = bm_create_synthetic(MbmFromData, npl, blockSize)

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
                nrBlocksInPivots = 8
                splitType::Bool = 1
                auxDataPar = allocate_aux_data_PDDRGF(MbmPar, nrBlocksInPivots, splitType, auxDataSeq)

                # get the block n-diagonal of M^-1 via RGF
                bndiag_of_inv_pddrgf!(MbmInvNdiagPar, MbmPar, auxDataPar, TimingData(), CountingData())

                # # convert back to sparse
                # MinvSp = bm_convert(MbmInvNdiag)

                # relErr = LinearAlgebra.norm(Array(MinvSp - Gr), 2) / LinearAlgebra.norm(Array(Gr), 2)
                # # making a rough assumption on backward stability. The additional
                # # 1.0E1 is because we see a loss in 1 digit in some cases
                # @test relErr < roundoffs[precx] * 1.0E4

                # MbmFromData = 0
                # Mbm = 0
                # auxData = 0
                # MbmInvNdiag = 0
                # GC.gc()
            end
        end
    end
end