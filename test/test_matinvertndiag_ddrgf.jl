# computing the block n-diagonal of the inverse of T
# by the DD-RGF method

include("common_to_test.jl")

# this factor relaxes the required relative tolerance
accFctr = 1.0E4

# 1 from disk, 2 is random
whereFrom = 2

if whereFrom == 1
    for systemx in systemNames
        for E in [Epoints[1]]
            for k in [kpoints[1]]
                for precx in precs
                    # load matrices and build M
                    listMatsToLoad = ["H", "S", "Sc"]
                    loadedMats, blockSizes = load_matrices(systemx, E, k,
                        listMatsToLoad, precx, whereFrom)
                    H = loadedMats[1]
                    S = loadedMats[2]
                    Se = loadedMats[3]
                    M = build_M_from_HS(H, S, Se, energVals[E])

                    # load Gr
                    listMatsToLoad = ["Gr"]
                    loadedMats, blockSizes = load_matrices(systemx, E, k,
                        listMatsToLoad, precx, whereFrom)
                    Gr = loadedMats[1]

                    # loading blockSizes only - this is redundant, but illustrates
                    # that this can be done without any matrix loading
                    listMatsToLoad = Vector{String}()
                    loadedMats, blockSizes = load_matrices(systemx, E, k,
                        listMatsToLoad, precx, whereFrom)

                    # convert to BlockMatrix
                    Mbm = bm_convert(M, blockSizes, Dict("in" => 3, "out" => 3))

                    # pre-allocate buffer data for DD-RGF
                    auxData = allocate_aux_data_DDRGF(Mbm, parse(Int, ARGS[3]), parse(Int, ARGS[4]))
                    # pre-allocate the output matrix
                    MbmInvNdiag = bm_similar(Mbm, 1)
                    # get the block n-diagonal of M^-1 via RGF
                    bndiag_of_inv_ddrgf_global!(MbmInvNdiag, Mbm, auxData, TimingData(), CountingData())

                    # convert back to sparse
                    MinvSp = bm_convert(MbmInvNdiag)

                    relErr = LinearAlgebra.norm(Array(MinvSp - Gr), 2) / LinearAlgebra.norm(Array(Gr), 2)
                    # making a rough assumption on backward stability. The additional
                    # 1.0E1 is because we see a loss in 1 digit in some cases
                    @test relErr < roundoffs[precx] * accFctr
                end
            end
        end
    end
end