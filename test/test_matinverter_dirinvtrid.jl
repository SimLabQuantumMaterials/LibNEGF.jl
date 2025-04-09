# computing the block tridiagonal of the inverse of T
# by first computing inv(...)

include("common_to_test_matinverter.jl")

for systemx in systemNames
    for E in Epoints
        for k in kpoints
            # pre-compute the condition number in double precision
            # list of matrices to load
            listMatsToLoad = ["H", "S", "Sc"]
            loadedMats, blockSizes = load_matrices(systemx, E, k,
                listMatsToLoad, ComplexF64)
            H = loadedMats[1]
            S = loadedMats[2]
            Se = loadedMats[3]
            T = build_T_from_HS(H, S, Se, energVals[E])
            condNum = LinearAlgebra.cond(Array(T))

            for precx in precs
                # load matrices and build T
                listMatsToLoad = ["H", "S", "Sc"]
                loadedMats, blockSizes = load_matrices(systemx, E, k,
                    listMatsToLoad, precx)
                H = loadedMats[1]
                S = loadedMats[2]
                Se = loadedMats[3]
                T = build_T_from_HS(H, S, Se, energVals[E])

                # load Gr
                listMatsToLoad = ["Gr"]
                loadedMats, blockSizes = load_matrices(systemx, E, k,
                    listMatsToLoad, precx)
                Gr = loadedMats[1]

                # loading blockSizes only - this is redundant, but illustrates
                # that this can be done without any matrix loading
                listMatsToLoad = []
                loadedMats, blockSizes = load_matrices(systemx, E, k,
                    listMatsToLoad, precx)

                # get the block tridiagonal of T^-1 via inv(T)
                TInvTrid = btrid_of_inv_direct(T, blockSizes)

                relErr = LinearAlgebra.norm(TInvTrid - Gr, 2) / LinearAlgebra.norm(Gr, 2)
                # making a rough assumption on backward stability. The additional
                # 1.0E1 is because we see a loss in 1 digit in some cases
                @test relErr < roundoffs[precx] * condNum * 1.0E1
            end
        end
    end
end