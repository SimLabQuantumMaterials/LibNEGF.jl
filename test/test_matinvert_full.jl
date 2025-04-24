# computing the direct inverse via inv(...), this test
# is added here for completeness, and as a base for the
# subsequent ones

include("common_to_test_matinverter.jl")

for systemx in systemNames
    for E in Epoints
        for k in kpoints

            # pre-compute the condition number of T
            # load matrices and build T
            listMatsToLoad = ["H", "S", "Sc"]
            loadedMats, blockSizes = load_matrices(systemx, E, k,
                listMatsToLoad, ComplexF64)
            H = loadedMats[1]
            S = loadedMats[2]
            Se = loadedMats[3]
            T = build_T_from_HS(H, S, Se, energVals[E])
            # LA.cond(..) makes use of LA.opnorm(..)
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
                Tdense = Array(T)
                TdenseInv = inv(Tdense)
                relErr = LinearAlgebra.opnorm(Tdense * TdenseInv - LinearAlgebra.I, 2) / 1.0
                # making a rough assumption on backward stability. The additional
                # 1.0E1 is because we see a loss in 1 digit in some cases
                @test relErr < roundoffs[precx] * condNum * 1.0E1
            end
        end
    end
end