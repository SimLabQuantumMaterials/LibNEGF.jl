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
            loadedMats, blockSizes = loadMatrices(systemx, E, k,
                listMatsToLoad, ComplexF64)
            H = loadedMats[1]
            S = loadedMats[2]
            Se = loadedMats[3]
            T = buildTFromHS(H, S, Se, energVals[E])
            condNum = LinearAlgebra.cond(Array(T))

            for precx in precs
                # load matrices and build T
                listMatsToLoad = ["H", "S", "Sc"]
                loadedMats, blockSizes = loadMatrices(systemx, E, k,
                    listMatsToLoad, precx)
                H = loadedMats[1]
                S = loadedMats[2]
                Se = loadedMats[3]
                T = buildTFromHS(H, S, Se, energVals[E])
                Tdense = Array(T)
                TdenseInv = inv(Tdense)
                relErr = LinearAlgebra.norm(Tdense * TdenseInv - LinearAlgebra.I, 2) / sqrt(size(Tdense)[1])
                # making a rough assumption on backward stability
                @test relErr < roundoffs[precx] * condNum
            end
        end
    end
end