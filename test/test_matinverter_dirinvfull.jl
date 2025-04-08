# computing the direct inverse via inv(...), this test
# is added here for completeness, and as a base for the
# subsequent ones

include("common_to_test_matinverter.jl")
using Printf

for systemx in systemNames
    for E in Epoints
        for k in kpoints
            for precx in precs
                # list of matrices to load
                listMatsToLoad = ["H", "S", "Sc"]
                # then, in actual desired precision
                loadedMats, blockSizes = loadMatrices(systemx, E, k,
                    listMatsToLoad, precx)
                H = loadedMats[1]
                S = loadedMats[2]
                Sc = loadedMats[3]
                T = buildTFromHS(H, S, Sc, energVals[E])
                Tdense = Array(T)
                TdenseInv = inv(Tdense)
                relErr = LinearAlgebra.norm(Tdense * TdenseInv - LinearAlgebra.I, 2) / sqrt(size(Tdense)[1])
                # making a rough assumption on backward stability
                @test relErr < roundoffs[precx] * LinearAlgebra.cond(Tdense)
            end
        end
    end
end