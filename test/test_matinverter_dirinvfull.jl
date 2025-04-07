# computing the direct inverse via inv(...), this test
# is added here for completeness, and as a base for the
# subsequent ones

include("common_to_test_matinverter.jl")

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
                # the convert(...) in the following line is to avoid casting
                # to ComplexF64
                T = buildTFromHS(H, S, Sc, energVals[E])
                Tdense = Array(T)
                TdenseInv = inv(Tdense)
                relErr = LinearAlgebra.norm(Tdense * TdenseInv - LinearAlgebra.I, 2) / LinearAlgebra.norm(Tdense, 2)
                # we are hardcoding this value of 1.0E3 here, as we know
                # that the conditioning of Tdense is around 1.0E3 for the
                # test matrices at hand
                @test relErr < roundoffs[precx] * 1.0E3
            end
        end
    end
end