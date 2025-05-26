module TestLibNEGFMetal

using LibNEGF, Test
import LinearAlgebra, Metal

# this testset tests some basic functionality of Metal.jl

@testset "Metal" begin
    @testset "Metal Sum of Matrices" begin
        # check that adding two matrices on CPU and GPU gives
        # the same result

        include("common_to_test.jl")

        for systemx in systemNames
            for E in Epoints
                for k in kpoints
                    for precx in precs
                        # list of matrices to load
                        listMatsToLoad = ["H", "S"]
                        # load them to CPU memory first
                        loadedMats, blockSizes = load_matrices(systemx, E, k,
                            listMatsToLoad, precx)
                        # we focus here on testing dense matrices with Metal.jl
                        # and we omit sparse matrices
                        H = loadedMats[1]
                        H = Array(H)
                        S = loadedMats[2]
                        S = Array(S)
                        # add them on the CPU
                        HpS1 = H + S
                        # now, add them on the GPU
                        Hmtl = Metal.MtlArray(H)
                        Smtl = Metal.MtlArray(S)
                        HpSmtl = Hmtl + Smtl
                        HpS2 = similar(HpS1)
                        copyto!(HpS2, HpSmtl)
                        relErr = LinearAlgebra.norm(HpS1 - HpS2, 2) / LinearAlgebra.norm(HpS1, 2)
                        @test relErr < roundoffs[precx]
                    end
                end
            end
        end
    end
end

end