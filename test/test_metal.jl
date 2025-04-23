module TestLibNEGFMetal

using LibNEGF, Test
import LinearAlgebra, Metal

# this testset tests some basic functionality of Metal.jl

@testset "Metal" begin
    @testset "Metal Sum of Matrices" begin
        # check that adding two matrices on CPU and GPU gives
        # the same result

        include("common_to_test_matloader.jl")

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

    # @testset "Matloader Consistent Matrices" begin
    #     # check here that the built T makes sense if directly
    #     # loaded or built via S,H,\Sigma_{c}

    #     include("common_to_test_matloader.jl")

    #     for systemx in systemNames
    #         for E in Epoints
    #             for k in kpoints
    #                 for precx in precs
    #                     # list of matrices to load
    #                     listMatsToLoad = ["H", "S", "Sc"]
    #                     # load in the desired precision
    #                     loadedMats, blockSizes = load_matrices(systemx, E, k,
    #                         listMatsToLoad, precx)
    #                     H = loadedMats[1]
    #                     S = loadedMats[2]
    #                     Se = loadedMats[3]
    #                     Tbuilt = build_T_from_HS(H, S, Se, energVals[E])
    #                     # list of matrices to load
    #                     listMatsToLoad = ["T"]
    #                     # load in the desired precision
    #                     loadedMats, blockSizes = load_matrices(systemx, E, k,
    #                         listMatsToLoad, precx)
    #                     Tload = loadedMats[1]
    #                     relErr = LinearAlgebra.norm(Tload - Tbuilt, 2) / LinearAlgebra.norm(Tload, 2)
    #                     @test relErr < roundoffs[precx]
    #                 end
    #             end
    #         end
    #     end
    # end
end

end