module TestLibNEGFMatloader

using LibNEGF, Test
import LinearAlgebra

# 1 from disk, 2 is random
whereFrom = 2

if whereFrom == 1
    @testset "Matloader" begin
        @testset "Matloader IO" begin
            # check that the read matrices are built properly with
            # the desired precision

            include("common_to_test.jl")

            for systemx in systemNames
                for E in Epoints
                    for k in kpoints
                        for precx in [ComplexF64]
                            # list of matrices to load
                            listMatsToLoad = ["H", "S", "Sc", "T", "Gr"]
                            # first, load in F64
                            loadedMatsF64, blockSizes = load_matrices(systemx, E, k,
                                listMatsToLoad, ComplexF64, whereFrom)
                            # then, in actual desired precision
                            loadedMats, blockSizes = load_matrices(systemx, E, k,
                                listMatsToLoad, precx, whereFrom)
                            for ix in 1:size(listMatsToLoad)[1]
                                # now, check that the matrices loaded to the desired precision
                                # match with the F64 ones up to the corresponding accuracy. we
                                # use the Frobenius norm for this
                                MF64 = loadedMatsF64[ix]
                                M = loadedMats[ix]
                                relErr = LinearAlgebra.norm(MF64 - M, 2) / LinearAlgebra.norm(MF64, 2)
                                @test relErr < roundoffs[precx]
                            end
                        end
                    end
                end
            end
        end

        @testset "Matloader Consistent Matrices" begin
            # check here that the built M makes sense if directly
            # loaded or built via S,H,\Sigma_{c}

            include("common_to_test.jl")

            for systemx in systemNames
                for E in Epoints
                    for k in kpoints
                        for precx in [ComplexF64]
                            # list of matrices to load
                            listMatsToLoad = ["H", "S", "Sc"]
                            # load in the desired precision
                            loadedMats, blockSizes = load_matrices(systemx, E, k,
                                listMatsToLoad, precx, whereFrom)
                            H = loadedMats[1]
                            S = loadedMats[2]
                            Se = loadedMats[3]
                            Mbuilt = build_M_from_HS(H, S, Se, energVals[E])
                            # list of matrices to load
                            listMatsToLoad = ["T"]
                            # load in the desired precision
                            loadedMats, blockSizes = load_matrices(systemx, E, k,
                                listMatsToLoad, precx, whereFrom)
                            Mload = loadedMats[1]
                            relErr = LinearAlgebra.norm(Mload - Mbuilt, 2) / LinearAlgebra.norm(Mload, 2)
                            @test relErr < roundoffs[precx]
                        end
                    end
                end
            end
        end
    end
end

end