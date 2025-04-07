module TestLibNEGFMatloader

using LibNEGF, Test
import LinearAlgebra, Printf

@testset "Matloader" begin
    @testset "Matloader IO" begin
        # check that the read matrices are built properly with
        # the desired precision

        include("common_to_test_matloader.jl")

        for systemx in systemNames
            for E in Epoints
                for k in kpoints
                    for precx in precs
                        # list of matrices to load
                        listMatsToLoad = ["H", "S", "Sc", "T", "Gr"]
                        # first, load in F64
                        loadedMatsF64, blockSizes = loadMatrices(systemx, E, k,
                            listMatsToLoad, ComplexF64)
                        # then, in actual desired precision
                        loadedMats, blockSizes = loadMatrices(systemx, E, k,
                            listMatsToLoad, precx)
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
        # check here that the built T makes sense if directly
        # loaded or built via S,H,\Sigma_{c}

        include("common_to_test_matloader.jl")

        for systemx in systemNames
            for E in Epoints
                for k in kpoints
                    for precx in precs
                        # list of matrices to load
                        listMatsToLoad = ["H", "S", "Sc"]
                        # load in the desired precision
                        loadedMats, blockSizes = loadMatrices(systemx, E, k,
                            listMatsToLoad, precx)
                        H = loadedMats[1]
                        S = loadedMats[2]
                        Sc = loadedMats[3]
                        Tbuilt = buildTFromHS(H, S, Sc, energVals[E])
                        # list of matrices to load
                        listMatsToLoad = ["T"]
                        # load in the desired precision
                        loadedMats, blockSizes = loadMatrices(systemx, E, k,
                            listMatsToLoad, precx)
                        Tload = loadedMats[1]
                        relErr = LinearAlgebra.norm(Tload - Tbuilt, 2) / LinearAlgebra.norm(Tload, 2)
                        @test relErr < roundoffs[precx]
                    end
                end
            end
        end
    end
end

end