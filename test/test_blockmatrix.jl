module TestLibNEGFBlockMatrix

using LibNEGF, Test, SparseArrays, Random
import LinearAlgebra

# fix seed to have reproducible tests
Random.seed!(1234)

@testset "Blockmatrix" begin
    @testset "Blockmatrix convert and back" begin
        # for the data type BlockMatrix, check that converting
        # to it and back to the sparse format works well

        include("common_to_test.jl")

        for k in kpoints
            for precx in [ComplexF64]
                # # list of matrices to load
                # listMatsToLoad = ["H", "S", "Sc"]
                # # load in the desired precision
                # loadedMats, blockSizes = load_matrices(systemx, E, k,
                #     listMatsToLoad, precx, whereFrom)
                # H = loadedMats[1]
                # S = loadedMats[2]
                # Se = loadedMats[3]
                # M = build_M_from_HS(H, S, Se, energVals[E])

                # TODO : I think this tests is not making a lot of sense at the moment

                Abm = bm_create_synthetic_random(npl, blockSize, precx, false)
                M = bm_convert(Abm)

                # # convert to BlockMatrix
                # Abm = bm_convert(M, blockSizes, Dict("in" => 3, "out" => 3))

                # convert back to sparse
                Asp = bm_convert(Abm)

                relErr = LinearAlgebra.norm(M - Asp, 2) / LinearAlgebra.norm(M, 2)
                @test relErr < roundoffs[precx]
                @test nnz(Asp) == nnz(M)
                @test typeof(Asp) == typeof(M)
            end
        end
    end
end
end