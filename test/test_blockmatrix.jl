module TestLibNEGFBlockMatrix

using LibNEGF, Test, SparseArrays
import LinearAlgebra

# 1 from disk, 2 is random
whereFrom = 2

if whereFrom == 1
    @testset "Blockmatrix" begin
        @testset "Blockmatrix convert and back" begin
            # for the data type BlockMatrix, check that converting
            # to it and back to the sparse format works well

            include("common_to_test.jl")

            for systemx in systemNames
                for E in Epoints
                    for k in kpoints
                        for precx in precs
                            # list of matrices to load
                            listMatsToLoad = ["H", "S", "Sc"]
                            # load in the desired precision
                            loadedMats, blockSizes = load_matrices(systemx, E, k,
                                listMatsToLoad, precx, whereFrom)
                            H = loadedMats[1]
                            S = loadedMats[2]
                            Se = loadedMats[3]
                            M = build_M_from_HS(H, S, Se, energVals[E])

                            # convert to BlockMatrix
                            Abm = bm_convert(M, blockSizes, Dict("in" => 3, "out" => 3))
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
    end
end

end