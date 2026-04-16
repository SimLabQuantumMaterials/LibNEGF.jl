module TestLibNEGFMatinverter

using LibNEGF, Test
import LinearAlgebra

@testset "Matinverter" begin
    # @testset "Matinverter Direct Inverse Full" begin
    #     LinearAlgebra.BLAS.set_num_threads(nrBLASthreads)
    #     include("test_matinvert_full.jl")
    # end
    # @testset "Matinverter Direct Inverse Ndiag" begin
    #     include("test_matinvertndiag_direct.jl")
    # end
    @testset "Matinverter RGF Ndiag" begin
        include("test_matinvertndiag_rgf.jl")
    end
    @testset "Matinverter DDRGF Ndiag" begin
        include("test_matinvertndiag_ddrgf.jl")
    end
end
end