module TestLibNEGFMatinverter

using LibNEGF, Test
import LinearAlgebra, PROPACK

@testset "Matinverter" begin
    # @testset "Matinverter Direct Inverse Full" begin
    #     LinearAlgebra.BLAS.set_num_threads(nrBLASthreads)
    #     include("test_matinvert_full.jl")
    # end
    # @testset "Matinverter Direct Inverse Ndiag" begin
    #     include("test_matinvertndiag_direct.jl")
    # end
    # @testset "Matinverter RGF Full" begin
    #     # under construction. This is the traditional RGF
    # end
    @testset "Matinverter RGF Ndiag" begin
        include("test_matinvertndiag_ddrgf.jl")
    end
    # @testset "Matinverter DD-RGF Full" begin
    #     # under construction. This is our (@ JSC) approach
    # end
    # @testset "Matinverter DD-RGF Ndiag" begin
    #     # under construction. This is our (@ JSC) approach
    # end
end
end