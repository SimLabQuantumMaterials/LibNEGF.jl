module TestLibNEGFMatinverter

using LibNEGF, Test
import LinearAlgebra

nrBLASthreads = 6

@testset "Matinverter" begin
    # @testset "Matinverter Direct Inverse Full" begin
    #     LinearAlgebra.BLAS.set_num_threads(nrBLASthreads)
    #     include("test_matinverter_dirinvfull.jl")
    # end
    @testset "Matinverter Direct Inverse Ndiag" begin
        LinearAlgebra.BLAS.set_num_threads(nrBLASthreads)
        include("test_matinverter_dirinvndiag.jl")
    end
    @testset "Matinverter RGF Full" begin
        # under construction. This is the traditional RGF
    end
    @testset "Matinverter RGF Trid" begin
        # under construction. This is the traditional RGF
    end
    @testset "Matinverter DD-RGF Full" begin
        # under construction. This is our (@ JSC) approach
    end
    @testset "Matinverter DD-RGF Trid" begin
        # under construction. This is our (@ JSC) approach
    end
end
end