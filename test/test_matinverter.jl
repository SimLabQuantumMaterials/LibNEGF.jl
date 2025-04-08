module TestLibNEGFMatinverter

using LibNEGF, Test
import LinearAlgebra

@testset "Matinverter" begin
    @testset "Matinverter Direct Inverse Full" begin
        LinearAlgebra.BLAS.set_num_threads(4)
        include("test_matinverter_dirinvfull.jl")
    end
    @testset "Matinverter Direct Inverse Trid" begin
        LinearAlgebra.BLAS.set_num_threads(4)
        include("test_matinverter_dirinvtrid.jl")
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