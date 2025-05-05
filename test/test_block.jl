module TestLibNEGFBlock

using LibNEGF, Test, SparseArrays
import LinearAlgebra

@testset "Block" begin
    @testset "Block convert and back" begin
        # Check we can create an object Block type
        # from matrix format and covert back to matrix.

        # TODO ?
        # include("common_to_test_Block.jl")

        # Sample matrix
        M = rand(5,5)
        # Convert to Block type
        B = Block(M)
        # Take the LU of M
        M_lu = lu(M, NoPivot())
        # Convert to Block type
        B_lu = Block(M_lu)
        # Convert Back
        M_back = B.Full
        M_lu_back = B_lu.Factors

        @test typeof(B) <: Block
        @test typeof(B.Full) <: Matrix
        @test typeof(B_lu) <: Block
        @test typeof(B_lu.Factors) <: LU
        @test typeof(M_back) <: Matrix
        @test typeof(M_lu_back) <: LU

        # Check standard operation on it
        @test B.Full + B.Full == 2 * B.Full
        @test B + B == B.Full + B.Full
    end
end

end
