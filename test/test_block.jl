module TestLibNEGFBlock

using LibNEGF, Test
import LinearAlgebra

@testset "Block" begin
    @testset "operator(+)::Block" begin
        npl = 5
        M = Matrix(undef, npl, npl)
        N = Matrix(undef, npl, npl)
        a = rand(5, 4)
        b = rand(3, 3)
        c = rand(5, 5)
        d = rand(5, 4)
        M[1, 2] = Block(a)
        M[3, 3] = Block(b)
        N[1, 1] = Block(c)
        N[1, 2] = Block(d)
        res = sum_BlockMatrix(M, N)
        @test res[1, 1] == c
        @test res[1, 2] == a + d
        @test res[3, 3] == b
        @test res[2, 2] == a broken = true
    end
    @testset "operator(*)::Block" begin
        npl = 5
        M = Matrix(undef, npl, npl)
        N = Matrix(undef, npl, npl)
        a = rand(4, 5)
        b = rand(3, 3)
        c = rand(5, 5)
        d = rand(3, 4)
        e = rand(5, 4)
        f = rand(5, 3)
        g = rand(4, 4)
        M[1, 2] = Block(e)
        M[1, 3] = Block(f)
        M[2, 1] = Block(a)
        M[3, 3] = Block(b)
        N[1, 1] = Block(c)
        N[2, 2] = Block(g)
        N[3, 2] = Block(d)
        res = prod_BlockMatrix(M, N)
        @test res[2, 1] == a * c
        @test res[3, 2] == b * d
        @test res[1, 2] == (e * g + f * d)
        @test res[1, 1] == e * g + f * d broken = true
    end
    @testset "operator(==)::Block" begin
        M = Matrix(undef, 5, 5)
        M[1, 1] = Block(rand(5, 5))
        M[2, 3] = Block()
        N = M
        @test N[1, 1] == M[1, 1]
        @test N[2, 3] == M[2, 3]
    end
    @testset "copy::Block" begin
        M = Block(5, 5)
        B = copy(M)
        @test B == M
    end
    @testset "bm_equal" begin
        M = Matrix(undef, 5, 5)
        N = Matrix(undef, 4, 5)
        @test bm_equal(M, N) broken = true
        N = Matrix(undef, 5, 5)
        N[4, 4] = Block()
        @test bm_equal(M, N) broken = true
        N = M
        @test bm_equal(M, N)
        M[1, 1] = Block(rand(5, 5))
        M[2, 3] = Block()
        N = M
        @test bm_equal(M, N)
    end
    @testset "bm_copy" begin
        M = Matrix(undef, 5, 5)
        M[1, 1] = Block(rand(5, 5))
        M[2, 3] = Block()
        N = bm_copy(M)
        @test bm_equal(N, M)
    end
    @testset "get_rcIndex" begin
        M = Matrix(undef, 5, 5)
        idx, idy = get_rcIndex(M)
        @test isempty(idx)
        @test isempty(idy)
        idx = [1, 2, 3]
        idy = [1, 3]
        M[idx, idy] .= 1
        @test get_rcIndex(M) == ([1, 1, 2, 2, 3, 3], [1, 3, 1, 3, 1, 3])
    end
    @testset "get_rcIndexAt" begin
        M = Matrix(undef, 5, 5)
        idx = get_rcIndexAt(M)
        idy = get_rcIndexAt(M, 1, size(M,1), 1, size(M,2), true)
        @test isempty(idx)
        @test isempty(idy)
        idx = [1, 2, 3]
        idy = [1, 3]
        M[idx, idy] .= 1
        @test get_rcIndexAt(M) == ([1, 1, 2, 2, 3, 3], [1, 3, 1, 3, 1, 3]) broken = true
        @test get_rcIndexAt(M, 1, 1) == [1, 1]
        @test get_rcIndexAt(M, 1, 1, 1, size(M,2), false) == [1, 3]
    end
    @testset "bm_similar" begin
        M = Matrix(undef, 5, 5)
        M[1, 1] = Block(rand(5, 5))
        M[2, 3] = Block()
        N = bm_similar(M)
        @test bm_equal(N, M) broken = true
        O = Matrix(undef, 5, 5)
        @test bm_equal(N, O)
    end
    @testset "get_blockSizes" begin
        npl = 5
        M = Matrix(undef, npl, npl)
        verif = []
        for i in 1:5
            M[i, i] = Block(rand(i, 2))
            push!(verif, i)
        end
        bl = get_blockSizes(M)
        @test bl[1] == verif
        # Think about a critical case
        @test bl[1] == 0 broken = true
    end
    @testset "full" begin
        origin = rand(5, 5)
        t2 = rand(4, 4)
        M = Matrix(undef, 3, 3)
        M[1, 1] = M[2, 2] = Block(origin)
        M[3, 3] = Block(t2)
        # @test typeof(origin) <: Matrix
        # @test typeof(M) <: Matrix
        # @test typeof(M[2, 2]) <: Block
        M = full(M)
        for elem in M
            @test typeof(elem) <: Number
        end
    end
    @testset "set_sparse_Block" begin
        M = rand(200, 200)
        npl = 2
        b = div(size(M, 1), npl)
        N = set_sparse_Block(M, [b, b])
        # O = set_sparse_Block(M, npl)
        @test M == full(N)
        # @test M == full(O)
    end
end

end