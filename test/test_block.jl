module TestLibNEGFBlock

using LibNEGF, Test
import LinearAlgebra

@testset "Block" begin
    @testset "operator(+)::Block" begin
	    npl = 5
	    M = Matrix(undef,npl,npl)
	    N = Matrix(undef,npl,npl)
	    a = rand(5,4)
	    b = rand(3,3)
	    c = rand(5,5)
	    d = rand(5,4)
	    M[1,2] = Block(a)
	    M[3,3] = Block(b)	
	    N[1,1] = Block(c)
	    N[1,2] = Block(d)
	    res = sum_BlockMatrix(M,N)
	    @test res[1,1] == c
	    @test res[1,2] == a+d
	    @test res[3,3] == b
	    @test res[2,2] == a broken=true
    end
    @testset "operator(*)::Block" begin
	    npl = 5
	    M = Matrix(undef,npl,npl)
	    N = Matrix(undef,npl,npl)
	    a = rand(4,5)
	    b = rand(3,3)
	    c = rand(5,5)
	    d = rand(3,4)
	    e = rand(5,4)
	    f = rand(5,3)
	    g = rand(4,4)
	    M[1,2] = Block(e)
	    M[1,3] = Block(f)
	    M[2,1] = Block(a)
	    M[3,3] = Block(b)	
	    N[1,1] = Block(c)
	    N[2,2] = Block(g)
	    N[3,2] = Block(d)
	    @test res[2,1] == a*c
	    @test res[3,2] == b*d
	    @test res[1,2] == (e*g+f*d)
	    @test res[1,1] == e*g+f*d broken=true
    end
    @testset "operator(==)::Block" begin
	    M = Matrix(undef,5,5)
	    M[1,1] = Block(rand(5,5))
	    M[2,3] = Block()
	    N = M
	    @test N[1,1] == M[1,1]
	    @test N[2,3] == M[2,3]
    end
    @testset "copy::Block" begin
	    M = Block(undef,5,5)
	    B = copy(M)
	    @test B == M
    end
end