module TestLibNEGFSelectedinverse

using LibNEGF, Test, LinearAlgebra

@testset "Selectedinverse" begin
    @testset "blockMatrix_factorization" begin
	    npl = 5
	    rind = [1,1,1,2,3,3,3,4,5]
	    cind = [1,3,5,2,3,4,5,4,5]
	    bind = repeat([100],npl)
	    T = set_sparse_Block(bind, rind, cind, true)
	    n = sum(bind)
	    for i in 1:npl
	    	T[i,i].Full += n*I(bind[i])
	    end
	    T_LU = blockMatrix_factorization(T)
	    luT = lu(full(T), NoPivot())
	    T_LU_Matrix = full(T_LU)
	    appT_LU = UnitLowerTriangular(T_LU_Matrix) * UpperTriangular(T_LU_Matrix)
	    @test norm(full(T_LU) - luT.factors) / norm(full(T_LU)) <= 1e-15
	    @test norm(appT_LU - full(T)) / norm(appT_LU) <= 1e-15
    end
    @testset "blockMatrix_inverse" begin
	    npl = 5
	    rind = [1,1,1,2,3,3,3,4,5]
	    cind = [1,3,5,2,3,4,5,4,5]
	    bind = repeat([100], npl)
	    T = set_sparse_Block(bind, rind, cind, true)
	    n = sum(bind)
	    for i in 1:npl
	    	T[i,i].Full += n*I(bind[i])
	    end
	    T_app = blockMatrix_inverse(blockMatrix_factorization(T))
	    @test norm(inv(full(T)) - full(T_app)) / norm(full(T_app)) <= 1e-15
	    @test norm(full(T) * full(T_app) - I ) / norm(full(T) * full(T_app)) <= 1e-15
    end
end
end # module TestLibNEGFSelectedinverse