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
	    T_LU = blockMatrix_factorization(T, TimingData(), CountingData())
	    luT = lu(full(T), NoPivot())
	    T_LU_Matrix = full(T_LU)
	    appT_LU = UnitLowerTriangular(T_LU_Matrix) * UpperTriangular(T_LU_Matrix)
	    @test norm(T_LU_Matrix - luT.factors) / norm(T_LU_Matrix) <= 1e-15
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
		T_fact = blockMatrix_factorization(T, TimingData(), CountingData())
	    T_app = blockMatrix_inverse(T_fact, true, TimingData(), CountingData())
		
		## Check if diag(A*si(A)) = 1 ##
		check_diag = diag(full(T)*full(T_app))
		for i in 1:npl
			@test (1 - check_diag[i]) <= 1e-4
		end

		## Check shape(si(A)) == shape(inv(A)) ##
		invT = inv(full(T))
		invT = set_sparse_Block(invT, bind, true)
		
		## Check nonzeros block of si(A) equal to inv(A) blocks ##
		for i in 1:npl
			for j in 1:npl
				if isassigned(T_app,i,j)
					@test isapprox(T_app[i,j].Full, invT[i,j].Full, atol=1e-8)
				end
			end
		end
    end
end
end # module TestLibNEGFSelectedinverse