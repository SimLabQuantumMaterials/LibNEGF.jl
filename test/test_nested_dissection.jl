module TestLibNEGFNesteddissection

using LibNEGF, Test, LinearAlgebra

using Base.Threads
using Distributed
using ThreadPinning
pinthreads(:cores)
using Dagger
Dagger.MemPool.MEM_RESERVE_SWEEPS[] = 0
Dagger.MemPool.MEM_RESERVED[] = 0
using PrecompileTools: @setup_workload, @compile_workload
using Statistics
BLAS.set_num_threads(1)

@testset verbose = true "Nesteddissection" begin
    @testset "bm_get_reorder" begin
        npl = 7
        order = [1,2,3,4,5,6,7]
        reorder = bm_get_reorder_recTer(npl, nb_level=1)
        @test reorder == [1,2,3,5,6,7,4]
		reorder = bm_get_reorder_recTer(npl, nb_level=2)
		@test reorder == [1,3,2,5,7,6,4]
    end
	@testset "bm_reorder" begin
        npl = 7
		rind = [1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7]
	    cind = [1,2,1,2,3,2,3,4,3,4,5,4,5,6,5,6,7,6,7]
	    bind = repeat([10],npl)
	    T = set_sparse_Block(bind, rind, cind, true)
        reorder = bm_get_reorder_recTer(npl, nb_level=2)
		T = bm_reorder(T, reorder)
		original = Matrix(undef, npl, npl)
		original[1,1] = Block(rand(5, 5))
		original[1,3] = Block(rand(5, 5))
		original[2,2] = Block(rand(5, 5))
		original[2,3] = Block(rand(5, 5))
		original[2,7] = Block(rand(5, 5))
		original[3,1] = Block(rand(5, 5))
		original[3,2] = Block(rand(5, 5))
		original[3,3] = Block(rand(5, 5))
		original[4,4] = Block(rand(5, 5))
		original[4,6] = Block(rand(5, 5))
		original[4,7] = Block(rand(5, 5))
		original[5,5] = Block(rand(5, 5))
		original[5,6] = Block(rand(5, 5))
		original[6,4] = Block(rand(5, 5))
		original[6,5] = Block(rand(5, 5))
		original[6,6] = Block(rand(5, 5))
		original[7,2] = Block(rand(5, 5))
		original[7,4] = Block(rand(5, 5))
		original[7,7] = Block(rand(5, 5))
        @test bm_equal(T, original, false)
		@test bm_equal(T, original) broken = true
    end
    @testset "nd_factorization" begin
	    npl = 7
	    rind = [1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7]
	    cind = [1,2,1,2,3,2,3,4,3,4,5,4,5,6,5,6,7,6,7]
	    bind = repeat([10],npl)
	    T = set_sparse_Block(bind, rind, cind, true)
		n = sum(bind)
	    for i in 1:npl
	    	T[i,i].Full += n*I(bind[i])
	    end
		reorder = bm_get_reorder_recTer(npl, nb_level=2)
		T = bm_reorder(T, reorder)
		T = T * bm_transpose(T)
		T_full = full(T)
	    original = lu(T_full, NoPivot())
	    T_LU = nd_factorization(T)
		
		fT = full(T_LU)
	    appT = UnitLowerTriangular(fT) * UpperTriangular(fT)
	    @test norm(T_full - original.L*original.U) / norm(T_full) <= 1e-15
		@test norm(T_full - appT) / norm(T_full) <= 1e-15
    end
    # @testset "nd_inverse" begin
	#     npl = 5
	#     rind = [1,1,1,2,3,3,3,4,5]
	#     cind = [1,3,5,2,3,4,5,4,5]
	#     bind = repeat([100], npl)
	#     T = set_sparse_Block(bind, rind, cind, true)
	#     n = sum(bind)
	#     for i in 1:npl
	#     	T[i,i].Full += n*I(bind[i])
	#     end
	# 	T_LU = blockMatrix_factorization(T)
	#     T_app = blockMatrix_inverse(blockMatrix_factorization(T))
	#     @test norm(inv(full(T)) - full(T_app)) / norm(full(T_app)) <= 1e-15
	#     @test norm(full(T) * full(T_app) - I ) / norm(full(T) * full(T_app)) <= 1e-15
    # end
end

end # module TestLibNEGFNesteddissection