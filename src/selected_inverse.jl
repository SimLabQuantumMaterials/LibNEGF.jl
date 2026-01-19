"""
	prod_VecOfBlock!(A::Matrix, B::Matrix)::Matrix

Do the product of two `Block`, `A*B`.

# Arguments
- `A::Block` : the matrix on the left side.
- `B::Block` : the matrix on the right side.
"""
function prod_VecOfBlock!(res::Array, A::Array, B::Array)::Array
	for i in 1:size(A,1)
		res += prod(A[i], B[i])
	end
	return res
end

"""
	prod_VecOfBlock(A::Matrix, B::Matrix)::Matrix

Do the product of two `Block`, `A*B`.

# Arguments
- `A::Block` : the matrix on the left side.
- `B::Block` : the matrix on the right side.
"""
function prod_VecOfBlock(A::Array, B::Array, precx, td::TimingData, cd::CountingData)::Array
	temp = zeros(precx, A[1].row,B[1].col)
	for i in 1:size(A,1)
		# temp += prod(A[i], B[i])
		@timewrap td "_gemm" begin
            @countwrap cd "_gemm" A[i] B[i] temp begin
				LinearAlgebra.BLAS.gemm!('N', 'N', convert(precx, 1.0), A[i].Full, B[i].Full, convert(precx, 1.0), temp)
			end
		end
	end
	return temp
end

"""
	blockMatrix_factorization!(A::Matrix)::Matrix

Do the LU factorization on the `A` matrix in place.

# Arguments
- `A::Matrix` : The target matrix in full form.
"""
function blockMatrix_factorization!(A::Matrix, td::TimingData, cd::CountingData)::Matrix
	precx = typeof(A[1, 1].Full[1,1])
	npl = size(A, 1)
	be_zero = convert(precx, 0.0)
	be_one = convert(precx, 1.0)
	be_mone = convert(precx, -1.0)
	for i = 1:npl
		# Step 1 : create L(i,i) U(i,i)
		# A[i,i].Factors = lu(A[i,i].Full, NoPivot())
		be_getrf!(A[i,i].Full, td, cd)

		idx = []
		# Lii = Matrix(A[i,i].Factors.L)
		# Uii = Matrix(A[i,i].Factors.U)
		Lii = Matrix(LinearAlgebra.UnitLowerTriangular(A[i,i].Full))
		Uii = Matrix(LinearAlgebra.UpperTriangular(A[i,i].Full))
		for j = i+1:npl
			# Warning : supposing matrix is symmetric
			if isassigned(A,j,i)
				push!(idx,j)
				# Step 2 : Generate L(i+1,i) --- A[j,i].Full /= A[i,i].Factors.U
				be_trsm!('R', 'U', 'N', 'N', be_one, Uii, A[j,i].Full, td, cd)
			end
			if isassigned(A,i,j)
				# Step 3 : Generate U(i,i+1) --- A[i,j].Full = A[i,i].Factors.L \ A[i,j].Full
				be_trsm!('L', 'L', 'N', 'U', be_one, Lii, A[i,j].Full, td, cd)
			end
		end

		subind = collect(Iterators.product(idx,idx))
		# Step 4 : Update A(i+1:,i+1:)
		for k in subind
			k = CartesianIndex(k)
			if isassigned(A,k[1],k[2])
				# A[k].Full .-= prod(A[k[1],i], A[i,k[2]])
				be_gemm!('N', 'N', be_mone, A[k[1],i].Full, A[i,k[2]].Full, be_one, A[k[1],k[2]].Full, td, cd)
			else
				A[k[1],k[2]] = Block(zeros(A[k[1],i].row,A[i,k[2]].col))
				# A[k].Full = - prod(A[k[1],i], A[i,k[2]])
				be_gemm!('N', 'N', be_mone, A[k[1],i].Full, A[i,k[2]].Full, be_zero, A[k[1],k[2]].Full, td, cd)
			end
		end
	end

	return A
end

function blockMatrix_factorization_noT!(A::Matrix)::Matrix
	precx = typeof(A[1, 1].Full[1,1])
	npl = size(A, 1)
	for i = 1:npl
		# Step 1 : create L(i,i) U(i,i)
		# A[i,i].Factors = lu(A[i,i].Full, NoPivot())
		LinearAlgebra.LAPACK.getrf!(A[i,i].Full, collect(1:size(A[i,i].Full,1)))

		idx = []
		# Lii = Matrix(A[i,i].Factors.L)
		# Uii = Matrix(A[i,i].Factors.U)
		Lii = Matrix(LinearAlgebra.UnitLowerTriangular(A[i,i].Full))
		Uii = Matrix(LinearAlgebra.UpperTriangular(A[i,i].Full))
		for j = i+1:npl
			# Warning : supposing matrix is symmetric
			if isassigned(A,j,i)
				push!(idx,j)
				# Step 2 : Generate L(i+1,i) --- A[j,i].Full /= A[i,i].Factors.U
				LinearAlgebra.BLAS.trsm!('R', 'U', 'N', 'N', convert(precx, 1.0), Uii, A[j,i].Full)
			end
			if isassigned(A,i,j)
				# Step 3 : Generate U(i,i+1) --- A[i,j].Full = A[i,i].Factors.L \ A[i,j].Full
				LinearAlgebra.BLAS.trsm!('L', 'L', 'N', 'U', convert(precx, 1.0), Lii, A[i,j].Full)
			end
		end

		subind = collect(Iterators.product(idx,idx))
		# Step 4 : Update A(i+1:,i+1:)
		for k in subind
			k = CartesianIndex(k)
			if isassigned(A,k[1],k[2])
				# A[k].Full .-= prod(A[k[1],i], A[i,k[2]])
				LinearAlgebra.BLAS.gemm!('N', 'N', convert(precx, -1.0), A[k[1],i].Full, A[i,k[2]].Full, convert(precx, 1.0), A[k[1],k[2]].Full)
			else
				A[k[1],k[2]] = Block(zeros(A[k[1],i].row,A[i,k[2]].col))
				# A[k].Full = - prod(A[k[1],i], A[i,k[2]])
				LinearAlgebra.BLAS.gemm!('N', 'N', convert(precx, -1.0), A[k[1],i].Full, A[i,k[2]].Full, convert(precx, 0.0), A[k[1],k[2]].Full)
			end
		end
	end

	return A
end

#####
# Wrapper
#####

"""
	blockMatrix_factorization(A::Matrix)::Matrix

Do the LU factorization on the `A` matrix. 
Diagonal factors are stored in `Factors` field of `A`.
The algorithm is generic and avoid computation with zero blocks.

The computation is done block by block along the diagonal starting from the first diagonal block (top left) to the last diagonal block (bottom right).

**Warning** : `A` supposes to be symetric

# Arguments
- `A::Matrix` : The target matrix.
"""
function blockMatrix_factorization(A::Matrix, td::TimingData, cd::CountingData)::Matrix
	B = bm_copy(A)

	return blockMatrix_factorization!(B, td, cd)
end

"""
	blockMatrix_inverse!(A::Matrix, fillin::Bool=false, td::TimingData, cd::CountingData)::Matrix

Do the selected inverse of the matrix factorize `A` in place.

# Arguments
- `A::Matrix` : The targeted matrix in factorize form.
- `NoFillin::Bool` : Flag to know if we consider filled block during the computation.
"""
function blockMatrix_inverse!(A::Matrix, NoFillin::Bool, td::TimingData, cd::CountingData)::Matrix
	precx = typeof(A[1, 1].Full[1,1])
	npl = size(A,1)
	be_one = convert(precx, 1.0)
	be_mone = convert(precx, -1.0)
	# Step 0 : Compute A(i,i)
	# A[npl,npl] = Block(A[npl,npl].Factors.U \ (A[npl,npl].Factors.L \ I)) # No optimal this!
	be_getri!(A[npl,npl].Full, td, cd)

	for i = npl-1:-1:1
		# Take index of non zeros blocks in A[i+1:npl,i]
		rowA21, _ = get_rcIndexAt(A, i+1, npl, i, i)
		# Store updated L[i+1:,i]
		Lupdated = Vector{Matrix{precx}}(undef,npl-i)
		# Store updated U[i,i+1:]
		Uupdated = Vector{Matrix{precx}}(undef,npl-i)
		for j in i+1:npl
			# Get commun index of non zeros between sub-matrix A[i+1:npl,i+1:npl] and A[i+1:npl,i].
			_, colA22 = get_rcIndexAt(A, j, j, i+1, npl)
			comIdx = intersect(rowA21, colA22)
			if !isempty(comIdx)
				# WARNING :  Ensure the Step 2 will have the same result since its symmetric (not the case else!)
				# Step 1 : Compute A(i+1:npl,i)
				Lupdated[j-i] = prod_VecOfBlock(A[j,comIdx], A[comIdx,i], precx, td, cd)
				# Lupdated[j-i] = - temp / A[i,i].Factors.L
				be_trsm!('R', 'L', 'N', 'U', be_mone, A[i,i].Full, Lupdated[j-i], td, cd)
				# Step 2 : Compute A(i,i+1:npl)
				Uupdated[j-i] = prod_VecOfBlock(A[i,comIdx], A[comIdx,j], precx, td, cd)
				# Uupdated[j-i] = - A[i,i].Factors.U \ temp
				be_trsm!('L', 'U', 'N', 'N', be_mone, A[i,i].Full, Uupdated[j-i], td, cd)
			end
		end

		# Step 3 : Update A(i,i)
		# It will first compute the U[i,i+1:]A[i+1:,i+1:]L[i+1:,i] part, block by block.
		# Set the current block at zero
		# A[i,i].Full = zeros(precx, A[i,i].row, A[i,i].col)
		A[i,i].Factors = zeros(precx, A[i,i].row, A[i,i].col)
		for j in i+1:npl
			# if Uupdated not exist, Lupdated also by symmetric so we iterate on the next block
			if isassigned(Uupdated,j-i)
				if isassigned(A,j,i)
					# A[i,i].Full += Uupdated[j-i] * A[j,i].Full
					be_gemm!('N', 'N', be_one, Uupdated[j-i], A[j,i].Full, be_one, A[i,i].Factors, td, cd)
				else
						if NoFillin
							continue
						end
						A[j,i] = Block(size(Lupdated[j-i],1), size(Lupdated[j-i],2))
						A[i,j] = Block(size(Uupdated[j-i],1), size(Uupdated[j-i],2))
				end
				A[j,i].Full = Lupdated[j-i]
				A[i,j].Full = Uupdated[j-i]
			end
		end
		# Compute the rest part of A(i,i)
		# A[i,i] = Block(A[i,i].Factors.U \ (A[i,i].Factors.L \ I) - (A[i,i].Full / A[i,i].Factors.L))
		# A[i,i] = Block(LinearAlgebra.UpperTriangular(A[i,i].Full) \ (LinearAlgebra.UnitLowerTriangular(A[i,i].Full) \ I) - (A[i,i].Factors / LinearAlgebra.UnitLowerTriangular(A[i,i].Full)))
		be_trsm!('R', 'L', 'N', 'U', be_one, A[i,i].Full, A[i,i].Factors, td, cd)
		be_getri!(A[i,i].Full, td, cd)
		A[i,i] = Block(A[i,i].Full - A[i,i].Factors)
	end
	
	return A
end

"""
	blockMatrix_inverse(A::Matrix, opti::Bool=false)::Matrix

Do the selected inverse of the matrix factorize `A`.
The algorithm is generic and avoid computation with zero blocks.

The computation is done block by block along the diagonal starting from the last diagonal block (bottom right) to the first diagonal block (top left).

**Warning** : `A` supposes to be symetric

# Arguments
- `A::Matrix` : The targeted matrix in factorize form. 
- `NoFillin::Bool` : Flag to know if we consider filled block during the computation.
"""
function blockMatrix_inverse(A::Matrix, NoFillin::Bool, td::TimingData, cd::CountingData)::Matrix
	B = bm_copy(A)

	return blockMatrix_inverse!(B, NoFillin, td, cd)
end