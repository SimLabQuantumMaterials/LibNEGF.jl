"""
	prod_VecOfBlock(A::Matrix, B::Matrix)::Matrix

Do the product of two `Block`, `A*B`.

# Arguments
- `A::Block` : the matrix on the left side.
- `B::Block` : the matrix on the right side.
"""
function prod_VecOfBlock(A::Array, B::Array)::Array
	temp = zeros(A[1].row,B[1].col)
	for i in 1:size(A,1)
		temp += prod(A[i], B[i])
	end
	return temp
end

"""
	blockMatrix_factorization!(A::Matrix)::Matrix

Do the LU factorization on the `A` matrix in place.

# Arguments
- `A::Matrix` : The target matrix in full form.
"""
function blockMatrix_factorization!(A::Matrix)::Matrix
	npl = size(A, 1)
	for i = 1:npl
		# Step 1 : create L(i,i) U(i,i)
		A[i,i].Factors = lu(A[i,i].Full, NoPivot())

		idx = []
		for j = i+1:npl
			# Warning : supposing matrix is symmetric
			if isassigned(A,j,i)
				push!(idx,j)
				# Step 2 : Generate L(i+1,i)
				A[j,i].Full = A[j,i].Full / A[i,i].Factors.U
			end
			if isassigned(A,i,j)
				# Step 3 : Generate U(i,i+1)
				A[i,j].Full = A[i,i].Factors.L \ A[i,j].Full
			end
		end
		# Step 4 : Update A(i+1:,i+1:)
		for k in collect(Iterators.product(idx,idx))
			k = CartesianIndex(k)
			if isassigned(A,k[1],k[2])
				A[k].Full = A[k].Full - prod(A[k[1],i], A[i,k[2]])
			else
				A[k] = Block(A[k[1],i].row,A[i,k[2]].col)
				A[k].Full = - prod(A[k[1],i], A[i,k[2]])
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
function blockMatrix_factorization(A::Matrix)::Matrix
	B = bm_copy(A)

	return blockMatrix_factorization!(B)
end

"""
	blockMatrix_inverse!(A::Matrix, fillin::Bool=false)::Matrix

Do the selected inverse of the matrix factorize `A` in place.

# Arguments
- `A::Matrix` : The targeted matrix in factorize form.
- `NoFillin::Bool` : Flag to know if we consider filled block during the computation.
"""
function blockMatrix_inverse!(A::Matrix, NoFillin::Bool=false)::Matrix
	npl = size(A,1)
	# Step 0 : Compute A(i,i)
	A[npl,npl] = Block(A[npl,npl].Factors.U \ (A[npl,npl].Factors.L \ I))

	for i = npl-1:-1:1
		# Take index of non zeros blocks in A[i+1:npl,i]
		rowA21, colA21 = get_rcIndexAt(A, i+1, npl, i, i)
		# Store updated L[i+1:,i]
		Lupdated = Vector(undef,npl-i)
		# Store updated U[i,i+1:]
		Uupdated = Vector(undef,npl-i)
		for j in i+1:npl
			# Get commun index of non zeros between sub-matrix A[i+1:npl,i+1:npl] and A[i+1:npl,i].
			rowA22, colA22 = get_rcIndexAt(A, j, j, i+1, npl)
			comIdx = intersect(rowA21, colA22)
			if !isempty(comIdx)
				# WARNING :  Ensure the Step 2 will have the same result since its symmetric (not the case else!)
				# Step 1 : Compute A(i+1:npl,i)
				temp = prod_VecOfBlock(A[j,comIdx], A[comIdx,i])
				Lupdated[j-i] = - temp / A[i,i].Factors.L
				# Step 2 : Compute A(i,i+1:npl)
				temp = prod_VecOfBlock(A[i,comIdx], A[comIdx,j])
				Uupdated[j-i] = - A[i,i].Factors.U \ temp
			end
		end

		# Step 3 : Update A(i,i)
		# It will first compute the U[i,i+1:]A[i+1:,i+1:]L[i+1:,i] part, block by block.
		# Set the current block at zero
		A[i,i].Full = zeros(A[i,i].row, A[i,i].col)
		for j in i+1:npl
			# if Uupdated not exist, Lupdated also by symmetric so we iterate on the next block
			if isassigned(Uupdated,j-i)
				if isassigned(A,j,i)
					A[i,i].Full += Uupdated[j-i] * A[j,i].Full
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
		A[i,i] = Block(A[i,i].Factors.U \ (A[i,i].Factors.L \ I) - (A[i,i].Full / A[i,i].Factors.L))
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
function blockMatrix_inverse(A::Matrix, NoFillin::Bool=false)::Matrix
	B = bm_copy(A)

	return blockMatrix_inverse!(B, NoFillin)
end