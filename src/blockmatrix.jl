"""
	BlockMatrix type

Struct conaining the whole matrix in field `M`, factors in field `F`
and other information such as the sizes of blocks and indeces of
non zeros blocks.

# Fields
- `M::ArrayOrLU_` : To store the whole matrix.
- `F::ArrayOrLU_` : To store LU factors (and plus if needed).
- `f_inv::Bool` : Flag to know if the LU factors need to be inversed or not.
- `blockSizes::Vector{Int}`: the sizes of the blocks along the block diagonal.
- `rowInd::Vector{Int}`: the row indices where block are stored.
- `colInd::Vector{Int}`: the column indices where block are stored.
"""
mutable struct BlockMatrix
	"M"
	M::ArrayOrLU_
	"F"
	F::ArrayOrLU_
	"f_inv"
	f_inv::Bool
	"blockSizes"
	blockSizes::Vector{Int}
	"rowInd"
	rowInd::Vector{Int}
	"colInd"
	colInd::Vector{Int}

	BlockMatrix() =
	begin
		new()
	end
	BlockMatrix(M,F,blockSizes) =
	begin
		rowInd, colInd = get_rcIndex(M);
		new(M, F, false, blockSizes, rowInd, colInd)
	end
	BlockMatrix(M,blockSizes) =
	begin
		rowInd, colInd = get_rcIndex(M);
		new(M, undef, false, blockSizes, rowInd, colInd)
	end
	BlockMatrix(M, blockSizes, rowInd, colInd) =
	begin
		new(M, undef, false, blockSizes, rowInd, colInd)
	end
	BlockMatrix(M, F, finv, blockSizes, rowInd, colInd) =
	begin
		new(M, F, finv, blockSizes, rowInd, colInd)
	end
end


"""
	copy_BlockMatrix(A::BlockMatrix)::BlockMatrix

Copy a `BlockMatrix` object

# Arguments
- `A::BlockMatrix` : The matrix we copy.
"""
function copy_BlockMatrix(A::BlockMatrix)::BlockMatrix
	B = BlockMatrix()
	try B.M = copy(A.M) catch; nothing end
	try B.F = copy(A.F) catch; nothing end
	try B.f_inv = copy(A.f_inv) catch; nothing end
	try B.blockSizes = copy(A.blockSizes) catch; nothing end
	try B.rowInd = copy(A.rowInd) catch; nothing end
	try B.colInd = copy(A.colInd) catch; nothing end
	return B
end


"""
	blockMatrix_factorization!(A::BlockMatrix)

Do the LU factorization of the matrix `A::BlockMatrix`
without computing zeros blocks.
In place version !

**Warning** : `A` supposes to be symetric

# Arguments
- `A::BlockMatrix` : The matrix to be factorize.
"""
function blockMatrix_factorization!(A::BlockMatrix)::BlockMatrix
	npl = length(A.blockSizes)
	for i = 1:npl-1
		# Step 1 : create L(i,i) U(i,i)
		A.M[i,i] = lu(A.M[i,i], NoPivot())
		# List of index of block in the row/column of the diagonal
		x_update = findall(x -> x == i, A.colInd)
		y_update = findall(x -> x == i, A.rowInd)
		# Remove index of block previous the diagonal (already treated in previous iteration)
		id_start = intersect(x_update,y_update)
		xup = x_update[x_update.>id_start]
		yup = y_update[y_update.>id_start]

		# Step 2 : Generate L(i+1,i)
		### NOT OPTIMAL ###
		for k in xup
			A.M[A.rowInd[k], i] =  A.M[A.rowInd[k], i] / A.M[i,i].U
		end
		# Step 3 : Generate U(i,i+1)
		### NOT OPTIMAL ###
		for k in yup
			A.M[i, A.colInd[k]] = A.M[i,i].L \ A.M[i, A.colInd[k]]
		end
		# Step 4 : Update A(i+1,i+1)
		### NOT OPTIMAL ###
		for k in collect(Iterators.product(A.rowInd[xup], A.colInd[yup]))
			try A.M[CartesianIndex(k)] = A.M[CartesianIndex(k)] - A.M[CartesianIndex(k)[1],i] * A.M[i,CartesianIndex(k)[2]]
			catch
				A.M[CartesianIndex(k)] = - A.M[CartesianIndex(k)[1],i] * A.M[i,CartesianIndex(k)[2]]
				A.rowInd, A.colInd = set_rcIndex!(A.rowInd, A.colInd, CartesianIndex(k)[1], CartesianIndex(k)[2])
			end
		end
	end
	A.M[npl,npl] = lu(A.M[npl,npl], NoPivot())

	return A
end

#####
# Wrapper
#####

"""
	blockMatrix_factorization(A::BlockMatrix)

Do the LU factorization of the matrix `A::BlockMatrix`
without computing zeros blocks.

**Warning** : `A` supposes to be symetric

# Arguments
- `A::BlockMatrix` : The matrix to be factorize.
"""
function blockMatrix_factorization(A::BlockMatrix)::BlockMatrix
	B = copy_bm(A)

	return blockMatrix_factorization!(B)
end
