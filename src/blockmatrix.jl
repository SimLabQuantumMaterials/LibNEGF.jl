"""
	Block type

# Fields
- `Full::Matrix` : To store the full block matrix.
- `Factors::LU` : To store LU factors (and plus if needed).
- `f_inv::Bool` : Flag to know if the LU factors need to be inversed or not.
- `row::Vector{Int}`: the row size.
- `col::Vector{Int}`: the column size.
"""
mutable struct Block
	"Full"
	Full::Union{Array,UndefInitializer}
	"Factors"
	Factors::Union{LU,UndefInitializer}
	"f_inv"
	f_inv::Bool
	"row"
	row::Int
	"col"
	col::Int

	Block() =
	begin
		new()
	end
	Block(M) =
	begin
		new(M, undef, false, size(M,1), size(M,2))
	end
	Block(M::LU) =
	begin
		new(undef, M, false, size(M,1), size(M,2))
	end
	Block(row, col) =
	begin
		new(undef, undef, false, row, col)
	end
end

"""
	get_rcIndex(M::Matrix, nrows::Int = size(M,1), ncols::Int = size(M,2))::Tuple{Vector{Int}, Vector{Int}}

Get the rows and columns index where the matrix `M` is not `undef`.

# Arguments
- `M::Matrix` : the matrix we want to analyze.
- `nrows::Int` : the number of row we analyze.
- `ncols::Int` : the number of column we analyze.
"""
function get_rcIndex(M::Matrix{Block}, nrows::Int = size(M,1), ncols::Int = size(M,2))::Tuple{Vector{Int}, Vector{Int}}

	rowInd = Vector{Int}()
	colInd = Vector{Int}()

	for i in 1:nrows
		for j in 1:ncols
			if isassigned(M,i,j)
				push!(rowInd,i)
				push!(colInd,j)
			end
		end
	end

	return rowInd, colInd
end

"""
	get_rcIndex(M::Array, nrows::Int = size(M,1), ncols::Int = size(M,2))::Tuple{Vector{Int}, Vector{Int}}

Get the rows and columns index where the matrix `M` is not `undef`.

# Arguments
- `M::Array` : the matrix we want to analyze.
- `nrows::Int` : the number of row we analyze.
- `ncols::Int` : the number of column we analyze.
"""
function get_rcIndex(M::Array, nrows::Int = size(M,1), ncols::Int = size(M,2))::Tuple{Vector{Int}, Vector{Int}}

	rowInd = Vector{Int}()
	colInd = Vector{Int}()

	for i in 1:nrows
		for j in 1:ncols
			if isassigned(M,i,j)
				push!(rowInd,i)
				push!(colInd,j)
			end
		end
	end

	return rowInd, colInd
end

"""
	get_rcIndexAt(M::Matrix, rowB::Int, rowE::Int, colB::Int, colE::Int)::Tuple{Vector{Int}, Vector{Int}}

Get the rows and columns index where the matrix `M` is not `undef` for specific index.

# Arguments
- `M::Matrix` : the matrix we want to analyze.
- `rowB::Int` : the starting index of row we analyze.
- `rowE::Int` : the ending index of row we analyze.
- `colB::Int` : the starting index of column we analyze.
- `colE::Int` : the ending index of column we analyze.
"""
function get_rcIndexAt(M::Matrix, rowB::Int, rowE::Int, colB::Int, colE::Int)::Tuple{Vector{Int}, Vector{Int}}

	rowInd = Vector{Int}()
	colInd = Vector{Int}()

	for i in rowB:rowE
		for j in colB:colE
			if isassigned(M, i, j)
				push!(rowInd,i)
				push!(colInd,j)
			end
		end
	end

	return rowInd, colInd
end

"""
	copy(A::Block)::Block

Copy a `Block` object.
"""
function Base.copy(A::Block)::Block
	B = Block()
	try B.Full = copy(A.Full) catch; nothing end
	try B.Factors = copy(A.Factors) catch; nothing end
	try B.f_inv = copy(A.f_inv) catch; nothing end
	try B.row = copy(A.row) catch; nothing end
	try B.col = copy(A.col) catch; nothing end
	return B
end

"""
	copy_bm(A::Matrix)::Matrix

Copy a matrix that contains `Block` inside.
"""
function copy_bm(A::Matrix)::Matrix
	B = Matrix(undef,size(A,1),size(A,2))
	for i in 1:size(A,1)
		for j in 1:size(A,2)
			if isassigned(A,i,j)
				B[i,j] = copy(A[i,j])
			end
		end
	end
	return B
end

"""
	get_blockSizes(A::Matrix[, npl::Int=size(A,1)])::Vector{Int}

Get the size of each block along the diagonal from matrix `A`.

# Arguments
- `A::Matrix` : the target block matrix.
- `npl::Int` : the number of block along the diagonal.
"""
function get_blockSizes(A::Matrix, npl::Int=size(A,1))::Vector{Int}
	b::Vector{Int} = Vector{Int}(undef, npl)
	for i = 1:npl
		b[i] = A[i,i].row
	end
	return b
end

"""
	Base.prod(A::Block, B::Block)::Array

Do `A*B` product of their `Full` fields.
"""
function Base.prod(A::Block, B::Block)::Array
	return A.Full * B.Full
end

"""
	prod_BM(A::Matrix, B::Matrix)::Matrix

Do the product of two `Block`, `A*B`.

# Arguments
- `A::Block` : the matrix on the left side.
- `B::Block` : the matrix on the right side.
"""
function prod_BM(A::Array, B::Array)::Array
	temp = zeros(A[1].row,B[1].col)
	for i in 1:size(A,2)
		temp += prod(A[i], B[i])
	end
	return temp
end

"""
	convert_ALU2M(A::Matrix, rind::Vector{Int}, cind::Vector{Int})::Array

Convert a block matrix of type `Matrix` to full matrix.

# Arguments
- `A::Matrix` : A block matrix.
- `rind::Vector{Int}` : row indeces vector.
- `cind::Vector{Int}` : column indeces vector.
"""
function convert_ALU2M(A::Matrix, rind::Vector{Int}, cind::Vector{Int})::Array
	npl = size(A,1)
	b = get_blockSizes(A)
	m = sum(b)
	# B = zeros(typeof(A[1,1][1,1]),m,m)
	B = zeros(Float64,m,m)

	for j in 1:length(rind)
		idx = 1 + sum(b[1:rind[j]-1]) : sum(b[1:rind[j]])
		idy = 1 + sum(b[1:cind[j]-1]) : sum(b[1:cind[j]])
		if typeof(A[rind[j],cind[j]].Full) <: LU
			B[idx,idy] = A[rind[j],cind[j]].Full.factors
		else
			B[idx,idy] = A[rind[j],cind[j]].Full
		end
	end

	return B
end

"""
	convert_M2Block(A::Matrix)::Block

Convert a `Matrix` type object `A` into `Block` type

# Arguments
- `A::Matrix` : The matrix object we convert into `Block` type.
"""
function convert_M2Block(A::Matrix)::Block
	return Block(A)
end

"""
	convert_M2Block(A::LU)::Block

Convert a `LU` type object `A` into `Block` type

# Arguments
- `A::LU` : The matrix of factors object we convert into `Block` type.
"""
function convert_M2Block(A::LU)::Block
	return Block(A)
end

"""
	convert_ALU2M(A::Matrix)::Array

Convert a block matrix of type `Matrix` to full matrix.

# Arguments
- `A::Matrix`: A block matrix.
"""
function convert_ALU2M(A::Matrix)::Array
	npl = size(A,1)
	b = get_blockSizes(A)
	m = sum(b)
	# B = zeros(typeof(A[1,1][1,1]),m,m)
	B = zeros(Float64,m,m)

	for i = 1:npl
		for j = 1:npl
			idx = 1 + sum(b[1:i-1]) : sum(b[1:i])
			idy = 1 + sum(b[1:j-1]) : sum(b[1:j])
			try
				if typeof(A[i,j].Full) <: LU
					B[idx,idy] = A[i,j].Full.factors
				else
					B[idx,idy] = A[i,j].Full
				end
			catch
				continue
			end
		end
	end

	return B
end

"""
	set_sparse_Block(b::Vector{Int}, rind::Vector{Int}, cind::Vector{Int}[, s_flag::Bool=false])::Block

Create a `Block` matrix.

# Arguments
- `b::Vector{Int}` : Vector that contains block sizes.
- `rind::Vector{Int}` : row indeces vector.
- `cind::Vector{Int}` : column indeces vector.
- `s_flag::Bool` : flag to make the matrix symetric (by block).
"""
function set_sparse_Block(b::Vector{Int}, rind::Vector{Int}, cind::Vector{Int}, s_flag::Bool=false)::Matrix
	npl = size(b,1)
	A::Matrix = Matrix(undef, npl, npl)
	idx = CartesianIndex.(rind,cind)
	for j in 1:length(idx)
		A[idx[j]] = Block(rand(Float64, b[idx[j][1]], b[idx[j][2]]))
		if s_flag && idx[j][1] != idx[j][2]
			A[idx[j][2],idx[j][1]] = Block(rand(Float64, b[idx[j][2]], b[idx[j][1]]))
		end
	end

	return A
end

"""
	set_sparse_Block(B::SparseArrays.SparseMatrixCSC, b::Vector{Int}[, s_flag::Bool=false])::Matrix

Create a `Matrix` matrix from a sparse matrix `B`.

# Arguments
- `B::Array` : sparse matrix.
- `b::Vector{Int}` : Vector that contains block sizes.
- `s_flag::Bool` : flag to make the matrix symetric (by block).
"""
function set_sparse_Block(B::SparseArrays.SparseMatrixCSC, b::Vector{Int}, s_flag::Bool=false)::Matrix
	npl = size(b,1)
	A::Matrix = Matrix(undef, npl, npl)

	for i in 1:npl
		for j in 1:npl
			idx = 1 + sum(b[1:i-1]) : sum(b[1:i])
			idy = 1 + sum(b[1:j-1]) : sum(b[1:j])
			if iszero(B[idx,idy])
				continue
			end
			A[i,j] = Block(B[idx,idy])
			if s_flag && i != j
				A[j,i] = Block(B[idy,idx])
			end
		end
	end

	return A
end

"""
	set_sparse_Block(B::Array, b::Vector{Int}[, s_flag::Bool=false])::Matrix

Create a `Matrix` matrix from a full matrix `B`.

# Arguments
- `B::Array` : full matrix.
- `b::Vector{Int}` : Vector that contains block sizes.
- `s_flag::Bool` : flag to make the matrix symetric (by block).
"""
function set_sparse_Block(B::Array, b::Vector{Int}, s_flag::Bool=false)::Matrix
	npl = size(b,1)
	A::Matrix = Matrix(undef, npl, npl)

	for i in 1:npl
		for j in 1:npl
			idx = 1 + sum(b[1:i-1]) : sum(b[1:i])
			idy = 1 + sum(b[1:j-1]) : sum(b[1:j])
			if iszero(B[idx,idy])
				continue
			end
			A[i,j] = Block(B[idx,idy])
			if s_flag && i != j
				A[j,i] = Block(B[idy,idx])
			end
		end
	end

	return A
end

"""
	blockMatrix_factorization!(A::Matrix)::Matrix

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
- `A::MAtrix` : The target matrix
"""
function blockMatrix_factorization(A::Matrix)::Matrix
	B = copy_bm(A)

	return blockMatrix_factorization!(B)
end

"""
	blockMatrix_inverse!(A::Matrix, fillin::Bool=false)::Matrix

Important : Right-looking version!!
"""
function blockMatrix_inverse!(A::Matrix, NoFillin::Bool=false)::Matrix
	npl = size(A,1)
	# Step 0 : Compute A(i,i)
	A[npl,npl].Full = A[npl,npl].Factors.U \ (A[npl,npl].Factors.L \ I)

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
				temp = prod_BM(A[j,comIdx], A[comIdx,i])
				Lupdated[j-i] = - temp / A[i,i].Factors.L
				# Step 2 : Compute A(i,i+1:npl)
				temp = prod_BM(A[i,comIdx], A[comIdx,j])
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
					A[j,i] = Block()
					A[i,j] = Block()
				end
				A[j,i].Full = Lupdated[j-i]
				A[i,j].Full = Uupdated[j-i]
			end
		end
		# Compute the rest part of A(i,i)
		A[i,i].Full = A[i,i].Factors.U \ (A[i,i].Factors.L \ I) - (A[i,i].Full / A[i,i].Factors.L)
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
	B = copy_bm(A)

	return blockMatrix_inverse!(B, NoFillin)
end
