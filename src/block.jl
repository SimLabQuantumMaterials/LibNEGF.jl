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

	@doc "Inner constructor"
	Block() =
	begin
		new(undef, undef, false, -1, -1)
	end
	Block(M) =
	begin
		new(M, undef, false, size(M,1), size(M,2))
	end
	Block(M::LU) =
	begin
		new(undef, M, false, size(M.L,1), size(M.L,2))
	end
	Block(row::Int, col::Int) =
	begin
		new(undef, undef, false, row, col)
	end
end

###
# Overload operators for Block
###

###
# (==) operator
###

"""
	Base.:(==)(A::Block, B::Array)::Bool

Overload the operator `==` to check if `Block` `A` and `Array` `B` are equal.

# Arguments
- `A::Block` : the first block for comparison.
- `B::Array` : the second block for comparison.
"""
function Base.:(==)(A::Block, B::Array)::Bool
	return A.Full == B
end

"""
	Base.:(==)(A::Array, B::Block)::Bool

Overload the operator `==` to check if `Array` `A` and `Block` `B` are equal.

# Arguments
- `A::Array` : the first block for comparison.
- `B::Block` : the second block for comparison.
"""
function Base.:(==)(A::Array, B::Block)::Bool
	return A == B.Full
end
	
"""
	Base.:(==)(A::Block, B::Block)::Bool

Overload the operator `==` to check if two `Block` `A` and `B` are equal.

# Arguments
- `A::Block` : the first block for comparison.
- `B::Block` : the second block for comparison.
"""
function Base.:(==)(A::Block, B::Block)::Bool
	return A.Full == B.Full && A.Factors == B.Factors && A.f_inv == B.f_inv && A.row == B.row && A.col == B.col
end

###
# (+) operator
###

"""
	+(A::Block, B::Block)::Array

Do `A+B` operation of their `Full` fields.
"""
function Base.:+(A::Block, B::Block)::Array
	return A.Full + B.Full
end

"""
	+(A::Array, B::Block)::Array

Do `A+B` operation of their `Full` fields.
"""
function Base.:+(A::Array, B::Block)::Array
	return A + B.Full
end

"""
	+(A::Block, B::Array)::Array

Do `A+B` operation of their `Full` fields.
"""
function Base.:+(A::Block, B::Array)::Array
	return A.Full + B
end

"""
	Base.prod(A::Block, B::Block)::Array

Do `A*B` product of their `Full` fields.
"""
function Base.prod(A::Block, B::Block)::Array
	return A.Full * B.Full
end

"""
	Base.prod(A::Array, B::Block)::Array

Do `A*B` product of their `Full` fields.
"""
function Base.prod(A::Array, B::Block)::Array
	return A * B.Full
end
	
"""
	Base.prod(A::Block, B::Array)::Array

Do `A*B` product of their `Full` fields.
"""
function Base.prod(A::Block, B::Array)::Array
	return A.Full * B
end

"""
	Base.:*(A::Block, B::Array)::Array

Do `A*B` product of their `Full` fields.
"""
function Base.:*(A::Block, B::Array)::Array
	return A.Full * B
end

"""
	Base.:*(A::Array, B::Block)::Array

Do `A*B` product of their `Full` fields.
"""
function Base.:*(A::Array, B::Block)::Array
	return A * B.Full
end

"""
	Base.:*(A::Block, B::Block)::Array

Do `A*B` product of their `Full` fields.
"""
function Base.:*(A::Block, B::Block)::Array
	return A.Full * B.Full
end

"""
	Base.:*(A::Number, B::Block)::Array

Do `A*B` product of their `Full` fields.
"""
function Base.:*(A::Number, B::Block)::Array
	return A * B.Full
end

"""
	Base.:*(A::Block, B::Number)::Array

Do `A*B` product of their `Full` fields.
"""
function Base.:*(A::Block, B::Number)::Array
	return A.Full * B
end

###
# copy overload
###

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
	bm_equal(A::Matrix, B::Matrix)::Bool

Check if two matrix `A` and `B` that contains `Block` object are equals or not.

# Arguments
- `A::Array` : the first matrix for comparison.
- `B::Array` : the secodn matrix for comparison.
"""
function bm_equal(A::Matrix, B::Matrix)::Bool
	sizeA = size(A)
	sizeB = size(B)
	if sizeA != sizeB
		return false
	end
	for i in 1:sizeA[1]
		for j in 1:sizeA[2]
			if !(isassigned(A,i,j) == isassigned(B,i,j))
				return false
			end
			if isassigned(A,i,j)
				if !(A[i,j] == B[i,j])
					return false
				end
			end
		end
	end
	return true
end

"""
	bm_copy(A::Matrix)::Matrix

Copy a matrix that contains `Block` inside.
"""
function bm_copy(A::Matrix)::Matrix
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
	bm_similar(A::Matrix)::Matrix

Copy a matrix that contains `Block` with undef fields.
"""
function bm_similar(A::Matrix)::Matrix
	return Matrix(undef, size(A,1), size(A,2))
end

"""
	sum_BlockMatrix(A::Array, B::Array)::Array

Do the addition of two `Matrix` that contains `Block` type.

# Arguments
- `A::Array` : A block matrix.
- `B::Array` : A block matrix.
"""
function sum_BlockMatrix(A::Array, B::Array)::Array
	@assert size(A)==size(B)
	C = Matrix(undef,size(A,1),size(A,2))
	for i in 1:size(A,1)
		for j in 1:size(A,2)
			if isassigned(A,i,j) && isassigned(B,i,j)
				@assert A[i,j].row==B[i,j].row && A[i,j].col==B[i,j].col
				C[i,j] = Block(A[i,j] + B[i,j])
			elseif isassigned(A,i,j)
				C[i,j] = copy(A[i,j])
			elseif isassigned(B,i,j)
				C[i,j] = copy(B[i,j])
			end
		end
	end
	return C
end

"""
	prod_BlockMatrix(A::Matrix, B::Matrix)::Matrix

Do the product of two matrix that contains `Block`, `A*B`.

# Arguments
- `A::Block` : the matrix on the left side.
- `B::Block` : the matrix on the right side.
"""
function prod_BlockMatrix(A::Array, B::Array)::Array
	C = Matrix(undef,size(A,1),size(B,2))
	for i in 1:size(A,1)
		for j in 1:size(A,2)
			for k in 1:size(B,2)
				if isassigned(A,i,j) && isassigned(B,j,k)
					if !isassigned(C,i,k)
						C[i,k] = Block(zeros(A[i,j].row,B[j,k].col))
					end
					C[i,k].Full += prod(A[i,j], B[j,k])
				end
			end
		end
	end
	return C
end

###
# Get elements of a `Block` matrix.
###
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
	get_rcIndexAt(M::Matrix[, rowB::Int=1, rowE::Int=size(M,1), colB::Int=1, colE::Int=size(M,2)])::Tuple{Vector{Int}, Vector{Int}}

Get the rows and columns index where the matrix `M` is not `undef` for specific index.

# Arguments
- `M::Matrix` : the matrix we want to analyze.
- `rowB::Int` : the starting index of row we analyze.
- `rowE::Int` : the ending index of row we analyze.
- `colB::Int` : the starting index of column we analyze.
- `colE::Int` : the ending index of column we analyze.
"""
function get_rcIndexAt(M::Matrix, rowB::Int=1, rowE::Int=size(M,1), colB::Int=1, colE::Int=size(M,2))::Tuple{Vector{Int}, Vector{Int}}

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
	get_rowSizes(A::Matrix[, npl::Int=size(A,1)])::Vector{Int}

Get the row sizes of each block along the diagonal from matrix `A`.

# Arguments
- `A::Matrix` : the target block matrix.
- `npl::Int` : the number of block along the diagonal.
"""
function get_rowSizes(A::Matrix, npl::Int=size(A,1))::Vector{Int}
	r = Vector{Int}(undef, npl)
	for i = 1:npl
		r[i] = A[i,i].row
	end
	return r
end

"""
	get_colSizes(A::Matrix[, npl::Int=size(A,1)])::Vector{Int}

Get the column sizes of each block along the diagonal from matrix `A`.

# Arguments
- `A::Matrix` : the target block matrix.
- `npl::Int` : the number of block along the diagonal.
"""
function get_colSizes(A::Matrix, npl::Int=size(A,1))::Vector{Int}
	c = Vector{Int}(undef, npl)
	for i = 1:npl
		c[i] = A[i,i].col
	end
	return c
end

"""
	get_blockSizes(A::Matrix[, npl::Int=size(A,1)])::Vector{Int}

Get the size of each block along the diagonal from matrix `A`.

# Arguments
- `A::Matrix` : the target block matrix.
- `npl::Int` : the number of block along the diagonal.
"""
function get_blockSizes(A::Matrix, npl::Int=size(A,1))::Tuple{Vector{Int}, Vector{Int}}
	r = Vector{Int}(undef, npl)
	c = Vector{Int}(undef, npl)
	for i = 1:npl
		r[i] = A[i,i].row
		c[i] = A[i,i].col
	end
	return r,c
end

###
# Generate elements
###

"""
	full(A::Matrix, rind::Vector{Int}, cind::Vector{Int})::Array

Convert a block matrix of type `Matrix` to full matrix.

**Warning : Working only if the diagonal is no empty.**

# Arguments
- `A::Matrix` : A block matrix.
- `rind::Vector{Int}` : row indeces vector.
- `cind::Vector{Int}` : column indeces vector.
"""
function full(A::Matrix, rind::Vector{Int}, cind::Vector{Int})::Array
	b = get_rowSizes(A)
	m = sum(b)
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
	full(A::Matrix)::Array

Convert a block matrix of type `Matrix` to full matrix.

**Warning : Working only if the diagonal is no empty.**
	
# Arguments
- `A::Matrix`: A block matrix.
"""
function full(A::Matrix)::Array
	npl = size(A,1)
	b = get_rowSizes(A)
	m = sum(b)
	prec = typeof(A[1,1].Full[1,1])
	B = zeros(prec,m,m)

	for i = 1:npl
		for j = 1:npl
			idx = 1 + sum(b[1:i-1]) : sum(b[1:i])
			idy = 1 + sum(b[1:j-1]) : sum(b[1:j])
			if isassigned(A,i,j)
				if typeof(A[i,j].Factors) <: LU
					B[idx,idy] = A[i,j].Factors.factors
				else
					B[idx,idy] = A[i,j].Full
				end
			end
		end
	end

	return B
end

"""
	set_sparse_Block(b::Vector{Int}, rind::Vector{Int}, cind::Vector{Int}[, s_flag::Bool=false])::Block

Create a matrix that contains `Block` (only work for square matrix).

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
			if iszero(Matrix(B[idx,idy]))
				continue
			end
			A[i,j] = Block(Matrix(B[idx,idy]))
			if s_flag && i != j
				A[j,i] = Block(Matrix(B[idy,idx]))
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
	set_sparse_Block(B::Array, npl::Int[, s_flag::Bool=false])::Matrix

Create a `Matrix` matrix from a full matrix `B`.

# Arguments
- `B::Array` : full matrix.
- `npl::Int` : Number of principal layer.
- `s_flag::Bool` : flag to make the matrix symetric (by block).
"""
function set_sparse_Block(B::Array, npl::Int, s_flag::Bool=false)::Matrix
	b = Vector{Int}(div(size(B,1),npl), npl)
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
	block_create_synthetic_random(nrLayers::Int, blocksDim::Int, nrsType::DataType)

Create a random synthetic block tridiagonal matrix.

# Arguments
- `nrLayers::Int`: the number of principal layers.
- `blocksDim::Int`: the average size of the principal layers (i.e., blocks).
- `nrsType::DataType`: the type of the underlying data.
"""
function block_create_synthetic_random(nrLayers::Int, blocksDim::Int, nrsType::DataType, isHermitian::Bool)::Array
    # IMPORTANT : this function assumes that all of the principal layers are of
    #             the same size
	bind2 = repeat([blocksDim], nrLayers)

	# row index
    rind2 = []
    for i in 2:nrLayers-1
        rind2 = cat(rind2, repeat([i], 3); dims=1)
    end
    rind2 = cat([1, 1], rind2, [nrLayers, nrLayers]; dims=1)
    
	# col index
    cind2 = []
    for i in 2:nrLayers-1
        cind2 = cat(cind2, [i - 1, i, i + 1]; dims=1)
    end
    cind2 = cat([1, 2], cind2, [nrLayers - 1, nrLayers]; dims=1)

    # Original matrix
	A::Matrix = Matrix(undef, nrLayers, nrLayers)
	idx = CartesianIndex.(rind2,cind2)
	for j in 1:length(idx)
		A[idx[j]] = Block(convert(nrsType, 0.3) * be_random_array(nrsType, (bind2[idx[j][1]], bind2[idx[j][2]])))
		if isHermitian && idx[j][1] != idx[j][2]
			A[idx[j][2],idx[j][1]] = Block(convert(nrsType, 0.3) * be_random_array(nrsType, (bind2[idx[j][2]], bind2[idx[j][1]])))
		end
	end

    return A
end

"""
	show_sparse(A::Array[, tol::Float64=1e-18])

Print in IO the sparsity of the matrix `A`.

Dot represent nonzeros values.

# Arguments
- `A::Array`: a matrix.
- `tol::Float64`: the threshold to show values.
"""
function show_sparse(A::Array, tol::Float64=1e-18)
	println(SparseArrays._show_with_braille_patterns(stdout, droptol!(sparse(A),tol)));
end