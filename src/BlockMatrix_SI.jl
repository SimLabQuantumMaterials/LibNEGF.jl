### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ cf347746-1f87-11f0-175c-13d95d87cda3
begin
	# Fill in Alphabetic order
	import PlutoUI: combine

	using BenchmarkTools
	using BlockArrays
	using Dagger
	using Distributions, Random
	using LinearAlgebra
	using MAT
	using PlutoUI
	using SparseArrays
	using Test
end

# ╔═╡ ee4cd4ee-ec4c-48e1-be47-ec68a4a3a5cc
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
	Block(row, col) =
	begin
		new(undef, undef, false, row, col)
	end
end

# ╔═╡ 5d4a35e6-8736-48f0-9c44-6161b4522432
begin
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
end

# ╔═╡ 2077700a-c883-4857-b5b4-1543051aba43
# Test part
begin
@testset "operator(==)::Block" begin
	M = Matrix(undef,5,5)
	M[1,1] = Block(rand(5,5))
	M[2,3] = Block()
	N = M
	@test N[1,1] == M[1,1]
	@test N[2,3] == M[2,3]
end

@testset "bm_equal" begin
	M = Matrix(undef,5,5)
	N = Matrix(undef,4,5)
	@test bm_equal(M,N) broken = true
	N = Matrix(undef,5,5)
	N[4,4] = Block()
	@test bm_equal(M,N) broken = true
	N = M
	@test bm_equal(M,N)
	M[1,1] = Block(rand(5,5))
	M[2,3] = Block()
	N = M
	@test bm_equal(M,N)
end
end

# ╔═╡ d4e84dcb-f5a4-415c-8e52-670bdcd4d497
begin
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
end

# ╔═╡ 33c7922b-82c0-4aa7-9f51-f438aa6650fb
# Test part
@testset "get_rcIndex" begin
	M = Matrix(undef,5,5)
	idx, idy = get_rcIndex(M)
	@test isempty(idx)
	@test isempty(idy)
	idx = [1,2,3]
	idy = [1,3]
	M[idx, idy] .= 1
	@test get_rcIndex(M) == ([1, 1, 2, 2, 3, 3], [1, 3, 1, 3, 1, 3])
end

# ╔═╡ 13bf0e4e-f39e-4436-8b8d-d6ceb66e0892
begin
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

end

# ╔═╡ 49a8a53f-43be-4154-ae58-f9876ead5933
# Test part
@testset "get_rcIndexAt" begin
	M = Matrix(undef,5,5)
	idx, idy = get_rcIndexAt(M)
	@test isempty(idx)
	@test isempty(idy)
	idx = [1,2,3]
	idy = [1,3]
	M[idx, idy] .= 1
	@test get_rcIndexAt(M, 1, 1) == ([1, 1, 2, 2, 3, 3], [1, 3, 1, 3, 1, 3]) broken=true
	@test get_rcIndexAt(M, 1, 1) == ([1, 1], [1, 3])
end

# ╔═╡ dc4ee2ae-9658-4511-958f-0a3f0feae526
begin
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
end

# ╔═╡ 3fe437dc-742e-49eb-adab-3f24bf7934da
# Test part
@testset "copy_bm" begin
	M = Matrix(undef,5,5)
	M[1,1] = Block(rand(5,5))
	M[2,3] = Block()
	N = copy_bm(M)
	@test bm_equal(N, M)
end

# ╔═╡ 88f2413d-737b-4a1c-a33f-9de6fb20e432
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

# ╔═╡ 0c6e0a18-0c0d-4c0d-9f20-253aff91d318
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

# ╔═╡ 1bb77e02-84ae-43e5-8ac3-0ebb98be217a
# Test part
@testset "get_blockSizes" begin
	npl = 5
	M = Matrix(undef,npl,npl)
	verif = []
	for i in 1:5
		M[i,i] = Block(rand(i,2))
		push!(verif,i)
	end
	bl = get_blockSizes(M)
	@test bl == verif
	# Think about a critical case
	@test bl == 0 broken=true
end

# ╔═╡ 093c361f-7a5a-4f3a-a35e-917ba3f047b1
begin
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
	
md"""
### Addition Part

This hidden cell is the `+` operation for `Block` object.
"""
end

# ╔═╡ c766a6eb-8b69-4365-8814-30fc1c34196a
begin
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
	
"""
	prod_BM(A::Matrix, B::Matrix)::Matrix

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
end

# ╔═╡ 1c11ef7a-6ac2-441c-b979-85d71a34035e
"""
	find_sparse(A::AbstractSparseArray[, percent::Float64=0.01, tol::Float64=1e-16])

Find the tolerance `tol` where the sparsity reach `percent` threshold.

# Arguments
- `A::AbstractSparseArray`: a matrix.
- `percent::Float64` : the percent of spartify we want at the end. 
- `tol::Float64`: the starting point for reach the `percent`.
"""
function find_sparse(A::SparseArrays.SparseMatrixCSC, percent::Float64=0.01, tol::Float64=1e-16)
	while count(!iszero, droptol!(A,tol)) / prod(size(A)) > percent
		tol *= 10
	end
	return tol
end

# ╔═╡ 3ea60ccc-02f2-4067-bd35-803441b5439c
# Test part
@testset "prod_BlockMatrix" begin
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
	res = prod_BlockMatrix(M,N)
	@test res[2,1] == a*c
	@test res[3,2] == b*d
	@test res[1,2] == (e*g+f*d)
	@test res[1,1] == e*g+f*d broken=true
end

# ╔═╡ bc7d0bb8-7214-434d-848f-333b5ca8012b
# Test part
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

# ╔═╡ df08f8fd-59c5-4f02-8d9d-00d0deef8a8e
begin
"""
	full(A::Matrix, rind::Vector{Int}, cind::Vector{Int})::Array

Convert a block matrix of type `Matrix` to full matrix.

# Arguments
- `A::Matrix` : A block matrix.
- `rind::Vector{Int}` : row indeces vector.
- `cind::Vector{Int}` : column indeces vector.
"""
function full(A::Matrix, rind::Vector{Int}, cind::Vector{Int})::Array
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
	full(A::Matrix)::Array

Convert a block matrix of type `Matrix` to full matrix.

# Arguments
- `A::Matrix`: A block matrix.
"""
function full(A::Matrix)::Array
	npl = size(A,1)
	b = get_blockSizes(A)
	m = sum(b)
	# B = zeros(typeof(A[1,1][1,1]),m,m)
	B = zeros(Float64,m,m)

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
end

# ╔═╡ 5ca65fad-c2dd-4a53-aaa8-3505320333de
# Test part
@testset "full" begin
	#TODO
end

# ╔═╡ 9ff50658-b348-4d66-a7e0-3ce49a5bf68b
begin
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
end

# ╔═╡ 7c2f3af3-fcf9-4257-94dd-54ce9cfe824a
# Test part
@testset "set_sparse_Block" begin
	#TODO
end

# ╔═╡ 511d9b15-31b1-42df-a212-12defec055ca
begin
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
end

# ╔═╡ 5be974bc-a29a-4e41-ad5c-35c127b81e6f
# Test part
@testset "convert_M2Block" begin
	#TODO
end

# ╔═╡ 040aaccf-2252-4538-93b5-e702b15d0445
begin
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
end

# ╔═╡ 64ab059b-e320-474b-a7ba-f45e59738c30
# Test part
@testset "blockMatrix_factorization" begin
	#TODO
end

# ╔═╡ ef2f9036-1eaa-4eef-854a-71e9f8ab15e8
begin
"""
	blockMatrix_inverse!(A::Matrix, fillin::Bool=false)::Matrix

Important : Right-looking version!!
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
	B = copy_bm(A)

	return blockMatrix_inverse!(B, NoFillin)
end
end

# ╔═╡ e39b8f10-7e02-471e-81bf-e2665ddb7787
# Test part
@testset "blockMatrix_inverse" begin
	#TODO
end

# ╔═╡ addbfada-c797-4aa2-8cf1-9b206ade7562
begin
	rind = [1,1,1,2,3,3,3,4,5]
	cind = [1,3,5,2,3,4,5,4,5]
	val = [1,1,1,2,3,3,3,4,5]
	bind = [500,400,200,300,100]
	sparse_percent = 0.1
	T_small = set_sparse_Block(bind, rind, cind, true)
	n = sum(bind)
	for i in 1:5
		T_small[i,i].Full += n*I(bind[i])
	end
	## Revoir l'idée car on est en Symmetrie ACTIVE!!
end

# ╔═╡ 354813cb-7c70-4547-af80-f87d8a003dbf
begin
	T_small_Matrix = full(T_small)
	realInv = inv(T_small_Matrix)

	T_LU = blockMatrix_factorization(T_small)
	T_LU_Matrix = full(T_LU)
	appT_LU = UnitLowerTriangular(T_LU_Matrix) * UpperTriangular(T_LU_Matrix)

	T_app = blockMatrix_inverse(T_LU)
	T_app_Matrix = full(T_app)
	I_app = T_small_Matrix * T_app_Matrix

	T_app_reduce = blockMatrix_inverse(T_LU, true)
	T_app_reduce_Matrix = full(T_app_reduce)
	I_app_reduce = T_small_Matrix * T_app_reduce_Matrix
	
	nothing
end

# ╔═╡ d7a092fd-688f-44ec-ab66-44ba0c4d7673
# begin
# 	println("## Show the matrix shape ##")
# 	show_sparse(T_small_Matrix)
# 	println("\n## Show the factorization shape ##")
# 	show_sparse(T_LU_Matrix)
# end

# ╔═╡ 2a160b98-5358-4a90-bd21-e411f987efb3
# begin
# 	println("## Show the approximation of the factorization ##")
# 	show_sparse(appT_LU - T_small_Matrix, 1e-13)
# end

# ╔═╡ c2b7f36f-cb70-4473-b6c8-df7584b7eca1
# begin
# 	println("## Show the real inverse shape ##")
# 	show_sparse(realInv)
# 	println("\n## Show the approximation of our inverse ##")
# 	show_sparse(T_app_Matrix)
# 	println("\n## Show the REDUCE approximation of our inverse ##")
# 	show_sparse(T_app_reduce_Matrix)
# end

# ╔═╡ fcd02301-db8f-4548-a9dc-0376b2282f50
# begin
# 	println("## Show the difference between real inverse and our inverse ##")
# 	local sparse_tol = find_sparse(sparse(T_app_Matrix-realInv))
# 	println("\n## Find sparsity of : ", sparse_percent, " at tolerance : " , sparse_tol, " ##")
# 	show_sparse(T_app_Matrix - realInv,sparse_tol/10)
# 	println()
# 	show_sparse(T_app_Matrix - realInv,sparse_tol)
# end

# ╔═╡ c0f3cc62-ff57-4f50-9773-82d65980c015
# begin
# 	println("## Show the difference between real inverse and our REDUCE inverse ##")
# 	local sparse_tol = find_sparse(sparse(T_app_reduce_Matrix - realInv))
# 	show_sparse(T_app_reduce_Matrix - realInv, 1e-10)
# 	println()
# 	println("\n## Find sparsity of : ", sparse_percent, " at tolerance : " , sparse_tol, " ##")
# 	show_sparse(T_app_reduce_Matrix - realInv, sparse_tol/10)
# 	println()
# 	show_sparse(T_app_reduce_Matrix - realInv, sparse_tol)
# end

# ╔═╡ 9e466309-8d05-4f55-b58a-50ed96641fe2
# ╠═╡ disabled = true
#=╠═╡
begin
	println("\n## Show the product of our approximation with original matrix shape ##")
	local sparse_percent = 0.1
	local sparse_tol = find_sparse(sparse(I_app), sparse_percent)
	println("\n## Find sparsity of : ", sparse_percent, " at tolerance : " , sparse_tol, " ##")
	show_sparse(I_app, sparse_tol/10)
	println()
	show_sparse(I_app, sparse_tol)
	println("\n## Show the product of our REDUCE approximation with original matrix shape ##")
	local sparse_tol = find_sparse(sparse(I_app_reduce), sparse_percent)
	println("\n## Find sparsity of : ", sparse_percent, " at tolerance : " , sparse_tol, " ##")
	show_sparse(I_app_reduce, sparse_tol/10)
	println()
	show_sparse(I_app_reduce, sparse_tol)
end
  ╠═╡ =#

# ╔═╡ de980275-7865-4f45-954e-6638466d9c6e
begin
	invtemp = UpperTriangular(T_LU_Matrix)\(UnitLowerTriangular(T_LU_Matrix) \I)
	errNoOpti = norm(T_small_Matrix*invtemp - I) / norm(I_app)
	println("LU Approach error : ||T*Gr-I|| / ||T*Gr|| = ", errNoOpti)
	errNoOpti = norm(I_app - I) / norm(I_app)
	println("Approach error : ||T*Gr-I|| / ||T*Gr|| = ", errNoOpti)
	errNoOpti = norm(T_small_Matrix*realInv - I) / norm(T_small_Matrix*realInv)
	println("Julia approach error : ||T*Gr-I|| / ||T*Gr|| = ", errNoOpti)
	errOpti = norm(I_app_reduce - I)/ norm(I_app_reduce)
	println("REDUCE approach error : ||T*Gr-I|| / ||T*Gr|| = ", errOpti)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
Dagger = "d58978e5-989f-55fb-8d15-ea34adc7bf54"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MAT = "23992714-dd62-5051-b70f-ba57cb901cac"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
BenchmarkTools = "~1.6.0"
BlockArrays = "~1.3.0"
Dagger = "~0.18.14"
Distributions = "~0.25.117"
MAT = "~0.10.7"
PlutoUI = "~0.7.62"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "7655dbeed6f542edf71a6b84198c72ced9a8f27d"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra"]
git-tree-sha1 = "4e25216b8fea1908a0ce0f5d87368587899f75be"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "e38fbc49a620f5d0b660d7f543db1009fe0f8336"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.0"

[[deps.BlockArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "b406207917260364a2e0287b42e4c6772cb9db88"
uuid = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
version = "1.3.0"

    [deps.BlockArrays.extensions]
    BlockArraysBandedMatricesExt = "BandedMatrices"

    [deps.BlockArrays.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"

[[deps.BufferedStreams]]
git-tree-sha1 = "6863c5b7fc997eadcabdbaf6c5f201dc30032643"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.2.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentCollections]]
git-tree-sha1 = "875d52bb9900dcf9e4e6bbd02da0e590eb200dc8"
uuid = "5060bff5-0b44-40c5-b522-fcd3ca5cecdd"
version = "0.1.1"

[[deps.Dagger]]
deps = ["Adapt", "DataStructures", "Distributed", "DistributedNext", "Graphs", "LinearAlgebra", "MacroTools", "MemPool", "OnlineStats", "PrecompileTools", "Preferences", "Profile", "Random", "Requires", "ScopedValues", "Serialization", "SharedArrays", "SparseArrays", "Statistics", "StatsBase", "TaskLocalValues", "TimespanLogging", "UUIDs"]
git-tree-sha1 = "8193108995b57fda5bf8b19310ea13ad33a00973"
uuid = "d58978e5-989f-55fb-8d15-ea34adc7bf54"
version = "0.18.14"

    [deps.Dagger.extensions]
    DistributionsExt = "Distributions"
    GraphVizExt = "GraphViz"
    GraphVizSimpleExt = "Colors"
    JSON3Ext = "JSON3"
    PlotsExt = ["DataFrames", "Plots"]
    PythonExt = "PythonCall"

    [deps.Dagger.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    GraphViz = "f526b714-d49f-11e8-06ff-31ed36ee7ee0"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
    Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DistributedNext]]
deps = ["Random", "Serialization", "Sockets"]
git-tree-sha1 = "c0b288d17e3f037404b8dd102f87163db8ce45f1"
uuid = "fab6aee4-877b-4bac-a744-3eca44acbb6f"
version = "1.0.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "03aa5d44647eaec98e1920635cdfed5d5560a8b9"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.117"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "1dc470db8b1131cfc7fb4c115de89fe391b9e780"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.12.0"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "MPIPreferences", "Mmap", "Preferences", "Printf", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "e856eef26cf5bf2b0f95f8f4fc37553c72c8641c"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.17.2"

    [deps.HDF5.extensions]
    MPIExt = "MPI"

    [deps.HDF5.weakdeps]
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "e94f84da9af7ce9c6be049e9067e511e17ff89ec"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.6+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f93a9ce66cd89c9ba7a4695a47fd93b4c6bc59fa"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.12.0+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "2bd56245074fab4015b9174f24ceba8293209053"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.27"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "1d2dd9b186742b0f317f2530ddcbf00eebb18e96"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.7"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "3aa3210044138a1749dbd350a9ba8680869eb503"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.3.0+1"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "c105fe467859e7f6e9a852cb15cb4301126fac07"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.11"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "ff91ca13c7c472cef700f301c8d752bc2aaff1a8"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.3+0"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.MemPool]]
deps = ["ConcurrentCollections", "DataStructures", "Distributed", "DistributedNext", "Mmap", "Preferences", "Random", "ScopedValues", "Serialization", "Sockets"]
git-tree-sha1 = "798176e74ab423b9c1b0e9de6652e1b4bc4ee821"
uuid = "f9f48841-c794-520a-933b-121f7ba6ed94"
version = "0.4.12"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bc95bf4149bf535c09602e3acdf950d9b4376227"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OnlineStats]]
deps = ["AbstractTrees", "Dates", "Distributions", "LinearAlgebra", "OnlineStatsBase", "OrderedCollections", "Random", "RecipesBase", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "8437102a36046b73a50af12430ec3e8e98768d80"
uuid = "a15396b6-48d5-5d58-9928-6d29437db91e"
version = "1.7.1"

[[deps.OnlineStatsBase]]
deps = ["AbstractTrees", "Dates", "LinearAlgebra", "OrderedCollections", "Statistics", "StatsBase"]
git-tree-sha1 = "a5a5a68d079ce531b0220e99789e0c1c8c5ed215"
uuid = "925886fa-5bf2-5e8e-b522-a9147a512338"
version = "1.7.1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML", "Zlib_jll"]
git-tree-sha1 = "da913f03f17b449951e0461da960229d4a3d1a8c"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.7+1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a9697f1d06cc3eb3fb3ad49cc67f2cfabaac31ea"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.16+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "966b85253e959ea89c53a9abebbf2e964fbf593b"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.32"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "d3de2694b52a01ce61a036f18ea9c0f61c4a9230"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.62"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "1147f140b4c8ddab224c94efa9569fc23d63ab44"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "02c8bd479d26dbeff8a7eb1d77edfc10dacabc01"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.11"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TaskLocalValues]]
git-tree-sha1 = "d155450e6dff2a8bc2fcb81dcb194bd98b0aeb46"
uuid = "ed4db957-447d-4319-bfb6-7fa9ae7ecf34"
version = "0.1.2"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TimespanLogging]]
deps = ["Distributed", "Profile"]
git-tree-sha1 = "51be7dd35b0c8a5a613dc7af272d587ea6943d24"
uuid = "a526e669-04d3-4846-9525-c66122c55f63"
version = "0.1.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f5733a5a9047722470b95a81e1b172383971105c"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.3+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═cf347746-1f87-11f0-175c-13d95d87cda3
# ╠═ee4cd4ee-ec4c-48e1-be47-ec68a4a3a5cc
# ╠═5d4a35e6-8736-48f0-9c44-6161b4522432
# ╟─2077700a-c883-4857-b5b4-1543051aba43
# ╟─d4e84dcb-f5a4-415c-8e52-670bdcd4d497
# ╟─33c7922b-82c0-4aa7-9f51-f438aa6650fb
# ╟─13bf0e4e-f39e-4436-8b8d-d6ceb66e0892
# ╟─49a8a53f-43be-4154-ae58-f9876ead5933
# ╟─dc4ee2ae-9658-4511-958f-0a3f0feae526
# ╟─3fe437dc-742e-49eb-adab-3f24bf7934da
# ╟─88f2413d-737b-4a1c-a33f-9de6fb20e432
# ╟─1c11ef7a-6ac2-441c-b979-85d71a34035e
# ╟─0c6e0a18-0c0d-4c0d-9f20-253aff91d318
# ╟─1bb77e02-84ae-43e5-8ac3-0ebb98be217a
# ╟─c766a6eb-8b69-4365-8814-30fc1c34196a
# ╟─3ea60ccc-02f2-4067-bd35-803441b5439c
# ╠═093c361f-7a5a-4f3a-a35e-917ba3f047b1
# ╟─bc7d0bb8-7214-434d-848f-333b5ca8012b
# ╠═df08f8fd-59c5-4f02-8d9d-00d0deef8a8e
# ╟─5ca65fad-c2dd-4a53-aaa8-3505320333de
# ╟─9ff50658-b348-4d66-a7e0-3ce49a5bf68b
# ╟─7c2f3af3-fcf9-4257-94dd-54ce9cfe824a
# ╟─511d9b15-31b1-42df-a212-12defec055ca
# ╟─5be974bc-a29a-4e41-ad5c-35c127b81e6f
# ╠═040aaccf-2252-4538-93b5-e702b15d0445
# ╟─64ab059b-e320-474b-a7ba-f45e59738c30
# ╠═ef2f9036-1eaa-4eef-854a-71e9f8ab15e8
# ╟─e39b8f10-7e02-471e-81bf-e2665ddb7787
# ╠═addbfada-c797-4aa2-8cf1-9b206ade7562
# ╠═354813cb-7c70-4547-af80-f87d8a003dbf
# ╠═d7a092fd-688f-44ec-ab66-44ba0c4d7673
# ╠═2a160b98-5358-4a90-bd21-e411f987efb3
# ╠═c2b7f36f-cb70-4473-b6c8-df7584b7eca1
# ╠═fcd02301-db8f-4548-a9dc-0376b2282f50
# ╠═c0f3cc62-ff57-4f50-9773-82d65980c015
# ╠═9e466309-8d05-4f55-b58a-50ed96641fe2
# ╠═de980275-7865-4f45-954e-6638466d9c6e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
