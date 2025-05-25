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

"""
Overload operators for Block
"""

"""
(==) operator
"""
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
(+) operator
"""
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

"""
copy overload
"""
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