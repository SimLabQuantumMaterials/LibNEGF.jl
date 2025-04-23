"""
	ArrayOrLU_ = Union{Array, LU, UndefInitializer}

Type for a matrix that can contain an `Array`, `LU factor` and/or `undef`.
"""
ArrayOrLU_ = Union{Array,LU,UndefInitializer}

# TODO : documentation
function be_copy_to_hw(M::Array)::Array
    # CPU -> CPU, return a shallow copy, which is
    # good enough in this case
    return copy(M)
end

# TODO : documentation
function be_copy_from_hw(M::Array)::Array
    # CPU -> CPU, return a shallow copy, which is
    # good enough in this case
    return copy(M)
end

# TODO : documentation
function be_copy_to_hw!(Mout::Array, Min::Array)
    copy!(Mout, Min)
end

# TODO : documentation
function be_copy_from_hw!(Mout::Array, Min::Array)
    copy!(Mout, Min)
end

# TODO : documentation
function be_copy_in_hw!(Mout::Array, Min::Array)
    copy!(Mout, Min)
end

# TODO : documentation
function be_inv(M::Array)::Array
    return inv(M)
end