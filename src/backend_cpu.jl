"""
	ArrayOrLU_ = Matrix{Union{Array, LU, Nothing}}

Type for a matrix that can contain an `Array`, `LU factor` and/or `undef`.
"""
ArrayOrLU_ = Matrix{Union{Array,LU,Nothing}}

# TODO : write tests for many of the following backend function, but these
#        tests have to be backend-blind

# TODO : add documentation for all of the following functions

# ----------------------------------------------------
# 'base' types first e.g. Array and Metal.MtlArray
# TODO : check : is Julia inlining these? Or use macros instead?

# done : all the tests have been added for this section

function be_copy_to_hw!(Mout::Array, Min::Array)
    copy!(Mout, Min)
end

function be_copy_to_hw(M::Array)::Array
    return Array(M)
end

function be_copy_from_hw!(Mout::Array, Min::Array)
    copy!(Mout, Min)
end

function be_copy_from_hw(M::Array)::Array
    return Array(M)
end

function be_copy_in_hw!(Mout::Array, Min::Array)
    copy!(Mout, Min)
end

function be_copy_in_hw(M::Array)::Array
    copy(M)
end

# function be_set_to_zero_dev_array!(M::Metal.MtlArray)
#     be_copy_to_hw!(M, zero(size(M)[1], size(M)[2]))
# end

function be_zero_array(nrsType::DataType, dimsOfArr::Tuple{Int, Int})::Array
    return Array(zeros(nrsType, dimsOfArr))
end













# ------------------------------------------------------------------

# # TODO : port all of the functions in backend_apple.jl to make sure
# #        the RGF implementation is blind to hardware, but also has good
# #        performance

# function be_copy_to_hw(M::Array)::Array
#     # CPU -> CPU, return a shallow copy, which is
#     # good enough in this case
#     return copy(M)
# end

# function be_copy_from_hw(M::Array)::Array
#     # CPU -> CPU, return a shallow copy, which is
#     # good enough in this case
#     return copy(M)
# end

# function be_copy_to_hw!(Mout::Array, Min::Array)
#     copy!(Mout, Min)
# end

# function be_copy_from_hw!(Mout::Array, Min::Array)
#     copy!(Mout, Min)
# end

# function be_copy_in_hw!(Mout::Array, Min::Array)
#     copy!(Mout, Min)
# end

# function be_copy_in_hw(M::Array)::Array
#     copy(M)
# end

# function be_copy_in_hw!(Mout::LU, Min::LU)
#     copy!(Mout, Min)
# end

# function be_copy_in_hw(M::LU)::LU
#     return copy(M)
# end

# function be_inv(M::Array)::Array
#     return inv(M)
# end

# function be_lu(M::Array)::LU
#     return lu(M)
# end