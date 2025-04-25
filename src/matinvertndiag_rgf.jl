# """
#     bndiag_of_inv_rgf!(M::BlockMatrix)

# For an input matrix `M`, possibly but not necessarily block n-diagonal,
# where n is tri, penta, etc., compute the block n-diagonal part of the inverse
# of `M`. This function uses the RGF method. This is the in-place version.

# # Arguments
# - `M::BlockMatrix`: the matrix to be inverted.
# """
# function bndiag_of_inv_rgf!(M::BlockMatrix)

#     # TODO : fix everything in this function to implement first the basic
#     #        RGF method

#     # does it make sense to have an in-place of this? If so, then make
#     # use of be_copy_in_hw(...) before calling this
# end

"""
    bndiag_of_inv_rgf(M::BlockMatrix)::BlockMatrix

For an input matrix `M`, possibly but not necessarily block n-diagonal,
where n is tri, penta, etc., compute the block n-diagonal part of the inverse
of `M`. This function uses the RGF method.

# Arguments
- `M::BlockMatrix`: the matrix to be inverted.
"""
function bndiag_of_inv_rgf(M::BlockMatrix)::BlockMatrix

    # TODO : fix everything in this function to implement first the basic
    #        RGF method

    return M
end

# TODO(?) : do we want/need catch-all versions of the above function?