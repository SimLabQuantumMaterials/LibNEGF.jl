# TODO : documentation
struct RgfBuffs
    buffM::BlockMatrix
end

# # TODO : documentation
# struct LapackBuffs
#     pivB::Vector{Int}
# end

# TODO : documentation
# buffers for RGF, packed in a single struct
struct AuxDataDDRGF
    rgfBuffs::RgfBuffs
    # lapackBuffs::LapackBuffs
end

# TODO : documentation
# TODO(?) : put this inside a constructor for AuxDataDDRGF
function allocate_aux_data_DDRGF(M::BlockMatrix)::AuxDataDDRGF
    npl = size(M.blockSizes)[1]

    # 1. RGF-related buffers

    # in general, these type of auxiliary block matrices will contain
    # Array-like object and not LU-like, as specified by the last param
    buffM = BlockMatrix(M.blockSizes, ArrayOrLU_(undef, npl, npl),
        M.ndiag, M.nrsType, 1)
    set_blocks_to_zero!(buffM)

    # # 2. LAPACK-related buffers

    # # buffer vector used by getrf! when pivoting
    # pivB = Vector{Int}(undef, maximum(M.blockSizes))

    # the final struct with the buffers

    rgfBuffs = RgfBuffs(buffM)
    # lapackBuffs = RgfBuffs(pivB)
    # auxData = AuxDataDDRGF(rgfBuffs, lapackBuffs)
    auxData = AuxDataDDRGF(rgfBuffs)

    return auxData
end

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
function bndiag_of_inv_ddrgf(Min::BlockMatrix, auxData::AuxDataDDRGF)::BlockMatrix
    # TODO : extend this code to n-diagonal, otherwise rename this function
    #        to keep it as the simple traditional RGF

    # IMPORTANT : we assume here that all of the blocks in Min and Mout argument
    #             Array-like, and that those in auxData.rgfBuffs.buffM are LU-like

    # a copy of Min, where we place the output
    Mout = similar_bm_but_zero(Min)
    npl = size(Mout.blockSizes)[1]
    buffM1 = auxData.rgfBuffs.buffM
    # Mout is used as a buffer in multiple places, this is just
    # labeling for clarity of the implementation
    buffM2 = Mout

    # FIRST, upward pass

    # bottom element
    be_lu!(buffM1.M[npl, npl], Min.M[npl, npl])

    exit()

    # middle elements
    for ix = npl-1:-1:1
        be_mldivide!(buffM1.M[ix+1, ix], Min.M[ix+1, ix], buffM1.M[ix+1, ix+1])

        be_copy_in_hw!(buffM2.M[ix, ix], Min.M[ix, ix])

        be_gemm!(Min.nrsType, Min.nrsType, -1.0, Min.M[ix, ix+1], buffM1.M[ix+1, ix],
            1.0, buffM2.M[ix, ix])

        be_mrdivide!(buffM1.M[ix, ix+1], Min.M[ix, ix+1], buffM1.M[ix+1, ix+1])
        be_lu!(buffM1.M[ix, ix], buffM2.M[ix, ix])
    end

    # THEN, downward pass

    # top element
    be_inv_from_lu!(Mout.M[1, 1], buffM1.M[1, 1])

    # middle elements
    for ix = 2:npl
        # upper diagonal of Mout
        be_gemm!(Min.nrsType, Min.nrsType, 1.0, Mout.M[ix-1, ix-1], buffM1.M[ix-1, ix],
            Mout.M[ix-1, ix])

        # diagonal of Mout
        be_inv_from_lu!(Mout.M[ix, ix], buffM.M[ix, ix])
        be_gemm!(Min.nrsType, Min.nrsType, 1.0, buffM1.M[ix, ix-1], buffM2.M[ix-1, ix],
            1.0, Mout.M[ix, ix])

        # lower diagonal of Mout
        be_gemm!(Min.nrsType, Min.nrsType, -1.0, Mout.M[ix-1, ix-1], buffM1.M[ix, ix-1],
            Mout.M[ix, ix-1])
    end

    return Mout
end

# TODO(?) : do we want/need catch-all versions of the above function?