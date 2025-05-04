"""
    BlockMatrix

Struct containing an attribute with the block sizes, and another one with
the matrix stored in `ArrayOrLU_` format. The latter consists of a block matrix
where the blocks can contain a `Array`, `LU factor` and/or `undef`.
"""
struct BlockMatrix
    blockSizes::Vector{Int}
    M::ArrayOrLU_
    ndiag::Dict{String,Int}
    nrsType::DataType
    # 0 is Array-like, 1 is LU-like
    isArrayOrLU::Bool
end

# IMPORTANT : we work here under the assumption that the matrices of type
#             BlockMatrix live always in the wanted hardware (cpu, apple, etc)

# TODO(?) : create in-place versions of the following functions

"""
    convert_S2BM_ndiag(M::SparseArrays.SparseMatrixCSC, blockSizes::Vector{Int},
        ndiag::Dict{String,Int})

Convert the sparse input matrix `M` to the `BlockMatrix` type.

# Arguments
- `M::BlockMatrix`: the matrix to be converted.
- `blockSizes::Vector{Int}`: the sizes of the blocks along the block diagonal.
- `ndiag::Dict{String,Int}`: the number of block diagonals for the input
  and the output, e.g. Dict("in" => 3, "out" => 3) for block tri-diagonal.
"""
function convert_S2BM_ndiag(M::SparseArrays.SparseMatrixCSC, blockSizes::Vector{Int},
    ndiag::Dict{String,Int})::BlockMatrix
    # npl stands for number of principal layers
    npl = size(blockSizes)[1]
    # in general, these type of block matrices will contain Array-like object and not LU-like
    A = BlockMatrix(blockSizes, ArrayOrLU_(undef, npl, npl), ndiag, typeof(M[1, 1]), 0)

    # loop over the block sizes, conversely over the block rows
    for ix = 1:npl
        # indices for the rows
        ibeg = sum(blockSizes[1:ix-1]) + 1
        iend = sum(blockSizes[1:ix])
        # now, copy the blocks within the ix-th row
        if ix > 1
            # left
            for jx = (ix-1):-1:max(1, ix - Int((ndiag["out"] - 1) / 2))
                jbeg = sum(blockSizes[1:jx-1]) + 1
                jend = sum(blockSizes[1:jx])
                A.M[ix, jx] = be_copy_to_hw(Array(M[ibeg:iend, jbeg:jend]))
            end
        end
        # center
        jbeg = ibeg
        jend = iend
        A.M[ix, ix] = be_copy_to_hw(Array(M[ibeg:iend, jbeg:jend]))
        if ix < npl
            # right
            for jx = (ix+1):1:min(size(blockSizes)[1], ix + Int((ndiag["out"] - 1) / 2))
                jbeg = sum(blockSizes[1:jx-1]) + 1
                jend = sum(blockSizes[1:jx])
                A.M[ix, jx] = be_copy_to_hw(Array(M[ibeg:iend, jbeg:jend]))
            end
        end
    end

    return A
end

"""
	convert_BM2S_ndiag(M::BlockMatrix)

Convert the input matrix `M` of type `BlockMatrix` to sparse.

# Arguments
- `M::BlockMatrix`: the matrix to be converted.
"""
function convert_BM2S_ndiag(M::BlockMatrix)::SparseArrays.SparseMatrixCSC
    n = sum(M.blockSizes)
    ndiag = M.ndiag
    nrsType = M.nrsType

    # create the empty sparse matrix to be the output, with the
    # appropriate underlying data type in nrsType
    A = SparseArrays.SparseMatrixCSC{nrsType,Int}(undef, n, n)

    # loop over the block sizes, conversely over the block rows
    for ix = 1:size(M.blockSizes)[1]
        # indices for the rows
        ibeg = sum(M.blockSizes[1:ix-1]) + 1
        iend = sum(M.blockSizes[1:ix])
        # now, copy the blocks within the ix-th row
        if ix > 1
            # left
            for jx = (ix-1):-1:max(1, ix - Int((ndiag["out"] - 1) / 2))
                jbeg = sum(M.blockSizes[1:jx-1]) + 1
                jend = sum(M.blockSizes[1:jx])
                # MInvTrid[ibeg:iend, jbeg:jend] = MdenseInv[ibeg:iend, jbeg:jend]
                # A.M[ix, jx] = Array(M[ibeg:iend, jbeg:jend])
                A[ibeg:iend, jbeg:jend] = sparse(be_copy_from_hw(M.M[ix, jx]))
            end
        end
        # center
        jbeg = ibeg
        jend = iend
        A[ibeg:iend, jbeg:jend] = sparse(be_copy_from_hw(M.M[ix, ix]))
        if ix < size(M.blockSizes)[1]
            # right
            for jx = (ix+1):1:min(size(M.blockSizes)[1], ix + Int((ndiag["out"] - 1) / 2))
                jbeg = sum(M.blockSizes[1:jx-1]) + 1
                jend = sum(M.blockSizes[1:jx])
                A[ibeg:iend, jbeg:jend] = sparse(be_copy_from_hw(M.M[ix, jx]))
            end
        end
    end

    return A
end

# TODO : documentation
function copy_BM(M::BlockMatrix)::BlockMatrix
    ndiag = M.ndiag
    npl = size(M.blockSizes)[1]

    A = BlockMatrix(M.blockSizes, ArrayOrLU_(undef, npl, npl), ndiag, M.nrsType, M.isArrayOrLU)

    # loop over the block sizes, conversely over the block rows
    for ix = 1:size(M.blockSizes)[1]
        # now, copy the blocks within the ix-th row
        if ix > 1
            # left
            for jx = (ix-1):-1:max(1, ix - Int((ndiag["out"] - 1) / 2))
                A.M[ix, jx] = be_copy_in_hw(M.M[ix, jx])
            end
        end
        # center
        A.M[ix, ix] = be_copy_in_hw(M.M[ix, ix])
        if ix < size(M.blockSizes)[1]
            # right
            for jx = (ix+1):1:min(size(M.blockSizes)[1], ix + Int((ndiag["out"] - 1) / 2))
                A.M[ix, jx] = be_copy_in_hw(M.M[ix, jx])
            end
        end
    end

    return A

end

# TODO : documentation
# this returns an empty similar
function similar_bm(M::BlockMatrix)
    npl = size(M.blockSizes)[1]
    return BlockMatrix(M.blockSizes, ArrayOrLU_(undef, npl, npl), M.ndiag, M.nrsType, M.isArrayOrLU)
end

# TODO : documentation
# this returns a similar matrix but filled with zeroes
function similar_bm_but_zero(M::BlockMatrix)::BlockMatrix
    blockSizes = M.blockSizes
    ndiag = M.ndiag
    npl = size(blockSizes)[1]
    A = similar_bm(M)

    set_blocks_to_zero!(A)

    return A
end

# TODO : documentation
function set_blocks_to_zero!(M::BlockMatrix)
    blockSizes = M.blockSizes
    ndiag = M.ndiag
    npl = size(blockSizes)[1]
    # blocks will be set to 0-Array or 0-LU accordingly
    isArrayOrLU = M.isArrayOrLU
    # just a label of M
    A = M

    # loop over the block sizes, conversely over the block rows
    for ix = 1:npl
        # indices for the rows
        ibeg = sum(blockSizes[1:ix-1]) + 1
        iend = sum(blockSizes[1:ix])
        # now, copy the blocks within the ix-th row
        if ix > 1
            # left
            for jx = (ix-1):-1:max(1, ix - Int((ndiag["in"] - 1) / 2))
                jbeg = sum(blockSizes[1:jx-1]) + 1
                jend = sum(blockSizes[1:jx])
                A.M[ix, jx] = be_zero_array(A.nrsType, (iend - ibeg + 1, jend - jbeg + 1))
            end
        end
        # center
        jbeg = ibeg
        jend = iend
        if isArrayOrLU == 0
            A.M[ix, ix] = be_zero_array(A.nrsType, (iend - ibeg + 1, jend - jbeg + 1))
        else
            A.M[ix, ix] = be_zero_lu(A.nrsType, iend - ibeg + 1)
        end
        if ix < npl
            # right
            for jx = (ix+1):1:min(size(blockSizes)[1], ix + Int((ndiag["in"] - 1) / 2))
                jbeg = sum(blockSizes[1:jx-1]) + 1
                jend = sum(blockSizes[1:jx])
                A.M[ix, jx] = be_zero_array(A.nrsType, (iend - ibeg + 1, jend - jbeg + 1))
            end
        end
    end
end