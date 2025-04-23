"""
    ArrayOrLU

Struct containing an attribute with the block sizes, and another one with
the matrix stored in `ArrayOrLU_` format. The latter consists of a block matrix
where the blocks can contain a `Array`, `LU factor` and/or `undef`.
"""
struct ArrayOrLU
    blockSizes::Vector{Int}
    M::ArrayOrLU_
end

# TODO (?) : extend build_T_from_HS(...) to operate also with objects of type
#            ArrayOrLU, and in particular make this usable already in combination
#            with Metal.jl and create a corresponding test as well

"""
	convert_S2ALU_trid(M::SparseArrays.SparseMatrixCSC, blockSizes::Vector{Int})

Convert the sparse input matrix `M` to the `ArrayOrLU` type.

**WARNING :** Only valid for tri-diagonal matrices.

# Arguments
- `M::SparseArrays.SparseMatrixCSC`: the matrix to be converted.
- `blockSizes::Vector{Int}`: the sizes of the blocks along the block diagonal.
"""
function convert_S2ALU_trid(M::SparseArrays.SparseMatrixCSC, blockSizes::Vector{Int})::ArrayOrLU
    # npl stands for number of principal layers
    npl = size(blockSizes)[1]
    A = ArrayOrLU(blockSizes, Matrix(undef, npl, npl))

    # loop over the block sizes, conversely over the block rows
    for ix = 1:npl
        # indices for the rows
        ibeg = sum(blockSizes[1:ix-1]) + 1
        iend = sum(blockSizes[1:ix])
        # now, copy the blocks within the ix-th row
        if ix > 1
            # left
            jbeg = sum(blockSizes[1:ix-2]) + 1
            jend = sum(blockSizes[1:ix-1])
            A.M[ix, ix-1] = Array(M[ibeg:iend, jbeg:jend])
        end
        # center
        jbeg = ibeg
        jend = iend
        A.M[ix, ix] = Array(M[ibeg:iend, jbeg:jend])
        if ix < npl
            # right
            jbeg = sum(blockSizes[1:ix]) + 1
            jend = sum(blockSizes[1:ix+1])
            A.M[ix, ix+1] = Array(M[ibeg:iend, jbeg:jend])
        end
    end

    return A
end


"""
	convert_ALU2S_trid(M::ArrayOrLU)

Convert the input matrix `M` of type `ArrayOrLU` to sparse.

**WARNING :** Only valid for tri-diagonal matrices.

# Arguments
- `M::ArrayOrLU`: the matrix to be converted.
"""
function convert_ALU2S_trid(M::ArrayOrLU)::SparseArrays.SparseMatrixCSC
    n = sum(M.blockSizes)

    # extract the precision of the underlying data. We assume the element
    # M.M[1,1] to be different from undef
    if typeof(M.M[1, 1]) == LU
        nrsType = typeof(M.M[1, 1].L[1, 1])
    else
        nrsType = typeof(M.M[1, 1][1, 1])
    end

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
            jbeg = sum(M.blockSizes[1:ix-2]) + 1
            jend = sum(M.blockSizes[1:ix-1])
            A[ibeg:iend, jbeg:jend] = M.M[ix, ix-1]
        end
        # center
        jbeg = ibeg
        jend = iend
        A[ibeg:iend, jbeg:jend] = M.M[ix, ix]
        if ix < size(M.blockSizes)[1]
            # right
            jbeg = sum(M.blockSizes[1:ix]) + 1
            jend = sum(M.blockSizes[1:ix+1])
            A[ibeg:iend, jbeg:jend] = M.M[ix, ix+1]
        end
    end

    return A
end