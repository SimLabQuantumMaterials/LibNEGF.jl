"""
    BlockMatrix

Encapsulates the data for a block n-diagonal matrix. The actual matrix is
stored in `M`, with underlying scalars type specified by `nrsType`. The
blocks along the diagonal can be either an `Array`-like or an `LU`-like struct,
which is indicated via `isArrayOrLU`. The number of offdiagonals is stored
in the dictionary `ndiag`, e.g. Dict("in" => 3, "out" => 3) tells us that an
algorithm will read tye block 3-diagonal of `M` only, and return a block 3-diagonal.
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

"""
    bm_convert(M::SparseArrays.SparseMatrixCSC, blockSizes::Vector{Int},
        ndiag::Dict{String,Int})

Converts the sparse input matrix `M` to the `BlockMatrix` type.

# Arguments
- `M::BlockMatrix`: the matrix to be converted.
- `blockSizes::Vector{Int}`: the sizes of the blocks along the block diagonal.
- `ndiag::Dict{String,Int}`: the number of block diagonals for the input
  and the output, e.g. Dict("in" => 3, "out" => 3) for block tri-diagonal.
"""
function bm_convert(M::SparseArrays.SparseMatrixCSC, blockSizes::Vector{Int},
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
	bm_convert(M::BlockMatrix)

Converts the input matrix `M` of type `BlockMatrix` to sparse.

# Arguments
- `M::BlockMatrix`: the matrix to be converted.
"""
function bm_convert(M::BlockMatrix)::SparseArrays.SparseMatrixCSC
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

"""
	copy_BM(M::BlockMatrix)

Receives a BlockMatrix object and returns a deep copy of it.

# Arguments
- `M::BlockMatrix`: the matrix to be copied.
"""
function bm_copy(M::BlockMatrix)::BlockMatrix
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

"""
	bm_copy!(Mout::BlockMatrix, Min::BlockMatrix)

In place copy of BlockMatrix to BlockMatrix.

# Arguments
- `Mout::BlockMatrix`: the copy of `Min``.
- `Min::BlockMatrix`: the matrix to be copied.
"""
function bm_copy!(Mout::BlockMatrix, Min::BlockMatrix)
    ndiag = Min.ndiag
    npl = size(Min.blockSizes)[1]

    # loop over the block sizes, conversely over the block rows
    for ix = 1:size(Min.blockSizes)[1]
        # now, copy the blocks within the ix-th row
        if ix > 1
            # left
            for jx = (ix-1):-1:max(1, ix - Int((ndiag["out"] - 1) / 2))
                be_copy_in_hw!(Mout.M[ix, jx], Min.M[ix, jx])
            end
        end
        # center
        be_copy_in_hw!(Mout.M[ix, ix], Min.M[ix, ix])
        if ix < size(Min.blockSizes)[1]
            # right
            for jx = (ix+1):1:min(size(Min.blockSizes)[1], ix + Int((ndiag["out"] - 1) / 2))
                be_copy_in_hw!(Mout.M[ix, jx], Min.M[ix, jx])
            end
        end
    end
end

"""
	bm_similar(M::BlockMatrix)

Receives a BlockMatrix object and returns a BlockMatrix with the same properties
(block n-diagonal wise), but the dense blocks filled according to the value in
`filling`.

# Arguments
- `M::BlockMatrix`: the matrix to be used as base.
- `filling:Int`: 0 for empty blocks, 1 for zero, 2 for random.
"""
function bm_similar(M::BlockMatrix, filling::Int)::BlockMatrix
    npl = size(M.blockSizes)[1]

    if filling == 0
        return BlockMatrix(M.blockSizes, ArrayOrLU_(undef, npl, npl), M.ndiag, M.nrsType, M.isArrayOrLU)
    else
        blockSizes = M.blockSizes
        ndiag = M.ndiag
        npl = size(blockSizes)[1]
        A = BlockMatrix(M.blockSizes, ArrayOrLU_(undef, npl, npl), M.ndiag, M.nrsType, M.isArrayOrLU)
        bm_blocks_define!(A, filling)
        return A
    end
end

"""
	bm_blocks_define!(M::BlockMatrix)

Receives a BlockMatrix object, and set its dense blocks to either zero or random.

# Arguments
- `M::BlockMatrix`: the matrix to be modified.
- `filling:Int`: 1 for zero blocks, 2 for random.
"""
function bm_blocks_define!(M::BlockMatrix, filling::Int)
    if filling == 2 && M.isArrayOrLU == 1
        println("ERROR: filling up BlockMatrix with random blocks and block-diagonal LUs
                 makes no sense")
        exit()
    end

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
                if filling == 1
                    A.M[ix, jx] = be_zero_array(A.nrsType, (iend - ibeg + 1, jend - jbeg + 1))
                else
                    A.M[ix, jx] = be_random_array(A.nrsType, (iend - ibeg + 1, jend - jbeg + 1))
                end
            end
        end
        # center
        jbeg = ibeg
        jend = iend
        if isArrayOrLU == 0
            if filling == 1
                A.M[ix, ix] = be_zero_array(A.nrsType, (iend - ibeg + 1, jend - jbeg + 1))
            else
                A.M[ix, ix] = be_random_array(A.nrsType, (iend - ibeg + 1, jend - jbeg + 1))
            end
        else
            A.M[ix, ix] = be_zero_lu(A.nrsType, iend - ibeg + 1)
        end
        if ix < npl
            # right
            for jx = (ix+1):1:min(size(blockSizes)[1], ix + Int((ndiag["in"] - 1) / 2))
                jbeg = sum(blockSizes[1:jx-1]) + 1
                jend = sum(blockSizes[1:jx])
                if filling == 1
                    A.M[ix, jx] = be_zero_array(A.nrsType, (iend - ibeg + 1, jend - jbeg + 1))
                else
                    A.M[ix, jx] = be_random_array(A.nrsType, (iend - ibeg + 1, jend - jbeg + 1))
                end
            end
        end
    end
end

"""
	bm_blocks_define_identity!(M::BlockMatrix)

Receives a BlockMatrix object, and set its dense blocks to the identity.
This has, for now, been restricted to the identity, i.e. we are constructing
here the identity in block 1-diagonal form.

# Arguments
- `M::BlockMatrix`: the matrix to be modified.
"""
function bm_blocks_define_identity!(M::BlockMatrix)
    blockSizes = M.blockSizes
    ndiag = M.ndiag
    if ndiag["in"] > 1
        println("ERROR: this function is meant, for now, only for block-diagonal block matrices")
        exit()
    end
    npl = size(blockSizes)[1]
    # just a label of M
    A = M

    # loop over the block sizes, conversely over the block rows
    for ix = 1:npl
        # indices for the rows
        ibeg = sum(blockSizes[1:ix-1]) + 1
        iend = sum(blockSizes[1:ix])
        A.M[ix, ix] = be_identity(A.nrsType, iend - ibeg + 1)
    end
end

# tB can be either 'C' (for adjoint) or 'N' for no adjoint
function bm_local_gemm!(tA::Char, tB::Char, alpha::Number, A_::BlockMatrix, B_::BlockMatrix, beta::Number,
    C_::BlockMatrix, ix::Int, jx::Int, td::TimingData, cd::CountingData)
    npl = size(A_.blockSizes)[1]
    nUpDiagA = Int((A_.ndiag["in"] - 1) / 2)
    nUpDiagB = Int((B_.ndiag["in"] - 1) / 2)
    nrsType = A_.nrsType

    C = C_.M
    A = A_.M
    B = B_.M

    if beta == convert(nrsType, 0.0)
        be_fill!(C[ix, jx], convert(A_.nrsType, 0.0))
    end

    for kx = 1:npl
        # avoid accessing undefs in A and B
        cond1 = (kx <= ix + nUpDiagA) && (kx >= ix - nUpDiagA)
        cond2 = (jx <= kx + nUpDiagB) && (jx >= kx - nUpDiagB)

        if cond1 && cond2
            # do the transposition by hand
            if tB == 'C'
                be_gemm!('N', 'C', alpha, A[ix, kx], B[jx, kx], convert(nrsType, 1.0), C[ix, jx], td, cd)
            else
                be_gemm!('N', 'N', alpha, A[ix, kx], B[kx, jx], convert(nrsType, 1.0), C[ix, jx], td, cd)
            end
        end
    end
end

"""
	bm_local_gemm!(tA::Char, tB::Char, alpha::Number, A_::BlockMatrix, B_::BlockMatrix, beta::Number, C_::BlockMatrix, ix::Int, jx::Int)

GEMM for BlockMatrix type matrices. The signature of the function
follows closely that being used in BLAS. This function does
C = beta*C + alpha*A*B.

# Arguments
- `tA::Char`: whether we take the adjoint of A ('C') or not ('N').
- `tB::Char`: whether we take the adjoint of B ('C') or not ('N').
"""
function bm_gemm!(tA::Char, tB::Char, alpha::Number, A::BlockMatrix, B::BlockMatrix, beta::Number, C::BlockMatrix,
    td::TimingData, cd::CountingData)
    ndiag = C.ndiag
    npl = size(B.blockSizes)[1]

    # loop over the block sizes, conversely over the block rows
    for ix = 1:npl
        # now, copy the blocks within the ix-th row
        # left
        if ix > 1
            for jx = (ix-1):-1:max(1, ix - Int((ndiag["out"] - 1) / 2))
                bm_local_gemm!(tA, tB, alpha, A, B, beta, C, ix, jx, td, cd)
            end
        end
        # center
        bm_local_gemm!(tA, tB, alpha, A, B, beta, C, ix, ix, td, cd)
        # right
        if ix < npl
            for jx = (ix+1):1:min(npl, ix + Int((ndiag["out"] - 1) / 2))
                bm_local_gemm!(tA, tB, alpha, A, B, beta, C, ix, jx, td, cd)
            end
        end
    end
end