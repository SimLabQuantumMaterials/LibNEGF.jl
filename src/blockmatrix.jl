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
    isHermitian::Bool
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
    ndiag::Dict{String,Int}, isHermitian::Bool)::BlockMatrix
    # npl stands for number of principal layers
    npl = size(blockSizes)[1]
    # in general, these type of block matrices will contain Array-like object and not LU-like
    A = BlockMatrix(copy(blockSizes), ArrayOrLU_(undef, npl, npl), ndiag, typeof(M[1, 1]), 0, isHermitian)

    # loop over the block sizes, conversely over the block rows
    for ix = 1:npl
        # indices for the rows
        ibeg = sum(blockSizes[1:ix-1]) + 1
        iend = sum(blockSizes[1:ix])
        # now, copy the blocks within the ix-th row
        if !isHermitian
            if ix > 1
                # left
                for jx = (ix-1):-1:max(1, ix - Int((ndiag["out"] - 1) / 2))
                    jbeg = sum(blockSizes[1:jx-1]) + 1
                    jend = sum(blockSizes[1:jx])
                    A.M[ix, jx] = be_copy_to_hw(Array(M[ibeg:iend, jbeg:jend]))
                end
            end
        end
        # center
        jbeg = ibeg
        jend = iend
        A.M[ix, ix] = be_copy_to_hw(Array(M[ibeg:iend, jbeg:jend]))
        if ix < npl
            # right
            for jx = (ix+1):1:min(npl, ix + Int((ndiag["out"] - 1) / 2))
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
    npl = size(M.blockSizes)[1]

    # create the empty sparse matrix to be the output, with the
    # appropriate underlying data type in nrsType
    A = SparseArrays.SparseMatrixCSC{nrsType,Int}(undef, n, n)

    # loop over the block sizes, conversely over the block rows
    for ix = 1:npl
        # indices for the rows
        ibeg = sum(M.blockSizes[1:ix-1]) + 1
        iend = sum(M.blockSizes[1:ix])
        # now, copy the blocks within the ix-th row
        if ix > 1
            # left
            for jx = (ix-1):-1:max(1, ix - Int((ndiag["out"] - 1) / 2))
                jbeg = sum(M.blockSizes[1:jx-1]) + 1
                jend = sum(M.blockSizes[1:jx])
                if !M.isHermitian
                    A[ibeg:iend, jbeg:jend] = sparse(be_copy_from_hw(M.M[ix, jx]))
                else
                    A[ibeg:iend, jbeg:jend] = sparse(be_copy_from_hw(M.M[jx, ix]))'
                end
            end
        end
        # center
        jbeg = ibeg
        jend = iend
        A[ibeg:iend, jbeg:jend] = sparse(be_copy_from_hw(M.M[ix, ix]))
        if ix < npl
            # right
            for jx = (ix+1):1:min(npl, ix + Int((ndiag["out"] - 1) / 2))
                jbeg = sum(M.blockSizes[1:jx-1]) + 1
                jend = sum(M.blockSizes[1:jx])
                A[ibeg:iend, jbeg:jend] = sparse(be_copy_from_hw(M.M[ix, jx]))
            end
        end
    end

    return A
end

"""
	bm_convert(M::BlockMatrix, permVec::Vector{Int})

Converts the input matrix `M` of type `BlockMatrix` to sparse.
This allows us converting a permuted block matrix to sparse.

# Arguments
- `M::BlockMatrix`: the matrix to be converted.
- `permVec::Vector{Int}`: the DDRGF permutation vector.
"""
function bm_convert(M::BlockMatrix, permVec::Vector{Int})::SparseArrays.SparseMatrixCSC
    n = sum(M.blockSizes)
    ndiag = M.ndiag
    nrsType = M.nrsType
    pv = permVec
    npl = size(M.blockSizes)[1]

    # create the empty sparse matrix to be the output, with the
    # appropriate underlying data type in nrsType
    A = SparseArrays.SparseMatrixCSC{nrsType,Int}(undef, n, n)

    # loop over the block sizes, conversely over the block rows
    for ix = 1:npl
        ixPerm = pv[ix]
        # indices for the rows
        ibeg = sum(M.blockSizes[1:ixPerm-1]) + 1
        iend = sum(M.blockSizes[1:ixPerm])
        # now, copy the blocks within the ix-th row
        if !M.isHermitian
            if ix > 1
                # left
                for jx = (ix-1):-1:max(1, ix - Int((ndiag["out"] - 1) / 2))
                    jxPerm = pv[jx]
                    jbeg = sum(M.blockSizes[1:jxPerm-1]) + 1
                    jend = sum(M.blockSizes[1:jxPerm])
                    A[ibeg:iend, jbeg:jend] = sparse(be_copy_from_hw(M.M[ixPerm, jxPerm]))
                end
            end
        end
        # center
        jbeg = ibeg
        jend = iend
        A[ibeg:iend, jbeg:jend] = sparse(be_copy_from_hw(M.M[ixPerm, ixPerm]))
        if ix < npl
            # right
            for jx = (ix+1):1:min(npl, ix + Int((ndiag["out"] - 1) / 2))
                jxPerm = pv[jx]
                jbeg = sum(M.blockSizes[1:jxPerm-1]) + 1
                jend = sum(M.blockSizes[1:jxPerm])
                A[ibeg:iend, jbeg:jend] = sparse(be_copy_from_hw(M.M[ixPerm, jxPerm]))
            end
        end
    end

    return A
end

"""
	bndiag_of_inv_ddrgf_create_sparse_permutator(permVec::Vector{Int}, blockSizes::Vector{Int},
        nrsType::DataType)

Create sparse matrix that implements permutations from the `permVec` vector.

# Arguments
- `permVec::Vector{Int}`: the permutation vector.
- `blockSizes::Vector{Int}`: the sizes of the principal layers.
- `nrsType::DataType`: the type of the underlying data.
"""
function bndiag_of_inv_ddrgf_create_sparse_permutator(permVec::Vector{Int}, blockSizes::Vector{Int},
    nrsType::DataType)::SparseArrays.SparseMatrixCSC

    n = sum(blockSizes)
    # ndiag = M.ndiag
    # nrsType = M.nrsType
    pv = permVec
    npl = size(blockSizes)[1]
    # we also need the permuted array of block sizes
    blockSizesPerm = copy(blockSizes)
    for ix = 1:npl
        blockSizesPerm[pv[ix]] = blockSizes[ix]
    end

    # create the empty sparse matrix to be the output, with the
    # appropriate underlying data type in nrsType
    A = SparseArrays.SparseMatrixCSC{nrsType,Int}(undef, n, n)

    for jx = 1:npl
        ix = pv[jx]
        iStart = 1 + sum(blockSizesPerm[1:ix-1])
        iEnd = sum(blockSizesPerm[1:ix])
        jStart = 1 + sum(blockSizes[1:jx-1])
        jEnd = sum(blockSizes[1:jx])
        A[iStart:iEnd, jStart:jEnd] = sparse(I, iEnd - iStart + 1, jEnd - jStart + 1)
    end

    return A
end

"""
	bm_copy(M::BlockMatrix)

Receives a BlockMatrix object and returns a deep copy of it.

# Arguments
- `M::BlockMatrix`: the matrix to be copied.
"""
function bm_copy(M::BlockMatrix)::BlockMatrix
    ndiag = M.ndiag
    npl = size(M.blockSizes)[1]

    A = BlockMatrix(copy(M.blockSizes), ArrayOrLU_(undef, npl, npl), ndiag, M.nrsType, M.isArrayOrLU, M.isHermitian)

    # loop over the block sizes, conversely over the block rows
    for ix = 1:npl
        # now, copy the blocks within the ix-th row
        if !M.isHermitian
            if ix > 1
                # left
                for jx = (ix-1):-1:max(1, ix - Int((ndiag["out"] - 1) / 2))
                    A.M[ix, jx] = be_copy_in_hw(M.M[ix, jx])
                end
            end
        end
        # center
        A.M[ix, ix] = be_copy_in_hw(M.M[ix, ix])
        if ix < npl
            # right
            for jx = (ix+1):1:min(npl, ix + Int((ndiag["out"] - 1) / 2))
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
    for ix = 1:npl
        # now, copy the blocks within the ix-th row
        if !Min.isHermitian
            if ix > 1
                # left
                for jx = (ix-1):-1:max(1, ix - Int((ndiag["out"] - 1) / 2))
                    be_copy_in_hw!(Mout.M[ix, jx], Min.M[ix, jx])
                end
            end
        end
        # center
        be_copy_in_hw!(Mout.M[ix, ix], Min.M[ix, ix])
        if ix < npl
            # right
            for jx = (ix+1):1:min(npl, ix + Int((ndiag["out"] - 1) / 2))
                be_copy_in_hw!(Mout.M[ix, jx], Min.M[ix, jx])
            end
        end
    end
end

"""
	bm_similar(M::BlockMatrix, filling::Int)

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
        return BlockMatrix(copy(M.blockSizes), ArrayOrLU_(undef, npl, npl), M.ndiag, M.nrsType, M.isArrayOrLU, M.isHermitian)
    else
        blockSizes = M.blockSizes
        npl = size(blockSizes)[1]
        A = BlockMatrix(copy(M.blockSizes), ArrayOrLU_(undef, npl, npl), M.ndiag, M.nrsType, M.isArrayOrLU, M.isHermitian)
        bm_blocks_define!(A, filling)
        return A
    end
end

function bm_empty(blockSizes::Vector{Int}, npl::Int, ndiag::Int, isArrayOrLU::Bool, nrsType::DataType, isHermitian::Bool)::BlockMatrix
    return BlockMatrix(copy(blockSizes), ArrayOrLU_(undef, npl, npl), Dict("in" => ndiag, "out" => ndiag), nrsType, isArrayOrLU, isHermitian)
end

"""
	bm_blocks_define!(M::BlockMatrix, filling::Int)

Receives a BlockMatrix object, and sets its dense blocks to either zero or random.

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
        iSize = blockSizes[ix]
        # now, copy the blocks within the ix-th row
        if !A.isHermitian
            if ix > 1
                # left
                for jx = (ix-1):-1:max(1, ix - Int((ndiag["in"] - 1) / 2))
                    jSize = blockSizes[jx]
                    if filling == 1
                        A.M[ix, jx] = be_zero_array(A.nrsType, (iSize, jSize))
                    else
                        A.M[ix, jx] = be_random_array(A.nrsType, (iSize, jSize))
                    end
                end
            end
        end
        # center
        if isArrayOrLU == 0
            if filling == 1
                A.M[ix, ix] = be_zero_array(A.nrsType, (iSize, iSize))
            else
                A.M[ix, ix] = be_random_array(A.nrsType, (iSize, iSize))
            end
        else
            A.M[ix, ix] = be_zero_lu(A.nrsType, iSize)
        end
        if ix < npl
            # right
            for jx = (ix+1):1:min(size(blockSizes)[1], ix + Int((ndiag["in"] - 1) / 2))
                jSize = blockSizes[jx]
                if filling == 1
                    A.M[ix, jx] = be_zero_array(A.nrsType, (iSize, jSize))
                else
                    A.M[ix, jx] = be_random_array(A.nrsType, (iSize, jSize))
                end
            end
        end
    end
end

"""
	bm_blocks_define_complement11!(M::BlockMatrix, A::ArrayOrLUView_, filling::Int)

Sets/pre-allocates all those blocks that are beyond block tridiagonal.

# Arguments
- `M::BlockMatrix`: some metadata.
- `A::ArrayOrLUView_`: the matrix to be modified.
- `filling:Int`: 1 for zero blocks, 2 for random.
"""
function bm_blocks_define_complement11!(M::BlockMatrix, A::ArrayOrLUView_, filling::Int)
    # TODO : integrate the use of M.ndiag["out"]
    # ndiag = M.ndiag

    blockSizes = M.blockSizes
    npl = size(blockSizes)[1]

    # loop over the block sizes, conversely over the block rows
    for ix = 1:npl
        # indices for the rows
        iSize = blockSizes[ix]

        # left
        for jx = (ix-2):-1:1
            jSize = blockSizes[jx]
            if filling == 1
                A[ix, jx] = be_zero_array(M.nrsType, (iSize, jSize))
            else
                A[ix, jx] = be_random_array(M.nrsType, (iSize, jSize))
            end
        end

        # right
        for jx = (ix+2):1:npl
            jSize = blockSizes[jx]
            if filling == 1
                A[ix, jx] = be_zero_array(M.nrsType, (iSize, jSize))
            else
                A[ix, jx] = be_random_array(M.nrsType, (iSize, jSize))
            end
        end
    end
end

# TODO : try to assign the type AuxDataDDRGF to auxData ?
"""
	bm_blocks_define_complement22!(M::BlockMatrix, auxData, filling::Int)

Sets/pre-allocates all those blocks that are beyond block tridiagonal in
the matrix `M`.

# Arguments
- `M::BlockMatrix`: the matrix in which the extra allocations will be placed.
- `auxData`: some metadata.
- `filling:Int`: 1 for zero blocks, 2 for random.
"""
function bm_blocks_define_complement22_non_recurs!(M::BlockMatrix, auxData, filling::Int)
    blockSizeD2 = auxData.blockSizeD2
    permVecInv = auxData.permVecInv
    buffTHat = auxData.buffTHat
    blockSizes = buffTHat.blockSizes
    nrTasks = auxData.nrTasks

    for ix = 1:nrTasks-1
        # first, the upper one
        ixLperm = blockSizeD2 * ix
        jxLperm = ixLperm + 1
        ixL = permVecInv[ixLperm]
        jxL = permVecInv[jxLperm]

        iSize = blockSizes[ixL]
        jSize = blockSizes[jxL]
        if filling == 1
            M.M[ixL, jxL] = be_zero_array(M.nrsType, (iSize, jSize))
        else
            M.M[ixL, jxL] = be_random_array(M.nrsType, (iSize, jSize))
        end

        # then, the lower one
        jxLperm = blockSizeD2 * ix
        ixLperm = jxLperm + 1
        ixL = permVecInv[ixLperm]
        jxL = permVecInv[jxLperm]

        iSize = blockSizes[ixL]
        jSize = blockSizes[jxL]
        if filling == 1
            M.M[ixL, jxL] = be_zero_array(M.nrsType, (iSize, jSize))
        else
            M.M[ixL, jxL] = be_random_array(M.nrsType, (iSize, jSize))
        end
    end
end

# recursively permute depending on the number of levels in DDRGF,
# this is exclusive to the global (i.e., fine-grid) Mout, it takes
# an index at a certain level in the recursion and returns the corresponding
# index at the finest level
function recursPermIndx(levelNr::Int, listOfAuxData, idx_::Int)
    idx = listOfAuxData[levelNr].permVecInv[idx_]
    if levelNr == 1
        return idx
    else
        return recursPermIndx(levelNr - 1, listOfAuxData, idx)
    end
end

# TODO : try to assign the type AuxDataDDRGF to auxData ?
"""
	bm_blocks_define_complement22!(M::BlockMatrix, auxData, filling::Int)

Sets/pre-allocates all those blocks that are beyond block tridiagonal in
the matrix `M`. This function is specific to Mout, where the blocks are defined
into the depth of the global recursion of DDRGF.

# Arguments
- `M::BlockMatrix`: the matrix in which the extra allocations will be placed.
- `auxData`: some metadata.
- `filling:Int`: 1 for zero blocks, 2 for random.
"""
function bm_blocks_define_complement22_recurs!(M::BlockMatrix, listOfAuxData, filling::Int)
    buffTHat = listOfAuxData[1].buffTHat
    blockSizes = buffTHat.blockSizes

    nrLevels = size(listOfAuxData)[1]

    for ix_ = 1:nrLevels
        blockSizeD2 = listOfAuxData[ix_].blockSizeD2
        nrTasks = listOfAuxData[ix_].nrTasks
        for ix = 1:nrTasks-1
            # first, the upper one
            ixLperm = blockSizeD2 * ix
            jxLperm = ixLperm + 1
            ixL = recursPermIndx(ix_, listOfAuxData, ixLperm)
            jxL = recursPermIndx(ix_, listOfAuxData, jxLperm)

            iSize = blockSizes[ixL]
            jSize = blockSizes[jxL]
            if filling == 1
                M.M[ixL, jxL] = be_zero_array(M.nrsType, (iSize, jSize))
            else
                M.M[ixL, jxL] = be_random_array(M.nrsType, (iSize, jSize))
            end

            # then, the lower one
            jxLperm = blockSizeD2 * ix
            ixLperm = jxLperm + 1
            ixL = recursPermIndx(ix_, listOfAuxData, ixLperm)
            jxL = recursPermIndx(ix_, listOfAuxData, jxLperm)

            iSize = blockSizes[ixL]
            jSize = blockSizes[jxL]
            if filling == 1
                M.M[ixL, jxL] = be_zero_array(M.nrsType, (iSize, jSize))
            else
                M.M[ixL, jxL] = be_random_array(M.nrsType, (iSize, jSize))
            end
        end
    end
end

# TODO : try to assign the type AuxDataDDRGF to auxData ?
"""
	bm_blocks_define_complement12!(M_::BlockMatrix, auxData, filling::Int)

Sets/pre-allocates, within `M`, the extra `12` part for DDRGF.

# Arguments
- `M_::BlockMatrix`: the matrix in which the extra allocations will be placed.
- `auxData`: some metadata.
- `filling:Int`: 1 for zero blocks, 2 for random.
"""
function bm_blocks_define_complement12!(M_::BlockMatrix, auxData, filling::Int)
    blockSizeD1 = auxData.blockSizeD1
    lastSizeD1 = auxData.lastSizeD1
    permVecInv = auxData.permVecInv
    M = M_
    blockSizes = M.blockSizes
    nrTasks = auxData.nrTasks
    sizeDomains = auxData.sizeDomains
    sizeDomains22 = sizeDomains[1:nrTasks]
    sizeDomains11 = sizeDomains[nrTasks+1:2*nrTasks]

    buffBlockSizeD1 = blockSizeD1
    for ix_ = 1:nrTasks
        ixLpermOffset = sum(sizeDomains22) + sum(sizeDomains11[1:ix_-1])

        if ix_ == nrTasks
            buffBlockSizeD1 = lastSizeD1
        end
        # first, the central sub-domain
        jxLperm = sum(sizeDomains22[1:ix_])
        for ix = 2:buffBlockSizeD1
            ixLperm = ixLpermOffset + ix

            ixL = permVecInv[ixLperm]
            jxL = permVecInv[jxLperm]

            iSize = blockSizes[ixL]
            jSize = blockSizes[jxL]
            if filling == 1
                M.M[ixL, jxL] = be_zero_array(M.nrsType, (iSize, jSize))
            else
                M.M[ixL, jxL] = be_random_array(M.nrsType, (iSize, jSize))
            end
        end

        if ix_ < nrTasks
            # then, the right sub-domain
            jxLperm = sum(sizeDomains22[1:ix_]) + 1
            for ix = 1:buffBlockSizeD1-1
                ixLperm = ixLpermOffset + ix

                ixL = permVecInv[ixLperm]
                jxL = permVecInv[jxLperm]

                iSize = blockSizes[ixL]
                jSize = blockSizes[jxL]
                if filling == 1
                    M.M[ixL, jxL] = be_zero_array(M.nrsType, (iSize, jSize))
                else
                    M.M[ixL, jxL] = be_random_array(M.nrsType, (iSize, jSize))
                end
            end
        end
    end
end

# TODO : try to assign the type AuxDataDDRGF to auxData ?
"""
	bm_blocks_define_complement21!(M_::BlockMatrix, auxData, filling::Int)

Sets/pre-allocates, within `M`, the extra `21` part for DDRGF.

# Arguments
- `M_::BlockMatrix`: the matrix in which the extra allocations will be placed.
- `auxData`: some metadata.
- `filling:Int`: 1 for zero blocks, 2 for random.
"""
function bm_blocks_define_complement21!(M_::BlockMatrix, auxData, filling::Int)
    blockSizeD1 = auxData.blockSizeD1
    lastSizeD1 = auxData.lastSizeD1
    permVecInv = auxData.permVecInv
    M = M_
    blockSizes = M.blockSizes
    nrTasks = auxData.nrTasks
    sizeDomains = auxData.sizeDomains
    sizeDomains22 = sizeDomains[1:nrTasks]
    sizeDomains11 = sizeDomains[nrTasks+1:2*nrTasks]

    buffBlockSizeD1 = blockSizeD1
    for jx_ = 1:nrTasks
        jxLpermOffset = sum(sizeDomains22) + sum(sizeDomains11[1:jx_-1])

        if jx_ == nrTasks
            buffBlockSizeD1 = lastSizeD1
        end

        # first, the central sub-domain
        ixLperm = sum(sizeDomains22[1:jx_])
        for jx = 2:buffBlockSizeD1
            jxLperm = jxLpermOffset + jx

            ixL = permVecInv[ixLperm]
            jxL = permVecInv[jxLperm]

            iSize = blockSizes[ixL]
            jSize = blockSizes[jxL]
            if filling == 1
                M.M[ixL, jxL] = be_zero_array(M.nrsType, (iSize, jSize))
            else
                M.M[ixL, jxL] = be_random_array(M.nrsType, (iSize, jSize))
            end
        end

        if jx_ < nrTasks
            # then, the right sub-domain
            ixLperm = sum(sizeDomains22[1:jx_]) + 1
            for jx = 1:buffBlockSizeD1-1
                jxLperm = jxLpermOffset + jx

                ixL = permVecInv[ixLperm]
                jxL = permVecInv[jxLperm]

                iSize = blockSizes[ixL]
                jSize = blockSizes[jxL]
                if filling == 1
                    M.M[ixL, jxL] = be_zero_array(M.nrsType, (iSize, jSize))
                else
                    M.M[ixL, jxL] = be_random_array(M.nrsType, (iSize, jSize))
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
        iSize = blockSizes[ix]
        A.M[ix, ix] = be_identity(A.nrsType, iSize)
    end
end

# TODO : is documentation deployment needed for this one?
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

# TODO : is documentation deployment needed for this one?
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

"""
	bm_create_synthetic(A_::BlockMatrix, nrLayers::Int, blocksDim::Int)

Create a semi-synthetic block tridiagonal matrix, based on the data in
the matrix `A_`.

# Arguments
- `A_::BlockMatrix`: where the data comes from.
- `nrLayers::Int`: the number of principal layers.
- `blocksDim::Int`: the size of the principal layers (i.e., blocks).
"""
function bm_create_synthetic(A_::BlockMatrix, nrLayers::Int, blocksDim::Int)::BlockMatrix
    # IMPORTANT : this function assumes that all of the principal layers are of
    #             the same size

    # the following two come from A_
    npl = size(A_.blockSizes)[1]
    # repeat the central layers (i.e. without first and last)
    npl -= 2
    ndiag = A_.ndiag

    # this is for A
    blockSizes = repeat([blocksDim], nrLayers)

    lowLayers = Int(floor(nrLayers / npl))
    restLayers = nrLayers - lowLayers * npl

    A = BlockMatrix(copy(blockSizes), ArrayOrLU_(undef, nrLayers, nrLayers), ndiag, A_.nrsType, 0, A_.isHermitian)

    # loop over chunks of layers
    for olx = 1:lowLayers+1
        # loop over the block sizes within a chunk, conversely over the block rows
        if olx < lowLayers + 1
            nrLoopLayers = npl
        else
            nrLoopLayers = restLayers
        end
        # ixL and jxL are local, and ixG and jxG global
        offsetG = (olx - 1) * npl
        for ixL = 1:nrLoopLayers
            ixG = ixL + offsetG
            # now, copy the blocks within the ix-th row
            if !A.isHermitian
                if ixL > 1
                    # left
                    for jxL = (ixL-1):-1:max(1, ixL - Int((ndiag["out"] - 1) / 2))
                        jxG = jxL + offsetG
                        A.M[ixG, jxG] = be_copy_in_hw((A_.M[1+ixL, 1+jxL])[1:blocksDim, 1:blocksDim])
                    end
                end
            end
            # center
            A.M[ixG, ixG] = be_copy_in_hw((A_.M[1+ixL, 1+ixL])[1:blocksDim, 1:blocksDim])
            if ixL < nrLoopLayers
                # right
                for jxL = (ixL+1):1:min(nrLoopLayers, ixL + Int((ndiag["out"] - 1) / 2))
                    jxG = jxL + offsetG
                    A.M[ixG, jxG] = be_copy_in_hw((A_.M[1+ixL, 1+jxL])[1:blocksDim, 1:blocksDim])
                end
            end

            # do the joints between chunks of principal layers
            if (ixL == npl) && (ixG < nrLayers)
                A.M[ixG, ixG+1] = be_copy_in_hw((A_.M[1+ixL-1, 1+ixL])[1:blocksDim, 1:blocksDim])
                A.M[ixG+1, ixG] = be_copy_in_hw((A_.M[1+ixL, 1+ixL-1])[1:blocksDim, 1:blocksDim])
            end
        end
    end

    # IMPORTANT : up to here, we have populated the semi-synthetic matrix with data
    #             coming from the device i.e. we do not use the layers next to the contacts.
    #             Next, we correct for this

    # first, top-left corner
    A.M[1, 1] = be_copy_in_hw((A_.M[1, 1])[1:blocksDim, 1:blocksDim])
    A.M[1, 2] = be_copy_in_hw((A_.M[1, 2])[1:blocksDim, 1:blocksDim])
    if !A.isHermitian
        A.M[2, 1] = be_copy_in_hw((A_.M[2, 1])[1:blocksDim, 1:blocksDim])
    end

    # then, bottom-right corner - for this, restore npl to the actual total
    npl += 2
    A.M[nrLayers, nrLayers] = be_copy_in_hw((A_.M[npl, npl])[1:blocksDim, 1:blocksDim])
    A.M[nrLayers-1, nrLayers] = be_copy_in_hw((A_.M[npl-1, npl])[1:blocksDim, 1:blocksDim])
    if !A.isHermitian
        A.M[nrLayers, nrLayers-1] = be_copy_in_hw((A_.M[npl, npl-1])[1:blocksDim, 1:blocksDim])
    end

    return A
end

"""
	bm_create_synthetic_random(nrLayers::Int, blocksDim::Int, nrsType::DataType)

Create a random synthetic block tridiagonal matrix.

# Arguments
- `nrLayers::Int`: the number of principal layers.
- `blocksDim::Int`: the average size of the principal layers (i.e., blocks).
- `nrsType::DataType`: the type of the underlying data.
"""
function bm_create_synthetic_random(nrLayers::Int, blocksDim::Int, nrsType::DataType, isHermitian::Bool)::BlockMatrix
    # IMPORTANT : this function assumes that all of the principal layers are of
    #             the same size

    # blockSizes = repeat([blocksDim], nrLayers)
    deltaRnd = 8.0
    blockSizes::Vector{Int} = Int.(round.(broadcast(*, deltaRnd, rand(nrLayers)) .+ (blocksDim - deltaRnd / 2.0)))

    # hardcoding block tridiagonal
    ndiag = Dict("in" => 3, "out" => 3)

    A = BlockMatrix(copy(blockSizes), ArrayOrLU_(undef, nrLayers, nrLayers), ndiag, nrsType, 0, isHermitian)

    # loop over chunks of layers
    for ix = 1:nrLayers
        if !isHermitian
            if ix > 1
                # left
                for jx = (ix-1):-1:max(1, ix - Int((ndiag["out"] - 1) / 2))
                    # add a damping of 0.3
                    blocksDimI = blockSizes[ix]
                    blocksDimJ = blockSizes[jx]
                    A.M[ix, jx] = convert(nrsType, 0.3) * be_random_array(nrsType, (blocksDimI, blocksDimJ))
                end
            end
        end
        # center
        blocksDimI = blockSizes[ix]
        A.M[ix, ix] = be_random_array(nrsType, (blocksDimI, blocksDimI))
        if ix < nrLayers
            # right
            for jx = (ix+1):1:min(nrLayers, ix + Int((ndiag["out"] - 1) / 2))
                # add a damping of 0.3
                blocksDimI = blockSizes[ix]
                blocksDimJ = blockSizes[jx]
                A.M[ix, jx] = convert(nrsType, 0.3) * be_random_array(nrsType, (blocksDimI, blocksDimJ))
            end
        end
    end

    return A
end

"""
	bm_reference!(M::BlockMatrix, B::ArrayOrLUView_)

Assign references in `A.M` to the blocks in `B`, using views.

# Arguments
- `M::BlockMatrix`: output.
- `B::ArrayOrLUView_`: input.
"""
function bm_reference!(M::BlockMatrix, B::ArrayOrLUView_)
    npl = size(M.blockSizes)[1]
    ndiag = M.ndiag

    # loop over the block sizes, conversely over the block rows
    for ix = 1:npl
        # now, copy the blocks within the ix-th row
        if !M.isHermitian
            if ix > 1
                # left
                for jx = (ix-1):-1:max(1, ix - Int((ndiag["out"] - 1) / 2))
                    M.M[ix, jx] = B[ix, jx]
                end
            end
        end
        # center
        M.M[ix, ix] = B[ix, ix]
        if ix < npl
            # right
            for jx = (ix+1):1:min(npl, ix + Int((ndiag["out"] - 1) / 2))
                M.M[ix, jx] = B[ix, jx]
            end
        end
    end
end

"""
	bm_reference!(M::BlockMatrix, B::ArrayOrLU_, iOffset::Int, jOffset::Int)

Assign references in `A.M` to the blocks in `B`, avoids using views.

# Arguments
- `M::BlockMatrix`: output.
- `B::ArrayOrLUView_`: input.
- `iOffset::Int`: to point to the sub-matrix to reference to.
- `jOffset::Int`: to point to the sub-matrix to reference to.
"""
function bm_reference!(M::BlockMatrix, B::ArrayOrLU_, iOffset::Int, jOffset::Int)
    npl = size(M.blockSizes)[1]
    ndiag = M.ndiag

    # loop over the block sizes, conversely over the block rows
    for ix = 1:npl
        # now, copy the blocks within the ix-th row
        if !M.isHermitian
            if ix > 1
                # left
                for jx = (ix-1):-1:max(1, ix - Int((ndiag["out"] - 1) / 2))
                    ix_ = iOffset + ix
                    jx_ = jOffset + jx
                    M.M[ix, jx] = B[ix_, jx_]
                end
            end
        end
        # center
        ix_ = iOffset + ix
        M.M[ix, ix] = B[ix_, ix_]
        if ix < npl
            # right
            for jx = (ix+1):1:min(npl, ix + Int((ndiag["out"] - 1) / 2))
                ix_ = iOffset + ix
                jx_ = jOffset + jx
                M.M[ix, jx] = B[ix_, jx_]
            end
        end
    end
end

# assign references in A.M to the blocks in B
function bm_reference_full!(M::BlockMatrix, B::ArrayOrLUView_)
    npl = size(M.blockSizes)[1]

    for ix = 1:npl
        for jx = 1:npl
            M.M[ix, jx] = B[ix, jx]
        end
    end
end

# assign references in A.M to the blocks in B
function bm_reference_full!(M::BlockMatrix, B::ArrayOrLU_, iOffset::Int, jOffset::Int)
    npl = size(M.blockSizes)[1]

    for ix = 1:npl
        for jx = 1:npl
            ix_ = iOffset + ix
            jx_ = jOffset + jx
            M.M[ix, jx] = B[ix_, jx_]
        end
    end
end