"""
    bndiag_of_inv_direct(M::SparseArrays.SparseMatrixCSC, blockSizes::Vector{Int})

For an input matrix `M`, block tridiagonal, compute the block tridiagonal
part of the inverse of `M`. This function first computes `inv(M)` and then
extracts its block tridiagonal.
"""
function bndiag_of_inv_direct(M::SparseArrays.SparseMatrixCSC,
    blockSizes::Vector{Int})::SparseArrays.SparseMatrixCSC
    Mdense = Array(M)
    MdenseInv = inv(Mdense)
    MInvTrid = copy(M)
    # loop over the block sizes, conversely over the block rows
    for ix = 1:size(blockSizes)[1]
        # indices for the rows
        ibeg = sum(blockSizes[1:ix-1]) + 1
        iend = sum(blockSizes[1:ix])
        # now, copy the blocks within the ix-th row
        if ix > 1
            # left
            jbeg = sum(blockSizes[1:ix-2]) + 1
            jend = sum(blockSizes[1:ix-1])
            MInvTrid[ibeg:iend, jbeg:jend] = MdenseInv[ibeg:iend, jbeg:jend]
        end
        # center
        jbeg = ibeg
        jend = iend
        MInvTrid[ibeg:iend, jbeg:jend] = MdenseInv[ibeg:iend, jbeg:jend]
        if ix < size(blockSizes)[1]
            # right
            jbeg = sum(blockSizes[1:ix]) + 1
            jend = sum(blockSizes[1:ix+1])
            MInvTrid[ibeg:iend, jbeg:jend] = MdenseInv[ibeg:iend, jbeg:jend]
        end
    end
    return MInvTrid
end