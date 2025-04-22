"""
function bndiag_of_inv_direct!(M::SparseArrays.SparseMatrixCSC,
    blockSizes::Vector{Int}, ndiag::Dict{String,Int})

For an input matrix `M`, possibly but not necessarily block n-diagonal,
where n is tri, penta, etc., compute the block n-diagonal part of the inverse
of `M`. This function first computes `inv(M)` and then extracts its block
n-diagonal. This is the in-place version. If the input and output are both
n-diagonal, this avoids some resizing of arrays.

# Arguments
- `M::SparseArrays.SparseMatrixCSC`: the matrix to be inverted.
- `blockSizes::Vector{Int}`: the array of block sizes.
- `ndiag::Dict{String,Int}`: the number of block diagonals for the input
  and the output, e.g. Dict("in" => 3, "out" => 3) for block tri-diagonal.
"""
function bndiag_of_inv_direct!(M::SparseArrays.SparseMatrixCSC,
    blockSizes::Vector{Int}, ndiag::Dict{String,Int})
    # TODO : this function needs to be tested for block n-diagonal
    #        where n>3
    # convert to dense and invert - ndiag["in"] is not relevant
    # for this function
    Mdense = Array(M)
    MdenseInv = inv(Mdense)
    # de-allocate so that the garbage collector can later take care
    # of that
    Mdense = 0
    # do not allocate memory for the block tridiagonal part of the inverse
    # but rather re-label the input M
    MInvTrid = M
    # loop over the block sizes, conversely over the block rows
    for ix = 1:size(blockSizes)[1]
        # indices for the rows
        ibeg = sum(blockSizes[1:ix-1]) + 1
        iend = sum(blockSizes[1:ix])
        # now, copy the blocks within the ix-th block row
        if ix > 1
            # left
            for jx = (ix-1):-1:max(1, ix - Int((ndiag["out"] - 1) / 2))
                jbeg = sum(blockSizes[1:jx-1]) + 1
                jend = sum(blockSizes[1:jx])
                MInvTrid[ibeg:iend, jbeg:jend] = MdenseInv[ibeg:iend, jbeg:jend]
            end
        end
        # center
        jbeg = ibeg
        jend = iend
        MInvTrid[ibeg:iend, jbeg:jend] = MdenseInv[ibeg:iend, jbeg:jend]
        if ix < size(blockSizes)[1]
            # right
            for jx = (ix+1):1:min(size(blockSizes)[1], ix + Int((ndiag["out"] - 1) / 2))
                jbeg = sum(blockSizes[1:jx-1]) + 1
                jend = sum(blockSizes[1:jx])
                MInvTrid[ibeg:iend, jbeg:jend] = MdenseInv[ibeg:iend, jbeg:jend]
            end
        end
    end
    # no return, this is in-place
end

"""
function bndiag_of_inv_direct(M::SparseArrays.SparseMatrixCSC,
    blockSizes::Vector{Int}, ndiag::Dict{String,Int})

For an input matrix `M`, possibly but not necessarily block n-diagonal,
where n is tri, penta, etc., compute the block n-diagonal part of the inverse
of `M`. This function first computes `inv(M)` and then extracts its block
n-diagonal. If the input and output are both n-diagonal, this avoids some
resizing of arrays.

# Arguments
- `M::SparseArrays.SparseMatrixCSC`: the matrix to be inverted.
- `blockSizes::Vector{Int}`: the array of block sizes.
- `ndiag::Dict{String,Int}`: the number of block diagonals for the input
  and the output, e.g. Dict("in" => 3, "out" => 3) for block tri-diagonal.
"""
function bndiag_of_inv_direct(M_::SparseArrays.SparseMatrixCSC,
    blockSizes::Vector{Int}, ndiag::Dict{String,Int})::SparseArrays.SparseMatrixCSC
    # copy, to avoid modifying the input matrix M
    M = copy(M_)
    # compute the block n-diagonal of the inverse in-place
    bndiag_of_inv_direct!(M, blockSizes, ndiag)
    return M
end

"""
function bndiag_of_inv_direct(M_::Any, blockSizes_::Any, ndiag::Any)

For an input matrix `M`, possibly but not necessarily block n-diagonal,
where n is tri, penta, etc., compute the block n-diagonal part of the inverse
of `M`. This function first computes `inv(M)` and then extracts its block
n-diagonal. If the input and output are both n-diagonal, this avoids some
resizing of arrays. This is the catch-all version.

# Arguments
- `M_::Any`: the matrix to be inverted.
- `blockSizes_::Any`: the array of block sizes.
- `ndiag::Any`: the number of block diagonals for the input
  and the output, e.g. Dict("in" => 3, "out" => 3) for block tri-diagonal.
"""
function bndiag_of_inv_direct(M_::Any, blockSizes_::Any, ndiag::Any)::T where {T}
    if typeof(M_)!=SparseArrays.SparseMatrixCSC
        # first, try to convert M_
        try
            M = convert(SparseArrays.SparseMatrixCSC, M_)
        catch
            println("Error in converting M within bndiag_of_inv_direct(...) to
            sparse CSC, stopping")
            exit()
        end
    else
        M = copy(M_)
    end
    if typeof(blockSizes_)!=Vector{Int}
        # then, try to convert blockSizes_
        try
            blockSizes = convert(Vector{Int}, blockSizes_)
        catch
            println("Error in converting blockSizes within bndiag_of_inv_direct(...)
            to Vector{Int}, stopping")
            exit()
        end
    else
        blockSizes = blockSizes_
    end
    if typeof(ndiag_)!=Dict{String,Int}
        # finally, try to convert ndiag_
        try
            ndiag = convert(Dict{String,Int}, ndiag_)
        catch
            println("Error in converting ndiag within bndiag_of_inv_direct(...)
            to Dict{String,Int}, stopping")
            exit()
        end
    else
        ndiag = ndiag_
    end

    # do the computation in-place, as M is already a copy
    bndiag_of_inv_direct!(M, blockSizes, ndiag)

    if typeof(M)!=T
        # finally, try to convert M to the desired output type T
        try
            M2T = convert(T, M)
        catch
            println("Error in converting M within bndiag_of_inv_direct(...)
            to the desired output type, stopping")
            exit()
        end
    else
        M2T = M
    end
    return M2T
end

"""
function bndiag_of_inv_direct!(M_::Any, blockSizes_::Any, ndiag_::Any)

For an input matrix `M`, possibly but not necessarily block n-diagonal,
where n is tri, penta, etc., compute the block n-diagonal part of the inverse
of `M`. This function first computes `inv(M)` and then extracts its block
n-diagonal. If the input and output are both n-diagonal, this avoids some
resizing of arrays. This is the in-place call for the catch-all version.

# Arguments
- `M_::Any`: the matrix to be inverted.
- `blockSizes_::Any`: the array of block sizes.
- `ndiag::Any`: the number of block diagonals for the input
  and the output, e.g. Dict("in" => 3, "out" => 3) for block tri-diagonal.
"""
function bndiag_of_inv_direct!(M_::Any, blockSizes_::Any, ndiag_::Any)
    # if the type of M_ is not sparse CSC, then this function call
    # makes no sense
    if typeof(M_)!=SparseArrays.SparseMatrixCSC
        println("It makes no sense to call this function as in-place with
        a type that is not SparseArrays.SparseMatrixCSC")
        exit()
    else
        # otherwise, just re-label to call later the in-place computation
        M = M_
    end
    if typeof(blockSizes_)!=Vector{Int}
        # then, try to convert blockSizes_
        try
            blockSizes = convert(Vector{Int}, blockSizes_)
        catch
            println("Error in converting blockSizes within bndiag_of_inv_direct(...)
            to Vector{Int}, stopping")
            exit()
        end
    else
        blockSizes = blockSizes_
    end
    if typeof(ndiag_)!=Dict{String,Int}
        # finally, try to convert ndiag_
        try
            ndiag = convert(Dict{String,Int}, ndiag_)
        catch
            println("Error in converting ndiag within bndiag_of_inv_direct(...)
            to Dict{String,Int}, stopping")
            exit()
        end
    else
        ndiag = ndiag_
    end

    # do the computation in-place
    bndiag_of_inv_direct!(M, blockSizes, ndiag)
end