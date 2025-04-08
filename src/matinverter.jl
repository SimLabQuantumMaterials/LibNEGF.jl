function btridOfInvViaDirInv(T, blockSizes)
    Tdense = Array(T)
    TdenseInv = inv(Tdense)
    TInvTrid = copy(T)
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
            TInvTrid[ibeg:iend, jbeg:jend] = TdenseInv[ibeg:iend, jbeg:jend]
        end
        # center
        jbeg = ibeg
        jend = iend
        TInvTrid[ibeg:iend, jbeg:jend] = TdenseInv[ibeg:iend, jbeg:jend]
        if ix < size(blockSizes)[1]
            # right
            jbeg = sum(blockSizes[1:ix]) + 1
            jend = sum(blockSizes[1:ix+1])
            TInvTrid[ibeg:iend, jbeg:jend] = TdenseInv[ibeg:iend, jbeg:jend]
        end
    end
    return TInvTrid
end