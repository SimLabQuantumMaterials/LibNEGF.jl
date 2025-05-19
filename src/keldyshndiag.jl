struct AuxDataKeldysh
    auxDataRGF::AuxDataDDRGF
    bmLargeBuff::BlockMatrix
end

function allocate_aux_data_Keldysh(M::BlockMatrix, auxDataRGF::AuxDataDDRGF)::AuxDataKeldysh
    npl = size(M.blockSizes)[1]
    # the number of block diagonals
    m = 1 + 2 * (M.ndiag["in"] - 1)

    # in general, these type of auxiliary block matrices will contain
    # Array-like object and not LU-like, as specified by the last param
    bmLargeBuff = BlockMatrix(M.blockSizes, ArrayOrLU_(undef, npl, npl),
        Dict("in" => m, "out" => m), M.nrsType, 0)
    bm_blocks_define!(bmLargeBuff, 1)

    # the final struct with the buffers
    auxDataKeldysh = AuxDataKeldysh(auxDataRGF, bmLargeBuff)

    return auxDataKeldysh
end

# computes C = A bndiag(B^{-1}) A^{H}
function keldyshndiag!(C::BlockMatrix, Binv::BlockMatrix, B::BlockMatrix, A::BlockMatrix, auxData::AuxDataKeldysh, td::TimingData, vsn::String)
    if vsn == "v1"
        keldyshndiag_v1!(C, Binv, B, A, auxData, td)
    elseif vsn == "v2"
        keldyshndiag_v2!(C, Binv, B, A, auxData, td)
    else
        println("ERROR: Keldysh implementation version not available")
    end
end

# first version, naive, inefficient
function keldyshndiag_v1!(C::BlockMatrix, Binv::BlockMatrix, B::BlockMatrix, A::BlockMatrix, auxData::AuxDataKeldysh, td::TimingData)
    bndiag_of_inv_ddrgf!(Binv, B, auxData.auxDataRGF, td)

    Binvsp = bm_convert(Binv)
    Asp = bm_convert(A)
    @timewrap td "_sp_symm_gemm" Csp = Asp * (Binvsp * Asp')
    Cbm = bm_convert(Csp, B.blockSizes, B.ndiag)
    bm_copy!(C, Cbm)
end

# tB can be either 'C' (for adjoint) or 'N' for no adjoint
function bm_local_gemm!(tA::Char, tB::Char, alpha::Number, A_::BlockMatrix, B_::BlockMatrix, beta::Number, C_::BlockMatrix, ix::Int, jx::Int)
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
                be_gemm!('N', 'C', alpha, A[ix, kx], B[jx, kx], convert(nrsType, 1.0), C[ix, jx])
            else
                be_gemm!('N', 'N', alpha, A[ix, kx], B[kx, jx], convert(nrsType, 1.0), C[ix, jx])
            end
        end
    end
end

function bm_gemm!(tA::Char, tB::Char, alpha::Number, A::BlockMatrix, B::BlockMatrix, beta::Number, C::BlockMatrix)
    ndiag = C.ndiag
    npl = size(B.blockSizes)[1]

    # loop over the block sizes, conversely over the block rows
    for ix = 1:npl
        # now, copy the blocks within the ix-th row
        # left
        if ix > 1
            for jx = (ix-1):-1:max(1, ix - Int((ndiag["out"] - 1) / 2))
                bm_local_gemm!(tA, tB, alpha, A, B, beta, C, ix, jx)
            end
        end
        # center
        bm_local_gemm!(tA, tB, alpha, A, B, beta, C, ix, ix)
        # right
        if ix < npl
            for jx = (ix+1):1:min(npl, ix + Int((ndiag["out"] - 1) / 2))
                bm_local_gemm!(tA, tB, alpha, A, B, beta, C, ix, jx)
            end
        end
    end
end

# a more efficient version
function keldyshndiag_v2!(C::BlockMatrix, Binv::BlockMatrix, B::BlockMatrix, A::BlockMatrix, auxData::AuxDataKeldysh, td::TimingData)
    bndiag_of_inv_ddrgf!(Binv, B, auxData.auxDataRGF, td)

    # the (block) indices ix and jx are running over auxData.bmLargeBuff

    # do auxData.bmLargeBuff = Binv * A'
    bm_gemm!('N', 'C', convert(Binv.nrsType, 1.0), Binv, A, convert(Binv.nrsType, 0.0), auxData.bmLargeBuff)

    # do C = A * auxData.bmLargeBuff
    bm_gemm!('N', 'N', convert(A.nrsType, 1.0), A, auxData.bmLargeBuff, convert(Binv.nrsType, 0.0), C)
end