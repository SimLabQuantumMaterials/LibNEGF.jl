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
    set_blocks_to_zero!(bmLargeBuff)

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

    Binvsp = convert_BM2S_ndiag(Binv)
    Asp = convert_BM2S_ndiag(A)
    @timewrap td "_sp_symm_gemm" Csp = Asp * (Binvsp * Asp')
    Cbm = convert_S2BM_ndiag(Csp, B.blockSizes, B.ndiag)
    copy_BM!(C, Cbm)
end

# tB can be either 'C' (for adjoint) or 'N' for no adjoint
function keldysh_gemm!(tB::Char, C_::BlockMatrix, A_::BlockMatrix, B_::BlockMatrix, ix::Int, jx::Int)
    npl = size(A_.blockSizes)[1]
    nUpDiagA = Int((A_.ndiag["in"] - 1) / 2)
    nUpDiagB = Int((B_.ndiag["in"] - 1) / 2)
    nrsType = A_.nrsType

    C = C_.M
    A = A_.M
    B = B_.M

    be_fill!(C[ix, jx], convert(A_.nrsType, 0.0))

    for kx = 1:npl
        # avoid accessing undefs in A and B
        cond1 = (kx <= ix + nUpDiagA) && (kx >= ix - nUpDiagA)
        cond2 = (jx <= kx + nUpDiagB) && (jx >= kx - nUpDiagB)

        if cond1 && cond2
            if tB == 'C'
                be_gemm!('N', 'C', convert(nrsType, 1.0), A[ix, kx], B[jx, kx], convert(nrsType, 1.0), C[ix, jx])
            else
                be_gemm!('N', 'N', convert(nrsType, 1.0), A[ix, kx], B[kx, jx], convert(nrsType, 1.0), C[ix, jx])
            end
        end
    end
end

# a more efficient version
function keldyshndiag_v2!(C::BlockMatrix, Binv::BlockMatrix, B::BlockMatrix, A::BlockMatrix, auxData::AuxDataKeldysh, td::TimingData)
    bndiag_of_inv_ddrgf!(Binv, B, auxData.auxDataRGF, td)

    npl = size(B.blockSizes)[1]

    # the (block) indices ix and jx are running over auxData.bmLargeBuff

    # FIRST, do auxData.bmLargeBuff = Binv * A'

    ndiag = auxData.bmLargeBuff.ndiag

    # loop over the block sizes, conversely over the block rows
    for ix = 1:npl
        # now, copy the blocks within the ix-th row
        # left
        if ix > 1
            for jx = (ix-1):-1:max(1, ix - Int((ndiag["out"] - 1) / 2))
                keldysh_gemm!('C', auxData.bmLargeBuff, Binv, A, ix, jx)
            end
        end
        # center
        keldysh_gemm!('C', auxData.bmLargeBuff, Binv, A, ix, ix)
        # right
        if ix < npl
            for jx = (ix+1):1:min(npl, ix + Int((ndiag["out"] - 1) / 2))
                keldysh_gemm!('C', auxData.bmLargeBuff, Binv, A, ix, jx)
            end
        end
    end

    # THEN, do C = A * auxData.bmLargeBuff

    ndiag = C.ndiag

    # loop over the block sizes, conversely over the block rows
    for ix = 1:npl
        # now, copy the blocks within the ix-th row
        # left
        if ix > 1
            for jx = (ix-1):-1:max(1, ix - Int((ndiag["out"] - 1) / 2))
                keldysh_gemm!('N', C, A, auxData.bmLargeBuff, ix, jx)
            end
        end
        # center
        keldysh_gemm!('N', C, A, auxData.bmLargeBuff, ix, ix)
        # right
        if ix < npl
            for jx = (ix+1):1:min(npl, ix + Int((ndiag["out"] - 1) / 2))
                keldysh_gemm!('N', C, A, auxData.bmLargeBuff, ix, jx)
            end
        end
    end
end