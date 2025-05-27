"""
	AuxDataKeldysh

Buffers used by Keldysh. The data in `auxDataRGF` are used for supporting
RGF operations, and `bmLargeBuff` to do the further GEMM.
"""
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

"""
    keldyshndiag!(C::BlockMatrix, Binv::BlockMatrix, B::BlockMatrix, A::BlockMatrix, auxData::AuxDataKeldysh, td::TimingData, vsn::String)

Computes C = A * Binv * A^{H}, where Binv is the block tridiagonal of

# Arguments
- `td::TimingData`: it allows us refined data measurements, i.e. not only at the benchmarks
level but further within Keldysh and RGF.
- `vsn:String`: the version of the implementation, currently available "v1" and "v2".
"""
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
    @timewrap td "_sp_symm_gemm" Csp = Binvsp * (Asp * Binvsp')
    Cbm = bm_convert(Csp, B.blockSizes, B.ndiag)
    bm_copy!(C, Cbm)
end

# a more efficient version
function keldyshndiag_v2!(C::BlockMatrix, Binv::BlockMatrix, B::BlockMatrix, A::BlockMatrix, auxData::AuxDataKeldysh, td::TimingData)
    bndiag_of_inv_ddrgf!(Binv, B, auxData.auxDataRGF, td)

    # the (block) indices ix and jx are running over auxData.bmLargeBuff

    # do auxData.bmLargeBuff = A * Binv'
    bm_gemm!('N', 'C', convert(Binv.nrsType, 1.0), A, Binv, convert(Binv.nrsType, 0.0), auxData.bmLargeBuff)

    # do C = Binv * auxData.bmLargeBuff
    bm_gemm!('N', 'N', convert(A.nrsType, 1.0), Binv, auxData.bmLargeBuff, convert(Binv.nrsType, 0.0), C)
end