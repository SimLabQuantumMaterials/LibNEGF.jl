"""
	AuxDataKeldysh

Buffers used by Keldysh. The data in `auxDataRGF` are used for supporting
RGF operations, and `bmLargeBuff` to do the further GEMM.
"""
struct AuxDataKeldysh
    auxDataRGF::AuxDataRGF
    buffS::BlockMatrix
    buffMdiag::BlockMatrix
end

function allocate_aux_data_Keldysh(M::BlockMatrix, S::BlockMatrix, nrBLASThreadsOuter::Int,
    nrBLASThreadsInner)::AuxDataKeldysh
    # npl = size(M.blockSizes)[1]

    auxDataRGF = allocate_aux_data_RGF(M, nrBLASThreadsOuter, nrBLASThreadsInner)

    # # the number of block diagonals
    # m = 1 + 2 * (M.ndiag["in"] - 1)

    # # in general, these type of auxiliary block matrices will contain
    # # Array-like object and not LU-like, as specified by the last param
    # bmLargeBuff = BlockMatrix(copy(M.blockSizes), ArrayOrLU_(undef, npl, npl),
    #     Dict("in" => m, "out" => m), M.nrsType, 0, )
    # bm_blocks_define!(bmLargeBuff, 1)

    # this is the main buffer, of the same structure as the central operator in Keldysh, i.e., S
    buffS = bm_copy(S)
    # in principle, we only need the lower triangular part of the following buffer
    buffMdiag = bm_copy(M)

    # the final struct with the buffers
    # auxDataKeldysh = AuxDataKeldysh(auxDataRGF, bmLargeBuff)
    auxDataKeldysh = AuxDataKeldysh(auxDataRGF, buffS, buffMdiag)

    return auxDataKeldysh
end

"""
    keldyshndiag!(C::BlockMatrix, Binv::BlockMatrix, B::BlockMatrix, A::BlockMatrix, auxData::AuxDataKeldysh, 
        td::TimingData, vsn::String)

Computes C = A * Binv * A^{H}, where Binv is the block tridiagonal of

# Arguments
- `td::TimingData`: it allows us refined data measurements, i.e. not only at the benchmarks
level but further within Keldysh and RGF.
- `vsn:String`: the version of the implementation, currently available "v1" and "v2".
"""
function keldyshndiag!(M::BlockMatrix, S::BlockMatrix, auxData::AuxDataKeldysh, td::TimingData,
    cd::CountingData)
    keldyshndiag_v3!(M, S, auxData, td, cd)

    # if vsn == "v1"
    #     keldyshndiag_v1!(C, Binv, B, A, auxData, td, cd)
    # elseif vsn == "v2"
    #     keldyshndiag_v2!(C, Binv, B, A, auxData, td, cd)
    # else
    #     println("ERROR: Keldysh implementation version not available")
    # end
end

# first version, naive, inefficient
function keldyshndiag_v1!(C::BlockMatrix, Binv::BlockMatrix, B::BlockMatrix, A::BlockMatrix, auxData::AuxDataKeldysh,
    td::TimingData, cd::CountingData)
    bndiag_of_inv_rgf_local!(Binv, B, auxData.auxDataRGF, td, cd)

    Binvsp = bm_convert(Binv)
    Asp = bm_convert(A)
    @timewrap td "_sp_symm_gemm" begin
        @countwrap cd "_sp_symm_gemm" Asp Binvsp Binvsp begin
            Csp = Binvsp * (Asp * Binvsp')
        end
    end
    Cbm = bm_convert(Csp, B.blockSizes, B.ndiag, false)
    bm_copy!(C, Cbm)
end

# a more efficient version
function keldyshndiag_v2!(C::BlockMatrix, Binv::BlockMatrix, B::BlockMatrix, A::BlockMatrix, auxData::AuxDataKeldysh,
    td::TimingData, cd::CountingData)
    bndiag_of_inv_rgf_local!(Binv, B, auxData.auxDataRGF, td, cd)

    # the (block) indices ix and jx are running over auxData.bmLargeBuff

    # do auxData.bmLargeBuff = A * Binv'
    bm_gemm!('N', 'C', convert(Binv.nrsType, 1.0), A, Binv, convert(Binv.nrsType, 0.0), auxData.bmLargeBuff, td, cd)

    # do C = Binv * auxData.bmLargeBuff
    bm_gemm!('N', 'N', convert(A.nrsType, 1.0), Binv, auxData.bmLargeBuff, convert(Binv.nrsType, 0.0), C, td, cd)
end

# implementation of RKD
# M : the T matrix in the case of block tridiagonal
# S : the central operator in Keldysh
function keldyshndiag_v3!(M::BlockMatrix, S::BlockMatrix, auxData::AuxDataKeldysh, td::TimingData,
    cd::CountingData)
    # we first run the upward pass of RGF, before any RKD steps
    keldyshndiag_upward_rgf!(M, auxData, td, cd)

    # IMPORTANT : the factors of the form T_{..}t_{..} and t_{..}T_{..} are already pre-computed
    #             and available in the block off-diagonal part of auxData.auxDataRGF.buffM

    # -------------

    # now we start with the actual RKD stuff

    bm_copy!(auxData.buffS, S)

    # upward pass of recursive Keldysh
    keldyshndiag_upward_rkd!(auxData, td, cd)

    # TODO #2 : central (upward/downward) pass of RKD

    # TODO #3 : downward pass of RKD
end

# upward RGF pass only
function keldyshndiag_upward_rgf!(M::BlockMatrix, auxData::AuxDataKeldysh, td::TimingData, cd::CountingData)
    minusOneCmplx = convert(M.nrsType, -1.0)
    plusOneCmplx = convert(M.nrsType, 1.0)
    npl = size(M.blockSizes)[1]

    buffM1 = auxData.auxDataRGF.buffM
    # auxData.buffMdiag is block-diagonal
    buffM2 = auxData.buffMdiag

    # bottom element
    be_lu!(buffM1.M[npl, npl], M.M[npl, npl], td, cd)

    # middle elements
    for ix = npl-1:-1:1
        # first run mrdivide, to make use of the mldivide data as a buffer for mrdivide

        # this is how we implement be_mrdivide!(..) via be_mldivide!(..)
        begin
            be_ctranspose!(buffM1.M[ix+1, ix], M.M[ix, ix+1], td, cd)
            be_mldivide!('C', buffM2.M[ix+1, ix], buffM1.M[ix+1, ix], buffM1.M[ix+1, ix+1], td, cd)
            be_ctranspose!(buffM1.M[ix, ix+1], buffM2.M[ix+1, ix], td, cd)
        end

        be_mldivide!('N', buffM1.M[ix+1, ix], M.M[ix+1, ix], buffM1.M[ix+1, ix+1], td, cd)

        be_copy_in_hw!(buffM2.M[ix, ix], M.M[ix, ix])
        be_gemm!('N', 'N', minusOneCmplx, M.M[ix, ix+1], buffM1.M[ix+1, ix], plusOneCmplx, buffM2.M[ix, ix], td, cd)
        # TODO : double-check, but this last LU factorization seems to not be needed
        be_lu!(buffM1.M[ix, ix], buffM2.M[ix, ix], td, cd)
    end
end

function keldyshndiag_upward_rkd!(auxData::AuxDataKeldysh, td::TimingData, cd::CountingData)
    npl = size(auxData.buffMdiag.blockSizes)[1]

    minusOneCmplx = convert(auxData.buffMdiag.nrsType, -1.0)
    plusOneCmplx = convert(auxData.buffMdiag.nrsType, 1.0)

    buffS = auxData.buffS
    buffTt = auxData.auxDataRGF.buffM

    for ix = npl-1:-1:1
        be_gemm!('N', 'C', minusOneCmplx, buffTt.M[ix, ix+1], buffS.M[ix, ix+1], plusOneCmplx, buffS.M[ix, ix], td, cd)
        be_gemm!('N', 'N', minusOneCmplx, buffTt.M[ix, ix+1], buffS.M[ix+1, ix+1], plusOneCmplx, buffS.M[ix, ix+1], td, cd)
        be_gemm!('N', 'C', minusOneCmplx, buffS.M[ix, ix+1], buffTt.M[ix, ix+1], plusOneCmplx, buffS.M[ix, ix], td, cd)
    end
end