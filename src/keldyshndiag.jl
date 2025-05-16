# computes C = A bndiag(B^{-1}) A^{H}
function keldyshndiag!(C::BlockMatrix, Binv::BlockMatrix, B::BlockMatrix, A::BlockMatrix, auxData::AuxDataDDRGF, td::TimingData)
    keldyshndiag_v1!(C, Binv, B, A, auxData, td)
end

# first version, naive, inefficient
function keldyshndiag_v1!(C::BlockMatrix, Binv::BlockMatrix, B::BlockMatrix, A::BlockMatrix, auxData::AuxDataDDRGF, td::TimingData)
    bndiag_of_inv_ddrgf!(Binv, B, auxData, td)

    Binvsp = convert_BM2S_ndiag(Binv)
    Asp = convert_BM2S_ndiag(A)
    Csp = Asp * (Binvsp * Asp')
    Cbm = convert_S2BM_ndiag(Csp, B.blockSizes, B.ndiag)
    copy_BM!(C, Cbm)
end