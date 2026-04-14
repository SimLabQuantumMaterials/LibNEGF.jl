# wrapper for RGF
function bndiag_of_inv_rgf_local_wrapper(MspIN::SparseArrays.SparseMatrixCSC, blockSizes::Vector{Int},
    ndiag::Dict{String,Int}, isHermitian::Bool)::SparseArrays.SparseMatrixCSC
    # convert the sparse input matrix to BlockMatrix type first
    MbmIN::BlockMatrix = bm_convert(MspIN, blockSizes, ndiag, isHermitian)
    MbmOUT::BlockMatrix = bm_copy(MbmIN)
    # allocate auxiliary data
    auxData::AuxDataRGF = allocate_aux_data_RGF(MbmIN)
    # call RGF
    bndiag_of_inv_rgf_local!(MbmOUT, MbmIN, auxData, TimingData(), CountingData())
    # create and return the output
    MspOUT::SparseArrays.SparseMatrixCSC = bm_convert(MbmOUT)
    return MspOUT
end