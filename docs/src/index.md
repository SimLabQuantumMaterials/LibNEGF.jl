# LibNEGF.jl's Documentation

Welcome to the documentation page of LibNEGF.jl.

```@docs
LibNEGF
load_energies(systemName::String)
load_matrices(systemName::String, E::Int, k::Int,
    whichMatsToLoad::Vector{String}, baseType::DataType,
    whereFrom::Int)
build_M_from_HS(H::SparseArrays.SparseMatrixCSC, S::SparseArrays.SparseMatrixCSC,
    Se::SparseArrays.SparseMatrixCSC, energVal::Float64)
bndiag_of_inv_direct(M_::SparseArrays.SparseMatrixCSC,
    blockSizes::Vector{Int}, ndiag::Dict{String,Int})
bndiag_of_inv_direct(M_::Any, blockSizes_::Any, ndiag::Any)
BlockMatrix
bm_convert(M::SparseArrays.SparseMatrixCSC, blockSizes::Vector{Int},
    ndiag::Dict{String,Int})
bm_convert(M::BlockMatrix)
bm_convert(M::BlockMatrix, permVec::Vector{Int})
bndiag_of_inv_ddrgf_create_sparse_permutator(permVec::Vector{Int}, blockSizes::Vector{Int},
    nrsType::DataType)
bm_copy(M::BlockMatrix)
bm_copy!(Mout::BlockMatrix, Min::BlockMatrix)
bm_similar(M::BlockMatrix, filling::Int)
bm_blocks_define!(M::BlockMatrix, filling::Int)
bm_blocks_define_complement11!(M::BlockMatrix, A::ArrayOrLUView_, filling::Int)
bm_blocks_define_complement22!(M::BlockMatrix, auxData, filling::Int)
bm_blocks_define_complement12!(M_::BlockMatrix, auxData, filling::Int)
bm_blocks_define_complement21!(M_::BlockMatrix, auxData, filling::Int)
bm_blocks_define_identity!(M::BlockMatrix)
bm_create_synthetic(A_::BlockMatrix, nrLayers::Int, blocksDim::Int)
bm_create_synthetic_random(nrLayers::Int, blocksDim::Int, nrsType::DataType)
bm_reference!(M::BlockMatrix, B::ArrayOrLUView_)
bm_reference!(M::BlockMatrix, B::ArrayOrLU_, iOffset::Int, jOffset::Int)
AuxDataRGF
allocate_aux_data_RGF(M::BlockMatrix, nrBLASThreadsOuter::Int, nrBLASThreadsInner::Int)
bndiag_of_inv_rgf_local!(Mout::BlockMatrix, Min::BlockMatrix, auxData::AuxDataRGF, td::TimingData,
    cd::CountingData)
```