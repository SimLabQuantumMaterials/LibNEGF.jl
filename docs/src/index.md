# LibNEGF.jl's Documentation

Welcome to the documentation page of LibNEGF.jl.

!!! note "Currently under general construction"
    This documentation is currently under initial development.

```@docs
LibNEGF
load_energies(systemName::String)
load_matrices(systemName::String, E::Int, k::Int,
    whichMatsToLoad::Vector{String}, baseType::DataType)
build_M_from_HS(H::SparseArrays.SparseMatrixCSC, S::SparseArrays.SparseMatrixCSC,
    Se::SparseArrays.SparseMatrixCSC, energVal::Float64)
bndiag_of_inv_direct(M_::SparseArrays.SparseMatrixCSC,
    blockSizes::Vector{Int}, ndiag::Dict{String,Int})
bndiag_of_inv_direct(M_::Any, blockSizes_::Any, ndiag::Any)
BlockMatrix
convert_S2BM_ndiag(M::SparseArrays.SparseMatrixCSC, blockSizes::Vector{Int},
    ndiag::Dict{String,Int})
convert_BM2S_ndiag(M::BlockMatrix)
```