# LibNEGF.jl's Documentation

Welcome to the documentation page of LibNEGF.jl.

!!! note "Currently under general construction"
    This documentation is currently under initial development.

```@docs
LibNEGF
load_energies(systemName::String)
load_matrices(systemName::String, E::Int, k::Int,
    whichMatsToLoad::Vector{String}, baseType::DataType)
build_T_from_HS(H::SparseArrays.SparseMatrixCSC, S::SparseArrays.SparseMatrixCSC,
    Se::SparseArrays.SparseMatrixCSC, energVal::Float64)
btrid_of_inv_direct(T::SparseArrays.SparseMatrixCSC, blockSizes::Vector{Int})
```