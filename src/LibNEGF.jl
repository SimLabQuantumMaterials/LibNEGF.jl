"A Julia package consisting of a rework of some parts of [libNEGF](https://github.com/libnegf/libnegf)."
module LibNEGF

using SparseArrays, CSV, MAT, LinearAlgebra

# from matloader.jl
export load_energies
export load_matrices
export build_M_from_HS

# from matinvertndiag_*.jl
export bndiag_of_inv_direct
export bndiag_of_inv_rgf

# from blockmatrix.jl
export BlockMatrix
export convert_S2BM_ndiag
export convert_BM2S_ndiag

# include the backend for the desired harwdware, replace
# HW by one of cpu, apple, nvidia, etc
include("backend_HW.jl")

include("matloader.jl")
include("blockmatrix.jl")
include("matinvertndiag_direct.jl")
include("matinvertndiag_rgf.jl")

end