"A Julia package consisting of a rework of some parts of [libNEGF](https://github.com/libnegf/libnegf)."
module LibNEGF

using SparseArrays, CSV, MAT, LinearAlgebra

# from matloader.jl
export load_energies
export load_matrices
export build_M_from_HS

# from matinvertndiag_*.jl
export bndiag_of_inv_direct
export bndiag_of_inv_ddrgf!
export allocate_aux_data_DDRGF

# from blockmatrix.jl
export BlockMatrix
export convert_S2BM_ndiag
export convert_BM2S_ndiag
export similar_bm_but_zero

# from backend_*.jl
export be_zero_array
export be_copy_in_hw!
export be_copy_from_hw
export be_copy_in_hw
export be_copy_to_hw
export be_copy_from_hw!
export be_copy_to_hw!
export be_zero_lu
export be_lu!
export be_lu
export be_A_from_LU
export be_mldivide!
export be_identity
export be_mul
export be_gemm!
export be_inv_from_lu!
export be_inv

# from utils.jl
# export @codeLocation

# include the backend for the desired harwdware, replace
# HW by one of cpu, apple, nvidia, etc
include("backend_HW.jl")

include("matloader.jl")
include("blockmatrix.jl")
include("matinvertndiag_direct.jl")
include("matinvertndiag_ddrgf.jl")
include("utils.jl")

end