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
export AuxDataDDRGF

# from blockmatrix.jl
export BlockMatrix
export convert_S2BM_ndiag
export convert_BM2S_ndiag
export similar_bm_but_zero
export copy_BM

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

include("utils/parsing.jl")

# from utils/
# this is the core set of utils, where some macros are included
# empty, and if then utils_optnl.jl is included those empty macros
# are replaced
include("utils/common.jl")
@ifdef "LIBNEGF_FINER_TIMINGS" begin
    if ENV["LIBNEGF_FINER_TIMINGS"]=="0" which_timer = "empty"
    else which_timer = "full" end
    include("utils/"*which_timer*"_timings.jl")
end

# # include the backend for the desired harwdware, replace
# # HW by one of cpu, apple, nvidia, etc
@ifdef "LIBNEGF_HW" begin
    include("backend_"*ENV["LIBNEGF_HW"]*".jl")
end

# from block.jl
export Block
export bm_equal
export bm_copy
export sum_BlockMatrix
export prod_BlockMatrix
export bm_similar
export get_rcIndex
export get_rcIndexAt
export get_blockSizes
export full
export set_sparse_Block

# from selected_inverse.jl
export blockMatrix_factorization
export blockMatrix_inverse
export blockMatrix_factorization!
export blockMatrix_inverse!

include("matloader.jl")
include("blockmatrix.jl")
include("matinvertndiag_direct.jl")
include("matinvertndiag_ddrgf.jl")
include("block.jl")
include("selected_inverse.jl")

end