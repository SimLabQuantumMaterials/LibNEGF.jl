"A Julia package consisting of a rework of some parts of [libNEGF](https://github.com/libnegf/libnegf)."
module LibNEGF

using SparseArrays, CSV, MAT, LinearAlgebra

# from matloader.jl
export load_energies
export load_matrices
export build_T_from_HS

# from matinverter.jl
export btrid_of_inv_direct

# from matutils.jl
export convert_S2ALU_trid
export convert_ALU2S_trid

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

include("matloader.jl")
include("matinverter.jl")
include("arrayorlu.jl")
include("block.jl")

end