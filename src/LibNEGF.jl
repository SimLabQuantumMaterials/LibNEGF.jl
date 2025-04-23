"A Julia package consisting of a rework of some parts of [libNEGF](https://github.com/libnegf/libnegf)."
module LibNEGF

using SparseArrays, CSV, MAT, LinearAlgebra

# from matloader.jl
export load_energies
export load_matrices
export build_M_from_HS

# from matinverter.jl
export bndiag_of_inv_direct

# from matutils.jl
export ArrayOrLU
export convert_S2ALU_trid
export convert_ALU2S_trid

# include the backend for the desired harwdware, replace
# HW by one of cpu, apple, nvidia, etc
include("backend_HW.jl")

include("matloader.jl")
include("matinverterdirect.jl")
include("arrayorlu.jl")

end