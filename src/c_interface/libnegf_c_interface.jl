# ------------------------------------------------------------------------------
# instructions to compile and create the shared library file

# NOTE : it seems that the main directory needs to be renamed
# to LibNEGF.jl

# # install JuliaC
# # 1. from the terminal
# LibNEGF.jl% julia
# # 2. open Julia's REPL
# julia> ]
# # 3. install JuliaC
# (@v1.12) pkg> app add JuliaC

# # then, go to the root directory of this project (which has been
# # renamed at this point as LibNEGF.jl) and run from the terminal:
# LibNEGF.jl% ./compile.sh cpu

# FIXME : is the following commented block still relevant?
# # above, the flag --export-ai generate a JSON file, which in turn can be used to
# # generate the header for the shared library .. I don't think this is really needed
# # at the moment, right? Because we have manually created our own interface header
# using JuliaLibWrapping
# # Translates the JSON spec into LibNEGFCInterface.h
# generate_c_header("LibNEGFCInterface.json", "LibNEGFCInterface.h")
# ------------------------------------------------------------------------------

# wrapper for RGF
function bndiag_of_inv_rgf_local_wrapper(MspIN::SparseArrays.SparseMatrixCSC{FieldType,Int}, blockSizes::Vector{Int},
    ndiag::Dict{String,Int}, isHermitian::Bool)::SparseArrays.SparseMatrixCSC{FieldType,Int}

    println(Core.stdout, "Running sequential RGF from its C interface function")

    # # convert the sparse input matrix to BlockMatrix type first
    MbmIN::BlockMatrix = bm_convert(MspIN, blockSizes, ndiag, isHermitian)
    MbmOUT::BlockMatrix = bm_copy(MbmIN)
    # allocate auxiliary data
    auxData::AuxDataRGF = allocate_aux_data_RGF(MbmIN)
    # call RGF
    bndiag_of_inv_rgf_local!(MbmOUT, MbmIN, auxData, TimingData(), CountingData())
    # create and return the output
    MspOUT::SparseArrays.SparseMatrixCSC{FieldType,Int} = bm_convert(MbmOUT)

    return MspIN
end

# C-COMPATIBLE RETURN STRUCT
# Note the change to Ptr{FieldType} for the non-zero values
struct CCscMatrix
    m::Cint
    n::Cint
    nnz::Cint
    colptr::Ptr{Cint}
    rowval::Ptr{Cint}
    nzval::Ptr{FieldType}
end

# THE C-CALLABLE WRAPPER
Base.@ccallable function run_bndiag_wrapper(
    # CSC Matrix Inputs
    m::Cint, n::Cint, nnz_in::Cint,
    colptr_in::Ptr{Cint}, rowval_in::Ptr{Cint}, nzval_in::Ptr{FieldType},

    # Vector Inputs
    n_blocks::Cint, block_sizes_in::Ptr{Cint},

    # Dictionary Inputs
    n_dict::Cint, dict_keys_in::Ptr{Cstring}, dict_vals_in::Ptr{Cint},

    # Boolean Input
    is_hermitian::Cint
)::CCscMatrix

    # --- A. RECONSTRUCT JULIA OBJECTS FROM C POINTERS ---
    colptr_j = unsafe_wrap(Array, colptr_in, n + 1; own=false) .+ 1
    rowval_j = unsafe_wrap(Array, rowval_in, nnz_in; own=false) .+ 1
    nzval_j = unsafe_wrap(Array, nzval_in, nnz_in; own=false) # Now an Array of FieldType
    MspIN = SparseMatrixCSC(Int(m), Int(n), colptr_j, rowval_j, nzval_j)

    blockSizes = Int.(unsafe_wrap(Array, block_sizes_in, n_blocks; own=false))

    keys_jv = unsafe_wrap(Array, dict_keys_in, n_dict; own=false)
    vals_jv = unsafe_wrap(Array, dict_vals_in, n_dict; own=false)
    ndiag = Dict{String,Int}()
    for i in 1:n_dict
        ndiag[unsafe_string(keys_jv[i])] = Int(vals_jv[i])
    end

    isHermitian = is_hermitian != 0

    # --- B. EXECUTE THE CORE ALGORITHM ---
    out_csc::SparseArrays.SparseMatrixCSC{FieldType,Int} = bndiag_of_inv_rgf_local_wrapper(MspIN, blockSizes, ndiag, isHermitian)

    # --- C. PREPARE THE RETURN VALUE FOR C ---
    out_nnz = SparseArrays.nnz(out_csc)
    out_m, out_n = size(out_csc)

    out_colptr_ptr = Libc.malloc(sizeof(Cint) * (out_n + 1))
    out_rowval_ptr = Libc.malloc(sizeof(Cint) * out_nnz)
    # Ensure memory is sized for FieldType (which is 16 bytes: two 8-byte floats)
    out_nzval_ptr = Libc.malloc(sizeof(FieldType) * out_nnz)

    out_colptr_0based = Cint.(out_csc.colptr .- 1)
    out_rowval_0based = Cint.(out_csc.rowval .- 1)
    out_nzval_complex = FieldType.(out_csc.nzval)

    unsafe_copyto!(Ptr{Cint}(out_colptr_ptr), pointer(out_colptr_0based), out_n + 1)
    unsafe_copyto!(Ptr{Cint}(out_rowval_ptr), pointer(out_rowval_0based), out_nnz)
    unsafe_copyto!(Ptr{FieldType}(out_nzval_ptr), pointer(out_nzval_complex), out_nnz)

    return CCscMatrix(
        Cint(out_m), Cint(out_n), Cint(out_nnz),
        Ptr{Cint}(out_colptr_ptr), Ptr{Cint}(out_rowval_ptr), Ptr{FieldType}(out_nzval_ptr)
    )
end