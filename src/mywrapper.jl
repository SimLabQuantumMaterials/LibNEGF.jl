module MyWrapper

using SparseArrays

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

# C-COMPATIBLE RETURN STRUCT
# Note the change to Ptr{ComplexF64} for the non-zero values
struct CCscMatrix
    m::Cint
    n::Cint
    nnz::Cint
    colptr::Ptr{Cint}
    rowval::Ptr{Cint}
    nzval::Ptr{ComplexF64} 
end

# THE C-CALLABLE WRAPPER
Base.@ccallable function run_bndiag_wrapper(
        # CSC Matrix Inputs
        m::Cint, n::Cint, nnz_in::Cint,
        colptr_in::Ptr{Cint}, rowval_in::Ptr{Cint}, nzval_in::Ptr{ComplexF64},
        
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
    nzval_j  = unsafe_wrap(Array, nzval_in, nnz_in; own=false) # Now an Array of ComplexF64
    MspIN    = SparseMatrixCSC(Int(m), Int(n), colptr_j, rowval_j, nzval_j)

    blockSizes = Int.(unsafe_wrap(Array, block_sizes_in, n_blocks; own=false))

    keys_jv = unsafe_wrap(Array, dict_keys_in, n_dict; own=false)
    vals_jv = unsafe_wrap(Array, dict_vals_in, n_dict; own=false)
    ndiag = Dict{String, Int}()
    for i in 1:n_dict
        ndiag[unsafe_string(keys_jv[i])] = Int(vals_jv[i]) 
    end

    isHermitian = is_hermitian != 0

    # --- B. EXECUTE THE CORE ALGORITHM ---
    out_csc = bndiag_of_inv_rgf_local_wrapper(MspIN, blockSizes, ndiag, isHermitian)

    # --- C. PREPARE THE RETURN VALUE FOR C ---
    out_nnz = SparseArrays.nnz(out_csc)
    out_m, out_n = size(out_csc)

    out_colptr_ptr = Libc.malloc(sizeof(Cint) * (out_n + 1))
    out_rowval_ptr = Libc.malloc(sizeof(Cint) * out_nnz)
    # Ensure memory is sized for ComplexF64 (which is 16 bytes: two 8-byte floats)
    out_nzval_ptr  = Libc.malloc(sizeof(ComplexF64) * out_nnz)

    out_colptr_0based = Cint.(out_csc.colptr .- 1)
    out_rowval_0based = Cint.(out_csc.rowval .- 1)
    out_nzval_complex = ComplexF64.(out_csc.nzval)

    unsafe_copyto!(Ptr{Cint}(out_colptr_ptr), pointer(out_colptr_0based), out_n + 1)
    unsafe_copyto!(Ptr{Cint}(out_rowval_ptr), pointer(out_rowval_0based), out_nnz)
    unsafe_copyto!(Ptr{ComplexF64}(out_nzval_ptr), pointer(out_nzval_complex), out_nnz)

    return CCscMatrix(
        Cint(out_m), Cint(out_n), Cint(out_nnz), 
        Ptr{Cint}(out_colptr_ptr), Ptr{Cint}(out_rowval_ptr), Ptr{ComplexF64}(out_nzval_ptr)
    )
end

end # module