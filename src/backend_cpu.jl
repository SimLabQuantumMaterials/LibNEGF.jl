"""
	CpuLU

Encapsulates the bare data needed for `LU`-like-related computations. This lives
on the CPU.
"""
mutable struct CpuLU
    A::Array
    piv::Vector{Int}
end

"""
	ArrayOrLU_

Contains a CPU `Array`, CPU `LU factor` and/or `undef`, with blocks stored
in the (CPU) device. This is at the base of BlockMatrix.
"""
ArrayOrLU_ = Matrix{Union{Array,CpuLU,Nothing}}

# ----------------------------------------------------
# 'base' types first e.g. Array and Metal.MtlArray
# TODO : check : is Julia inlining these? Or use macros instead?

function be_copy_to_hw!(Mout::Array, Min::Array)
    copy!(Mout, Min)
end

function be_copy_to_hw(M::Array)::Array
    return Array(M)
end

function be_copy_from_hw!(Mout::Array, Min::Array)
    copy!(Mout, Min)
end

function be_copy_from_hw(M::Array)::Array
    return Array(M)
end

function be_copy_in_hw!(Mout::Array, Min::Array)
    copy!(Mout, Min)
end

function be_copy_in_hw(M::Array)::Array
    copy(M)
end

function be_zero_array(nrsType::DataType, dimsOfArr::Tuple{Int,Int})::Array
    return Array(zeros(nrsType, dimsOfArr))
end

function be_identity(nrsType::DataType, n::Int)::Array
    return Array(LinearAlgebra.Diagonal(ones(nrsType, (n, n))))
end

function be_ctranspose!(Mout::Array, Min::Array, td::TimingData, cd::CountingData)
    @timewrap td "_ctranspose" begin
        @countwrap cd "_ctranspose" [(0,0)] begin
            adjoint!(Mout, Min)
        end
    end
end

function be_random_array(nrsType::DataType, dimsOfArr::Tuple{Int,Int})::Array
    return rand(nrsType, dimsOfArr)
end

function be_fill!(M::Array, x::Number)
    fill!(M, x)
end

# # ----------------------------------------------------
# # then composite types e.g. LU and MtlLU
# # TODO : check : is Julia inlining these? Or use macros instead?

# explicitly build the matrix A = LU from L and U
# WARNING : this function is not to be used if performance is important
function be_A_from_LU(M::CpuLU)::Array
    precx = typeof(M.A[1, 1])
    n = size(M.A)[1]

    idM = Array(LinearAlgebra.Diagonal(ones(precx, (n, n))))

    # first, compute PA = LU
    Ux = LinearAlgebra.BLAS.trmm('L', 'U', 'N', 'N', convert(precx, 1.0), M.A, idM)
    PAx = LinearAlgebra.BLAS.trmm('L', 'L', 'N', 'U', convert(precx, 1.0), M.A,
        Ux)

    # second, compute A, but for this we need P

    # create a vector px that matches the convention
    # of the p in LU (i.e. when calling lu(...))
    px = Vector{Int}(1:n)
    for ix in 1:n
        i1 = ix
        i2 = M.piv[ix]
        buffx = px[i1]
        px[i1] = px[i2]
        px[i2] = buffx
    end
    Px = zeros(precx, size(M.A))
    for ix in 1:n
        Px[ix, px[ix]] = 1
    end

    # finally, we can compute A = P' * PA
    Ax = Px' * PAx

    return Ax
end

function be_zero_lu(nrsType::DataType, n::Int)::CpuLU
    # IMPORTANT : the first option here gives issues at the level
    #             of the garbage collector
    # Mlu = CpuLU(zeros(nrsType, (n, n)), Vector{Int}(undef, n))
    Mlu = CpuLU(zeros(nrsType, (n, n)), Vector{Int}(ones(Int, (1, n))[1, :]))
    return Mlu
end

# # ----------------------------------------------------
# # finally, some functionality e.g. inv(...) and lu(...), where
# # all of the input and output matrices are assumed to be in the
# # desired hardware i.e. apple GPUs

function be_inv(M::Array)::Array
    return inv(M)
end

function be_lu!(Mout::CpuLU, Min::Array, td::TimingData, cd::CountingData)
    @timewrap td "_lu" begin
        @countwrap cd "_lu" [(0,0)] begin
            copy!(Mout.A, Min)
            Mout.A, Mout.piv, info = LinearAlgebra.LAPACK.getrf!(Mout.A, Mout.piv)
            if info != 0
                println("ERROR: LAPACK lu returned an error info")
                @code_location
                exit()
            end
        end
    end
end

function be_lu(M::Array, td::TimingData, cd::CountingData)::CpuLU
    @timewrap td "_lu" begin
        @countwrap cd "_lu" [(0,0)] begin
            Mlu = be_zero_lu(typeof(M[1, 1]), size(M)[1])
            be_lu!(Mlu, M, td, cd)
            return Mlu
        end
    end
end

function be_inv_from_lu!(Mout::Array, Min::CpuLU)
    copy!(Mout, Min.A)
    LinearAlgebra.LAPACK.getri!(Mout, Min.piv)
end

# this corresponds to mldivide, but using a precomputed LU
function be_mldivide!(trans::Char, Mout::Array, Min::Array, Mlu::CpuLU,
    td::TimingData, cd::CountingData)
    @timewrap td "_mldivide" begin
        @countwrap cd "_mldivide" [(0,0)] begin
            copy!(Mout, Min)
            LinearAlgebra.LAPACK.getrs!(trans, Mlu.A, Mlu.piv, Mout)
        end
    end
end

function be_gemm!(tA::Char, tB::Char, alpha::Number, A::Array,
    B::Array, beta::Number, C::Array, td::TimingData, cd::CountingData)
    @timewrap td "_gemm" begin
        @countwrap cd "_gemm" [size(A), size(B), size(C)] begin
            LinearAlgebra.BLAS.gemm!(tA, tB, alpha, A, B, beta, C)
        end
    end
end

function be_mul(M1::Array, M2::Array)::Array
    return M1 * M2
end