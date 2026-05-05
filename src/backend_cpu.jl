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
ArrayOrLUUnder_ = Union{Array,CpuLU,Nothing}
ArrayOrLU_ = Matrix{ArrayOrLUUnder_}
ArrayOrLUView_ = SubArray{ArrayOrLUUnder_,2,ArrayOrLU_,Tuple{UnitRange{Int},UnitRange{Int}},false}

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
    if Threads.nthreads() > 1
        adjoint!(Mout, Min)
    else
        @timewrap td "_ctranspose" begin
            @countwrap cd "_ctranspose" Min Min Min begin
                adjoint!(Mout, Min)
            end
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
    if Threads.nthreads() > 1
        copy!(Mout.A, Min)
        Mout.A, Mout.piv, info = LinearAlgebra.LAPACK.getrf!(Mout.A, Mout.piv)
        if info != 0
            println("ERROR: LAPACK lu returned an error info")
            @code_location
            exit()
        end
    else
        @timewrap td "_lu" begin
            @countwrap cd "_lu" Min Min Min begin
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
end

function be_lu(M::Array, td::TimingData, cd::CountingData)::CpuLU
    if Threads.nthreads() > 1
        Mlu = be_zero_lu(typeof(M[1, 1]), size(M)[1])
        be_lu!(Mlu, M, td, cd)
        return Mlu
    else
        @timewrap td "_lu" begin
            @countwrap cd "_lu" M M M begin
                Mlu = be_zero_lu(typeof(M[1, 1]), size(M)[1])
                be_lu!(Mlu, M, td, cd)
                return Mlu
            end
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
    if Threads.nthreads() > 1
        copy!(Mout, Min)
        LinearAlgebra.LAPACK.getrs!(trans, Mlu.A, Mlu.piv, Mout)
    else
        @timewrap td "_mldivide" begin
            @countwrap cd "_mldivide" Min Min Min begin
                copy!(Mout, Min)
                LinearAlgebra.LAPACK.getrs!(trans, Mlu.A, Mlu.piv, Mout)
            end
        end
    end
end

function be_gemm!(tA::Char, tB::Char, alpha::Number, A::Array,
    B::Array, beta::Number, C::Array, td::TimingData, cd::CountingData)
    if Threads.nthreads() > 1
        LinearAlgebra.BLAS.gemm!(tA, tB, alpha, A, B, beta, C)
    else
        @timewrap td "_gemm" begin
            @countwrap cd "_gemm" A B C begin
                LinearAlgebra.BLAS.gemm!(tA, tB, alpha, A, B, beta, C)
            end
        end
    end
end

function be_mul(M1::Array, M2::Array)::Array
    return M1 * M2
end

###########
# Count part for selected inverse
###########

function be_getri!(A::Array, td::TimingData, cd::CountingData)
    if Threads.nthreads() > 1
        LinearAlgebra.LAPACK.getri!(A, collect(1:size(A,1)))
    else
        @timewrap td "_getri" begin
            @countwrap cd "_getri" A A A begin
			    LinearAlgebra.LAPACK.getri!(A, collect(1:size(A,1)))
            end
        end
    end
end

function be_trsm!(side::Char, ul::Char, tA::Char, dA::Char, alpha::Number,
    A::Array, B::Array, td::TimingData, cd::CountingData)

    if Threads.nthreads() > 1
        LinearAlgebra.BLAS.trsm!(side, ul, tA, dA, alpha, A, B)
    else
        @timewrap td "_mldivide" begin
            @countwrap cd "_mldivide" A B B begin
				LinearAlgebra.BLAS.trsm!(side, ul, tA, dA, alpha, A, B)
            end
        end
    end
end

function be_getrf!(A::Array, td::TimingData, cd::CountingData)
    if Threads.nthreads() > 1
        LinearAlgebra.LAPACK.getrf!(A, collect(1:size(A,1)))
    else
        @timewrap td "_lu" begin
            @countwrap cd "_lu" A A A begin
				LinearAlgebra.LAPACK.getrf!(A, collect(1:size(A,1)))
            end
        end
    end
end

function be_potrf!(uplo::Char, A::Array, td::TimingData, cd::CountingData)
    if Threads.nthreads() > 1
        LinearAlgebra.LAPACK.potrf!(uplo, A)
    else
        @timewrap td "_lu" begin
            @countwrap cd "_lu" A A A begin
				LinearAlgebra.LAPACK.potrf!(uplo, A)
            end
        end
    end
end

function be_herk!(uplo::Char, tA::Char, alpha::Number, A::Array, beta::Number, C::Array, td::TimingData, cd::CountingData)
    if Threads.nthreads() > 1
        LinearAlgebra.BLAS.herk!(uplo, tA, alpha, A, beta, C)
    else
        @timewrap td "_gemm" begin
            @countwrap cd "_gemm" A A C begin
				LinearAlgebra.BLAS.herk!(uplo, tA, alpha, A, beta, C)
            end
        end
    end
end