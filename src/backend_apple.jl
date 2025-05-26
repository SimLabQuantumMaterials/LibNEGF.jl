using Metal

"""
	MtlLU

Encapsulates the bare data needed for `LU`-like-related computations. The data
lives entirely on the Metal GPU.
"""
struct MtlLU
    A::Metal.MtlArray
    piv::Metal.MtlVector
end

"""
	CpuLU

Encapsulates the bare data needed for `LU`-like-related computations. This lives
on the CPU, and it is used by device code either for mimicking the CPU or for comparisons
in tests.
"""
mutable struct CpuLU
    A::Array
    piv::Vector{Int}
end

"""
	ArrayOrLU_

Contains a Metal `Array`, Metal `LU factor` and/or `undef`, with blocks stored
in the (GPU) device. This is at the base of BlockMatrix.
"""
ArrayOrLU_ = Matrix{Union{Metal.MtlArray,MtlLU,Nothing}}

# ----------------------------------------------------
# 'base' types first e.g. Array and Metal.MtlArray
# TODO : check : is Julia inlining these? Or use macros instead?

function be_copy_to_hw!(Mout::Metal.MtlArray, Min::Array)
    copy!(Mout, Min)
end

function be_copy_to_hw(M::Array)::Metal.MtlArray
    return Metal.MtlArray(M)
end

function be_copy_from_hw!(Mout::Array, Min::Metal.MtlArray)
    copy!(Mout, Min)
end

function be_copy_from_hw(M::Metal.MtlArray)::Array
    return Array(M)
end

function be_copy_in_hw!(Mout::Metal.MtlArray, Min::Metal.MtlArray)
    copy!(Mout, Min)
end

function be_copy_in_hw(M::Metal.MtlArray)::Metal.MtlArray
    copy(M)
end

function be_zero_array(nrsType::DataType, dimsOfArr::Tuple{Int,Int})::Metal.MtlArray
    return Metal.MtlArray(zeros(nrsType, dimsOfArr))
end

function be_identity(nrsType::DataType, n::Int)::Metal.MtlArray
    return be_copy_to_hw(Array(LinearAlgebra.Diagonal(ones(nrsType, (n, n)))))
end

function be_ctranspose!(Mout::Metal.MtlArray, Min::Metal.MtlArray)
    Mincpu = be_copy_from_hw(Min)
    Moutcpu = be_copy_from_hw(Mout)
    adjoint!(Moutcpu, Mincpu)
    be_copy_to_hw!(Mout, Moutcpu)
end

# # ----------------------------------------------------
# # then composite types e.g. LU and MtlLU
# # TODO : check : is Julia inlining these? Or use macros instead?

# explicitly build the matrix A = LU from L and U
# WARNING : this is not to be used if performance is critical
function be_A_from_LU(M::MtlLU)::Metal.MtlArray
    Mcpu = be_copy_from_hw(M)

    precx = typeof(Mcpu.A[1, 1])
    n = size(Mcpu.A)[1]

    identM = Array(LinearAlgebra.Diagonal(ones(precx, (n, n))))

    # first, compute PA = LU
    Ux = LinearAlgebra.BLAS.trmm('L', 'U', 'N', 'N', convert(precx, 1.0), Mcpu.A, identM)
    PAx = LinearAlgebra.BLAS.trmm('L', 'L', 'N', 'U', convert(precx, 1.0), Mcpu.A,
        Ux)

    # second, compute A, but for this we need P

    # create a vector px that matches the convention
    # of the p in LU (i.e. when calling lu(...))
    px = Vector{Int}(1:n)
    for ix in 1:n
        i1 = ix
        i2 = Mcpu.piv[ix]
        buffx = px[i1]
        px[i1] = px[i2]
        px[i2] = buffx
    end
    Px = zeros(precx, size(Mcpu.A))
    for ix in 1:n
        Px[ix, px[ix]] = 1
    end

    # finally, we can compute A = P' * PA
    Ax = Px' * PAx

    return be_copy_to_hw(Ax)
end

function be_copy_to_hw!(Mout::MtlLU, Min::CpuLU)
    be_copy_to_hw!(Mout.A, Min.A)
    be_copy_to_hw!(Mout.piv, Min.piv)
end

function be_copy_from_hw(M::MtlLU)::CpuLU
    Acpu = be_copy_from_hw(M.A)
    pcpu = be_copy_from_hw(M.piv)
    Mcpu = CpuLU(Acpu, pcpu)
    return Mcpu
end

function be_zero_lu(nrsType::DataType, n::Int)::MtlLU
    Az = be_copy_to_hw(zeros(nrsType, (n, n)))
    pivz = be_copy_to_hw(Vector{Int}(ones(Int, (1, n))[1, :]))
    Mlu = MtlLU(Az, pivz)
    return Mlu
end

# # ----------------------------------------------------
# # finally, some functionality e.g. inv(...) and lu(...), where
# # all of the input and output matrices are assumed to be in the
# # desired hardware i.e. apple GPUs

function be_lu!(Mout::MtlLU, Min::Metal.MtlArray)
    Mincpu = be_copy_from_hw(Min)

    n = size(Mincpu)[1]
    precx = typeof(Mincpu[1, 1])
    Moutcpu = CpuLU(zeros(precx, (n, n)), Vector{Int}(undef, n))

    copy!(Moutcpu.A, Mincpu)
    Moutcpu.A, Moutcpu.piv, info = LinearAlgebra.LAPACK.getrf!(Moutcpu.A, Moutcpu.piv)
    if info != 0
        println("ERROR: LAPACK lu returned an error info")
        @code_location
        exit()
    end

    be_copy_to_hw!(Mout, Moutcpu)
end

function be_lu(M::Metal.MtlArray)::MtlLU
    n = size(M)[1]
    Metal.@allowscalar precx = typeof(M[1, 1])
    Mout = be_zero_lu(precx, n)

    be_lu!(Mout, M)

    return Mout
end

function be_inv_from_lu!(Mout::Metal.MtlArray, Min::MtlLU)
    Mincpu = be_copy_from_hw(Min)
    Moutcpu = be_copy_from_hw(Mout)
    copy!(Moutcpu, Mincpu.A)
    LinearAlgebra.LAPACK.getri!(Moutcpu, Mincpu.piv)
    be_copy_to_hw!(Mout, Moutcpu)
end

# this corresponds to mldivide, but using a precomputed LU
function be_mldivide!(trans::Char, Mout::Metal.MtlArray, Min::Metal.MtlArray, Mlu::MtlLU)
    Moutcpu = be_copy_from_hw(Mout)
    Mincpu = be_copy_from_hw(Min)
    Mlucpu = be_copy_from_hw(Mlu)

    copy!(Moutcpu, Mincpu)
    LinearAlgebra.LAPACK.getrs!(trans, Mlucpu.A, Mlucpu.piv, Moutcpu)

    be_copy_to_hw!(Mout, Moutcpu)
end

function be_gemm!(tA::Char, tB::Char, alpha::Number, A::Metal.MtlArray,
    B::Metal.MtlArray, beta::Number, C::Metal.MtlArray)
    Acpu = be_copy_from_hw(A)
    Bcpu = be_copy_from_hw(B)
    Ccpu = be_copy_from_hw(C)

    LinearAlgebra.BLAS.gemm!(tA, tB, alpha, Acpu, Bcpu, beta, Ccpu)

    be_copy_to_hw!(C, Ccpu)
end

function be_mul!(Mout::Metal.MtlArray, M1::Metal.MtlArray, M2::Metal.MtlArray)
    M1cpu = be_copy_from_hw(M1)
    M2cpu = be_copy_from_hw(M2)
    Moutcpu = M1cpu * M2cpu
    be_copy_to_hw!(Mout, Moutcpu)
end

function be_mul(M1::Metal.MtlArray, M2::Metal.MtlArray)::Metal.MtlArray
    Metal.@allowscalar nrsType = typeof(M1[1, 1])
    n = size(M1)[1]
    m = size(M2)[2]

    Mout = be_zero_array(nrsType, (n, m))
    be_mul!(Mout, M1, M2)
    return Mout
end