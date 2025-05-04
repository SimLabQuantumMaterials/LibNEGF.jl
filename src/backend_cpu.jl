# TODO : documentation
mutable struct CpuLU
    A::Array
    piv::Vector{Int}
end

"""
	ArrayOrLU_ = Matrix{Union{Array, LU, Nothing}}

Type for a matrix that can contain an `Array`, `LU factor` and/or `undef`.
"""
ArrayOrLU_ = Matrix{Union{Array,CpuLU,Nothing}}

# TODO : write tests for many of the following backend function, but these
#        tests have to be backend-blind

# TODO : add documentation for all of the following functions

# taken from:
# https://discourse.julialang.org/t/how-to-print-function-name-and-source-file-line-number/43486/2
macro code_location()
    return quote
        st = stacktrace(backtrace())
        myf = ""
        for frm in st
            funcname = frm.func
            if frm.func != :backtrace && frm.func != Symbol("macro expansion")
                myf = frm.func
                break
            end
        end
        println("in function ", $("$(__module__)"), ".$(myf) at ", $("$(__source__.file)"), ":", $("$(__source__.line)"))
    end
end

# ----------------------------------------------------
# 'base' types first e.g. Array and Metal.MtlArray
# TODO : check : is Julia inlining these? Or use macros instead?

# done : all the tests have been added for this section

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

# function be_set_to_zero_dev_array!(M::Metal.MtlArray)
#     be_copy_to_hw!(M, zero(size(M)[1], size(M)[2]))
# end

function be_zero_array(nrsType::DataType, dimsOfArr::Tuple{Int,Int})::Array
    return Array(zeros(nrsType, dimsOfArr))
end

function be_identity(nrsType::DataType, n::Int)
    return Array(LinearAlgebra.Diagonal(ones(nrsType, (n, n))))
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

# function be_copy_to_hw!(Mout::MtlLU, Min::LU)
#     # for now, we have to emulate this i.e. we do the copies
#     # 'manually'
#     be_copy_to_hw!(Mout.L, Min.L)
#     be_copy_to_hw!(Mout.U, Min.U)
#     be_copy_to_hw!(Mout.p, Min.p)
#     be_copy_to_hw!(Mout.P, Min.P)
# end

# function be_copy_to_hw(M::LU)::MtlLU
#     # for now, we have to emulate this i.e. we do the copies
#     # 'manually'
#     Mmtl = MtlLU(M.L, M.U, M.p, M.P)
#     return Mmtl
# end

# function be_copy_from_hw!(Mout::LU, Min::MtlLU)
#     # for now, we have to emulate this i.e. copy to CPU, do
#     # things on the CPU and return
#     Lcpu = be_copy_from_hw(Min.L)
#     Ucpu = be_copy_from_hw(Min.U)
#     Pcpu = be_copy_from_hw(Min.P)
#     A = Pcpu' * (Lcpu * Ucpu)
#     Alu = lu(A)
#     copy!(Mout, Alu)
#     # copy!(Mout.L, Alu.L)
#     # copy!(Mout.U, Alu.U)
#     # copy!(Mout.p, Alu.p)
#     # copy!(Mout.P, Alu.P)
# end

# function be_copy_from_hw(M::MtlLU)::LU
#     # for now, we have to emulate this i.e. copy to CPU, do
#     # things on the CPU and return
#     Lcpu = be_copy_from_hw(M.L)
#     Ucpu = be_copy_from_hw(M.U)
#     Pcpu = be_copy_from_hw(M.P)
#     A = Pcpu' * (Lcpu * Ucpu)
#     return lu(A)
# end

# function be_copy_in_hw!(Mout::MtlLU, Min::MtlLU)
#     be_copy_in_hw!(Mout.L, Min.L)
#     be_copy_in_hw!(Mout.U, Min.U)
#     be_copy_in_hw!(Mout.p, Min.p)
#     be_copy_in_hw!(Mout.P, Min.P)
# end

# function be_copy_in_hw(M::MtlLU)::MtlLU
#     Mcpy = MtlLU(M.L, M.U, M.p, M.P)
#     return Mcpy
# end

function be_zero_lu(nrsType::DataType, n::Int)::CpuLU
    Mlu = CpuLU(zeros(nrsType, (n, n)), Vector{Int}(undef, n))
    return Mlu
end

# # ----------------------------------------------------
# # finally, some functionality e.g. inv(...) and lu(...), where
# # all of the input and output matrices are assumed to be in the
# # desired hardware i.e. apple GPUs

# # for now, we have to emulate these i.e. copy to CPU, do things
# # on the CPU, and copy back to Metal

# # TODO(?) : create a function be_inv!(...) that actually modifies Min

# function be_inv!(Mout::Metal.MtlArray, Min::Metal.MtlArray)
#     Mincpu = be_copy_from_hw(Min)
#     MincpuInv = inv(Mincpu)
#     be_copy_to_hw!(Mout, MincpuInv)
# end

# function be_inv(M::Metal.MtlArray)::Metal.MtlArray
#     Mcpy = be_copy_in_hw(M)
#     # TODO(?) : change the following line once we have create a function
#     #           be_inv!(...) that actually modifies Min
#     be_inv!(Mcpy, M)
#     return Mcpy
# end

function be_lu!(Mout::CpuLU, Min::Array)
    copy!(Mout.A, Min)
    Mout.A, Mout.piv, info = LinearAlgebra.LAPACK.getrf!(Mout.A, Mout.piv)
    if info != 0
        println("ERROR: LAPACK lu returned an error info")
        @code_location
        exit()
    end
end

function be_lu(M::Array)::CpuLU
    Mlu = be_zero_lu(typeof(M[1,1]), size(M)[1])
    be_lu!(Mlu, M)
    return Mlu
end

# this corresponds to mldivide, but using a precomputed LU
function be_mldivide!(Mout::Array, Min::Array, Mlu::CpuLU)
    copy!(Mout, Min)
    LinearAlgebra.LAPACK.getrs!('N', Mlu.A, Mlu.piv, Mout)
end

# # this corresponds to mldivide, but using a precomputed LU
# function be_mldivide(M::Metal.MtlArray, Mlu::MtlLU)::Metal.MtlArray
#     Mcpy = be_copy_in_hw(M)
#     be_mldivide!(Mcpy, M, Mlu)
#     return Mcpy
# end

# # this corresponds to mrdivide, but using a precomputed LU
# function be_mrdivide!(Mout::Metal.MtlArray, Min::Metal.MtlArray, Mlu::MtlLU)
#     Mlucpu = be_copy_from_hw(Mlu)
#     Mincpu = be_copy_from_hw(Min)
#     X = Mincpu / Mlucpu
#     be_copy_to_hw!(Mout, X)
# end

# # this corresponds to mrdivide, but using a precomputed LU
# function be_mrdivide(M::Metal.MtlArray, Mlu::MtlLU)::Metal.MtlArray
#     Mcpy = be_copy_in_hw(M)
#     be_mrdivide!(Mcpy, M, Mlu)
#     return Mcpy
# end

# function be_gemm!(tA::DataType, tB::DataType, alpha::Number, A::Metal.MtlArray,
#     B::Metal.MtlArray, beta::Number, C::Metal.MtlArray)
#     # TODO : do we need to take care of data conversions?
#     Acpu = be_copy_from_hw(A)
#     Bcpu = be_copy_from_hw(B)
#     Ccpu = be_copy_from_hw(C)
#     LinearAlgebra.BLAS.gemm!(tA, tB, alpha, Acpu, Bcpu, beta, Ccpu)
#     be_copy_to_hw!(C, Ccpu)
# end

# function be_gemm!(tA::DataType, tB::DataType, alpha::Number, A::Metal.MtlArray,
#     B::Metal.MtlArray, C::Metal.MtlArray)
#     # TODO : do we need to take care of data conversions?
#     Acpu = be_copy_from_hw(A)
#     Bcpu = be_copy_from_hw(B)
#     Ccpu = LinearAlgebra.BLAS.gemm(tA, tB, alpha, Acpu, Bcpu)
#     be_copy_to_hw!(C, Ccpu)
# end

# function be_inverse_from_lu!(Mout::Metal.MtlArray, Mlu::MtlLU)
#     Mlucpu = be_copy_from_hw(Mlu)
#     # find the inverse by solving with the identity as rhs
#     Minvcpu = Mlucpu \ I
#     be_copy_from_hw!(Mout, Minvcpu)
# end

# function be_inverse_from_lu(Mlu::MtlLU)::Metal.MtlArray
#     Mlucpu = be_copy_from_hw(Mlu)
#     # find the inverse by solving with the identity as rhs
#     Minvcpu = Mlucpu \ I
#     Minvmtl = be_copy_from_hw(Minvcpu)
#     return Minvmtl
# end

# # function be_mul!(Mout::Metal.MtlArray, M1::Metal.MtlArray, M2::Metal.MtlArray)
# #     M1cpu = be_copy_from_hw(M1)
# #     M2cpu = be_copy_from_hw(M2)
# #     Moutcpu = M1cpu*M2cpu
# #     be_copy_to_hw!(Mout, Moutcpu)
# # end

function be_mul(M1::Array, M2::Array)::Array
    return M1 * M2
end

# # # this overrides M1 with the result of the subtraction
# # function be_minus!(M1::Metal.MtlArray, M2::Metal.MtlArray)
# #     M1cpu = be_copy_from_hw(M1)
# #     M2cpu = be_copy_from_hw(M2)
# #     Moutcpu = M1cpu - M2cpu
# #     be_copy_to_hw!(M1, Moutcpu)
# # end

# # function be_minus(M1::Metal.MtlArray, M2::Metal.MtlArray)::Metal.MtlArray
# #     M1cpu = be_copy_from_hw(M1)
# #     M2cpu = be_copy_from_hw(M2)
# #     Moutcpu = M1cpu - M2cpu
# #     Moutmtl = be_copy_to_hw(Moutcpu)
# #     return Moutmtl
# # end