using Metal

# TODO : documentation
struct MtlLU
    L::Metal.MtlArray
    U::Metal.MtlArray
    p::Metal.MtlVector
    P::Metal.MtlArray
end

"""
	ArrayOrLU_ = Matrix{Union{Metal.MtlArray, MtlLU, Nothing}}

Type for a matrix that can contain an `Array`, `LU factor` and/or `undef`,
with blocks stored in the (GPU) device.
"""
ArrayOrLU_ = Matrix{Union{Metal.MtlArray,MtlLU,Nothing}}

# TODO : write tests for many of the following backend function, but these
#        tests have to be backend-blind

# TODO : add documentation for all of the following functions

# ----------------------------------------------------
# 'base' types first e.g. Array and Metal.MtlArray
# TODO : check : is Julia inlining these? Or use macros instead?

# done : all the tests have been added for this section

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

# function be_set_to_zero_dev_array!(M::Metal.MtlArray)
#     be_copy_to_hw!(M, zero(size(M)[1], size(M)[2]))
# end

function be_zero_array(nrsType::DataType, dimsOfArr::Tuple{Int, Int})::Metal.MtlArray
    return Metal.MtlArray(zeros(nrsType, dimsOfArr))
end

# # ----------------------------------------------------
# # then composite types e.g. LU and MtlLU
# # TODO : check : is Julia inlining these? Or use macros instead?

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

# function be_zero_lu(nrsType::DataType, n::Int)::MtlLU
#     Mlu = MtlLU(zeros(nrsType, (n, n)), zeros(nrsType, (n, n)), zeros(Int, (n, n)), 1:n)
#     return Mlu
# end

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

# function be_lu!(Mout::MtlLU, Min::Metal.MtlArray)
#     Mincpu = be_copy_from_hw(Min)
#     McpuLU = lu(Mincpu)
#     be_copy_to_hw!(Mout, McpuLU)
# end

# function be_lu!(Mout::MtlLU, Min::Metal.MtlArray)
#     Mincpu = be_copy_from_hw(Min)
#     McpuLU = lu(Mincpu)
#     be_copy_to_hw!(Mout, McpuLU)
# end

# function be_lu(M::Metal.MtlArray)::MtlLU
#     Mcpu = be_copy_from_hw(M)
#     McpuLU = lu(Mcpu)
#     MmtlLU = be_copy_to_hw(McpuLU)
#     return MmtlLU
# end

# # this corresponds to mldivide, but using a precomputed LU
# function be_mldivide!(Mout::Metal.MtlArray, Min::Metal.MtlArray, Mlu::MtlLU)
#     Mlucpu = be_copy_from_hw(Mlu)
#     Mcpu = be_copy_from_hw(Min)
#     X = Mlucpu \ Mcpu
#     be_copy_to_hw!(Mout, X)
# end

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

# # function be_mul(M1::Metal.MtlArray, M2::Metal.MtlArray)::Metal.MtlArray
# #     Mcpy = be_copy_in_hw(M1)
# #     be_mul!(Mcpy, M1, M2)
# #     return Mcpy
# # end

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