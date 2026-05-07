"""
	Block_factorization!(A::Matrix)::Matrix

Do the LU factorization on the `A` matrix in place.

# Arguments
- `A::Matrix` : The target matrix in full form.
"""
function Block_factorization!(A::Matrix, td::TimingData, cd::CountingData)
    precx = typeof(A[1, 1].Full[1, 1])
    npl = size(A, 1)
    be_zero = convert(precx, 0.0)
    be_one = convert(precx, 1.0)
    be_mone = convert(precx, -1.0)

    for i = 1:npl
        # Step 1 : create L(i,i) U(i,i) --- A[i,i].Factors = lu(A[i,i].Full, NoPivot())
        be_getrf!(A[i, i].Full, td, cd)

        for j = i+1:npl
            if isassigned(A, j, i)
                # Step 2 : Generate L(i+1,i) --- A[j,i].Full /= A[i,i].Factors.U
                be_trsm!('R', 'U', 'N', 'N', be_one, A[i, i].Full, A[j, i].Full, td, cd)
            end
            if isassigned(A, i, j)
                # Step 3 : Generate U(i,i+1) --- A[i,j].Full = A[i,i].Factors.L \ A[i,j].Full
                be_trsm!('L', 'L', 'N', 'U', be_one, A[i, i].Full, A[i, j].Full, td, cd)
            end
        end

        # Step 4 : Update A(i+1:,i+1:)
        for k in i+1:npl
            if isassigned(A, i, k)
                be_gemm!('N', 'N', be_mone, A[k, i].Full, A[i, k].Full, be_one, A[k, k].Full, td, cd)
                for l in k+1:npl
                    if isassigned(A, i, l)
                        if isassigned(A, k, l)
                            # A[k].Full .-= prod(A[k[1],i], A[i,k[2]])
                            be_gemm!('N', 'N', be_mone, A[k, i].Full, A[i, l].Full, be_one, A[k, l].Full, td, cd)
                            be_gemm!('N', 'N', be_mone, A[l, i].Full, A[i, k].Full, be_one, A[l, k].Full, td, cd)
                        else
                            # A[k].Full = - prod(A[k,i], A[i,l])
                            A[k, l] = Block(zeros(A[k, i].row, A[i, l].col))
                            A[l, k] = Block(zeros(A[l, i].row, A[i, k].col))
                            be_gemm!('N', 'N', be_mone, A[k, i].Full, A[i, l].Full, be_zero, A[k, l].Full, td, cd)
                            be_gemm!('N', 'N', be_mone, A[l, i].Full, A[i, k].Full, be_zero, A[l, k].Full, td, cd)
                        end
                    end
                end
            end
        end
    end

    return A
end

function Block_factorization_noT!(A::Matrix)
    precx = typeof(A[1, 1].Full[1, 1])
    npl = size(A, 1)
    be_zero = convert(precx, 0.0)
    be_one = convert(precx, 1.0)
    be_mone = convert(precx, -1.0)

    for i = 1:npl
        # Step 1 : create L(i,i) U(i,i)
        # A[i,i].Factors = lu(A[i,i].Full, NoPivot())
        LinearAlgebra.LAPACK.getrf!(A[i, i].Full)

        for j = i+1:npl
            if isassigned(A, j, i)
                # Step 2 : Generate L(i+1,i) --- A[j,i].Full /= A[i,i].Factors.U
                LinearAlgebra.BLAS.trsm!('R', 'U', 'N', 'N', be_one, A[i, i].Full, A[j, i].Full)
            end
            if isassigned(A, i, j)
                # Step 3 : Generate U(i,i+1) --- A[i,j].Full = A[i,i].Factors.L \ A[i,j].Full
                LinearAlgebra.BLAS.trsm!('L', 'L', 'N', 'U', be_one, A[i, i].Full, A[i, j].Full)
            end
        end

        # Step 4 : Update A(i+1:,i+1:)
        for k in i+1:npl
            if isassigned(A, i, k)
                LinearAlgebra.BLAS.gemm!('N', 'N', be_mone, A[k, i].Full, A[i, k].Full, be_one, A[k, k].Full)
                for l in k+1:npl
                    if isassigned(A, i, l)
                        if isassigned(A, k, l)
                            # A[k].Full .-= prod(A[k[1],i], A[i,k[2]])
                            LinearAlgebra.BLAS.gemm!('N', 'N', be_mone, A[k, i].Full, A[i, l].Full, be_one, A[k, l].Full)
                            LinearAlgebra.BLAS.gemm!('N', 'N', be_mone, A[l, i].Full, A[i, k].Full, be_one, A[l, k].Full)
                        else
                            # A[k].Full = - prod(A[k,i], A[i,l])
                            A[k, l] = Block(zeros(A[k, i].row, A[i, l].col))
                            A[l, k] = Block(zeros(A[l, i].row, A[i, k].col))
                            LinearAlgebra.BLAS.gemm!('N', 'N', be_mone, A[k, i].Full, A[i, l].Full, be_zero, A[k, l].Full)
                            LinearAlgebra.BLAS.gemm!('N', 'N', be_mone, A[l, i].Full, A[i, k].Full, be_zero, A[l, k].Full)
                        end
                    end
                end
            end
        end
    end

    return A
end

#####
# Wrapper
#####

"""
	Block_factorization(A::Matrix)::Matrix

Do the LU factorization on the `A` matrix. 
Diagonal factors are stored in `Factors` field of `A`.
The algorithm is generic and avoid computation with zero blocks.

The computation is done block by block along the diagonal starting from the first diagonal block (top left) to the last diagonal block (bottom right).

**Warning** : `A` supposes to be symetric

# Arguments
- `A::Matrix` : The target matrix.
"""
function Block_factorization(A::Matrix, td::TimingData, cd::CountingData)::Matrix
    B = bm_copy(A)

    return Block_factorization!(B, td, cd)
end

"""
	Block_inverse!(A::Matrix, fillin::Bool=false, td::TimingData, cd::CountingData)::Matrix

Do the selected inverse of the matrix factorize `A` in place.

# Arguments
- `A::Matrix` : The targeted matrix in factorize form.
- `NoFillin::Bool` : Flag to know if we consider filled block during the computation.
"""
function Block_inverse!(A::Matrix, td::TimingData, cd::CountingData)
    precx = typeof(A[1, 1].Full[1, 1])
    npl = size(A, 1)
    be_zero = convert(precx, 0.0)
    be_one = convert(precx, 1.0)
    be_mone = convert(precx, -1.0)

    Uij = Vector{Matrix}(undef, 2)
    Lji = Vector{Matrix}(undef, 2)
    v_idx = Vector{Int}(undef, 2)
    for i = 1:2
        Uij[i] = zeros(precx, A[1, 1].row, A[1, 1].col)
        Lji[i] = zeros(precx, A[1, 1].row, A[1, 1].col)
    end
    Aii = zeros(precx, A[1, 1].row, A[1, 1].col)
    vid = 1


    # Step 0 : Compute inverse A(npl,npl)
    be_getri!(A[npl, npl].Full, td, cd)

    for i = npl-1:-1:1
        for j = i+1:npl
            if isassigned(A, i, j)
                for k = i+1:npl
                    if isassigned(A, i, k) && isassigned(A, k, j)
                        be_gemm!('N', 'N', be_one, A[i, k].Full, A[k, j].Full, be_one, Uij[vid], td, cd)
                        v_idx[vid] = j
                        be_gemm!('N', 'N', be_one, A[j, k].Full, A[k, i].Full, be_one, Lji[vid], td, cd)
                    end
                end

                # Uupdated[idx] = - A[i,i].Factors.U \ temp
                be_trsm!('L', 'U', 'N', 'N', be_mone, A[i, i].Full, Uij[vid], td, cd)
                # Lupdated[idx] = - temp / A[i,i].Factors.L
                be_trsm!('R', 'L', 'N', 'U', be_mone, A[i, i].Full, Lji[vid], td, cd)
                vid += 1
            end
        end

        # # Step 3 : Update A(i,i)
        for vj = 1:vid-1
            j = v_idx[vj]
            if isassigned(A, j, i)
                be_gemm!('N', 'N', be_one, Uij[vj], A[j, i].Full, be_one, Aii, td, cd)
                # A[j,i].Full = A[j,i].Factors
                LinearAlgebra.axpby!(be_one, Lji[vj], be_zero, A[j, i].Full)
                # A[i,j].Full = Uij[vj].Full
                LinearAlgebra.axpby!(be_one, Uij[vj], be_zero, A[i, j].Full)
                fill!(Uij[vj], zero(eltype(Uij[vj])))
                fill!(Lji[vj], zero(eltype(Lji[vj])))
            end
        end

        # Compute the rest part of A(i,i)
        # Aii = (Aii / A[i,i].Full.L))
        be_trsm!('R', 'L', 'N', 'U', be_one, A[i, i].Full, Aii, td, cd)
        # A[i,i].Full = A[i,i].Full.U \ (A[i,i].Full.L \ I)
        be_getri!(A[i, i].Full, td, cd)
        # A[i,i].Full -= Aii
        LinearAlgebra.axpy!(be_mone, Aii, A[i, i].Full)

        fill!(Aii, zero(eltype(Aii)))
        vid = 1
    end

    return A
end

"""
	Block_inverse(A::Matrix, opti::Bool=false)::Matrix

Do the selected inverse of the matrix factorize `A`.
The algorithm is generic and avoid computation with zero blocks.

The computation is done block by block along the diagonal starting from the last diagonal block (bottom right) to the first diagonal block (top left).

**Warning** : `A` supposes to be symetric

# Arguments
- `A::Matrix` : The targeted matrix in factorize form. 
- `NoFillin::Bool` : Flag to know if we consider filled block during the computation.
"""
function Block_inverse(A::Matrix, td::TimingData, cd::CountingData)::Matrix
    B = bm_copy(A)

    return Block_inverse!(B, td, cd)
end