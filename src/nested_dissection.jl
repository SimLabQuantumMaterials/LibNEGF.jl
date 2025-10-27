using Base.Threads
using Distributed
using ThreadPinning
pinthreads(:cores)
using Dagger
Dagger.MemPool.MEM_RESERVE_SWEEPS[] = 0
Dagger.MemPool.MEM_RESERVED[] = 0
using PrecompileTools: @setup_workload, @compile_workload
using Statistics
BLAS.set_num_threads(1)


"""
	bm_reorder!(A::Matrix, orderX::Vector{Int}, orderY::Vector{Int}=orderX)::Matrix

Reorder the `Block` matrix `A` along `orderX` and `orderY` for rows and columns respectively.

# Arguments
- `A::Matrix` : The target matrix to reshape.
- `orderX::Vector{Int}` : The reshape order of row.
- `orderY::Vector{Int}` : The reshape order of column.
"""
function bm_reorder(A::Matrix, orderX::Vector{Int}, orderY::Vector{Int}=orderX)::Matrix
    B = bm_similar(A)
    for i in range(1, length(orderX))
        for j in range(1, length(orderY))
            if isassigned(A, orderX[i], orderY[j])
                B[i, j] = A[orderX[i], orderY[j]]
            end
        end
    end
    return B
end

"""
	bm_get_reorder(npl::Int; order::Vector{Int}=Vector(range(1,npl)), mirror::Bool=true, nb_level::Int=1)::Vector{Int}

Recursively reordering vector `order` by generating a separator index `sep` considerate as parent index of two sub-set `order`.
In place version.

# Arguments
- `npl::Int` : the number of block along the diagonal.
- `order::Vector{Int}` : vector that contains the index order of permutation.
- `mirror::Bool` : Reorder index to get tree with mirror branch from root (as much as possible).
- `nb_level::Int` : the maximum number of level of dissection.
"""
function bm_get_reorder(npl::Int; order::Vector{Int}=Vector(range(1, npl)), mirror::Bool=true, nb_level::Int=1)::Vector{Int}
    # Draft sequential version
    # Stop criteria
    if nb_level > 0
        # Update the order with the separator
        nb_level = nb_level - 1
        sep = ceil(Int, npl / 2)
        push!(order, order[sep])
        deleteat!(order, sep)

        # Recursive call on the left submatrix
        order[1:sep-1] = bm_get_reorder(sep - 1; order=order[1:sep-1], mirror=mirror, nb_level=nb_level)

        # Recrusive call on the right submatrix
        rsep = (mirror) ? sep : floor(Int, npl / 2)
        order[sep:npl-1] = bm_get_reorder(rsep; order=order[sep:npl-1], mirror=mirror, nb_level=nb_level)
    end
    return order
end

###
# Wrapper
###

"""
	bm_get_reorder_recTer(npl::Int; order::Vector{Int}=Vector(range(1,npl)), mirror::Bool=true, nb_level::Int=1)::Vector{Int}

Recursive terminal call of `orderBM` to get `order` order of permutation for parallelization.

# Arguments
- `npl::Int` : the number of block along the diagonal.
- `order::Vector{Int}` : vector that contains the index order of permutation.
- `mirror::Bool` : Reorder index to get tree with mirror branch from root (as much as possible).
- `nb_level::Int` : the maximum number of level of dissection.
"""
function bm_get_reorder_recTer(npl::Int; order::Vector{Int}=Vector(range(1, npl)), mirror::Bool=true, nb_level::Int=1)::Vector{Int}

    return bm_get_reorder(npl; order=order, mirror=mirror, nb_level=nb_level)
end

"""
    nd_factorization!(A::Matrix)::Matrix

Do the LU factorization of A with the nested-dissection parallelization scheme.
In place version.

# Arguments
- `A::Matrix` : The target matrix to factorize.
"""
# function nd_factorization!(A::Matrix)::Matrix
#     Dagger.spawn_datadeps() do
#         npl = size(A, 1)
#         for i = 1:npl
#             # Step 1 : create L(i,i) U(i,i)
#             Dagger.@spawn LAPACK.potrf!('L', InOut(A[i, i].Full))

#             idx = []
#             for j = i+1:npl
#                 # Warning : supposing matrix is symmetric
#                 if isassigned(A, j, i)
#                     # Step 2 : Generate L(i+1,i)
#                     Dagger.@spawn BLAS.trsm!('R', 'U', 'N', 'N', 1.0, In(A[i, i].Full), InOut(A[j, i].Full))
#                     # Step 3 : Generate U(i,i+1)
#                     Dagger.@spawn BLAS.trsm!('L', 'L', 'N', 'U', 1.0, In(A[i, i].Full), InOut(A[i, j].Full))
#                     Dagger.@spawn BLAS.gemm!('N', 'T', -1.0, In(A[j, i].Full), In(A[j, i].Full), 1.0, InOut(A[j, j].Full))
#                     # Step 4 : Update A(i+1:,i+1:)
#                     for k in idx
#                         if isassigned(A, j, k)
#                             Dagger.@spawn BLAS.gemm!('N', 'T', -1.0, In(A[j, i].Full), In(A[k, i].Full), 1.0, InOut(A[j, k].Full))
#                             Dagger.@spawn BLAS.gemm!('N', 'T', -1.0, In(A[k, i].Full), In(A[j, i].Full), 1.0, InOut(A[k, j].Full))
#                         else
#                             A[j, k] = Block(zeros(A[j, i].row, A[i, k].col)) # Should be support later on Dagger (ongoing work).
#                             A[k, j] = Block(zeros(A[k, i].row, A[i, j].col)) # Should be support later on Dagger (ongoing work).
#                             Dagger.@spawn BLAS.gemm!('N', 'T', -1.0, In(A[j, i].Full), In(A[k, i].Full), 1.0, InOut(A[j, k].Full))
#                             Dagger.@spawn BLAS.gemm!('N', 'T', -1.0, In(A[k, i].Full), In(A[j, i].Full), 1.0, InOut(A[k, j].Full))
#                         end
#                     end
#                     push!(idx, j)
#                 end
#             end
#         end
#     end

#     return A
# end

function nd_factorization!(A::Matrix)::Matrix
    Dagger.spawn_datadeps() do
        npl = size(A, 1)
        for i = 1:npl
            # Step 1 : create L(i,i) U(i,i)
            Dagger.@spawn LAPACK.potrf!('L', InOut(A[i, i].Full))

            idx = []
            for j = i+1:npl
                # Warning : supposing matrix is symmetric
                if isassigned(A, j, i)
                    push!(idx, j)
                    # Step 2 : Generate L(i+1,i)
                    Dagger.@spawn BLAS.trsm!('R', 'L', 'T', 'N', 1.0, In(A[i, i].Full), InOut(A[j, i].Full))
                end
                if isassigned(A, i, j)
                    # Step 3 : Generate U(i,i+1)
                    Dagger.@spawn BLAS.trsm!('L', 'L', 'N', 'N', 1.0, In(A[i, i].Full), InOut(A[i, j].Full))
                end
            end
            # Step 4 : Update A(i+1:,i+1:)
            for k in collect(Iterators.product(idx, idx))
                k = CartesianIndex(k)
                if isassigned(A, k[1], k[2])
                    Dagger.@spawn BLAS.gemm!('N', 'T', -1.0, In(A[k[1], i].Full), In(A[k[2], i].Full), 1.0, InOut(A[k[1], k[2]].Full))
                else
                    A[k] = Block(zeros(A[k[1], i].row, A[i, k[2]].col)) # Should be support later on Threads (ongoing work).
                    Dagger.@spawn BLAS.gemm!('N', 'T', -1.0, In(A[k[1], i].Full), In(A[k[2], i].Full), 1.0, InOut(A[k[1], k[2]].Full))
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
	nd_factorization(A::Matrix)::Matrix

Do the LU factorization on the `A` matrix with the nested-dissection parallelization scheme. 

The computation is done block by block along the diagonal starting from the first diagonal block (top left) to the last diagonal block (bottom right).

**Warning** : `A` supposes to be symetric

# Arguments
- `A::MAtrix` : The target matrix
"""
function nd_factorization(A::Matrix)::Matrix
    B = bm_copy(A)

    return nd_factorization!(B)
end