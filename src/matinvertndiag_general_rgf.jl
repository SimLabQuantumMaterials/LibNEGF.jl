using TimerOutputs

"""
    general_allocate_aux_data_RGF(M::BlockMatrix)

Allocates buffers for a block n-diagonal matrix based on its specific `ndiag` pattern.
"""
function allocate_aux_data_general_RGF(M::BlockMatrix)::AuxDataRGF
    npl = size(M.blockSizes)[1]

    # buffM inherits the generalized ndiag sparsity pattern from M
    buffM = BlockMatrix(copy(M.blockSizes), ArrayOrLU_(undef, npl, npl),
        copy(M.ndiag), M.nrsType, 1, false)
    bm_blocks_define!(buffM, 1)

    bIdM = BlockMatrix(copy(M.blockSizes), ArrayOrLU_(undef, npl, npl),
        Dict("in" => 1, "out" => 1), M.nrsType, 0, false)
    bm_blocks_define_identity!(bIdM)

    # the final struct with the buffers
    auxData = AuxDataRGF(buffM, bIdM, 0)

    return auxData
end

"""
    general_bndiag_of_inv_rgf!(Mout::BlockMatrix, Min::BlockMatrix, auxData::AuxDataRGF, td::TimingData, cd::CountingData)

Computes the block n-diagonal part of the inverse of `Min` for an arbitrary block n-diagonal system.
"""
function bndiag_of_inv_general_rgf!(Mout::BlockMatrix, Min::BlockMatrix, auxData::AuxDataRGF, td::TimingData, cd::CountingData)
    
    minusOneCmplx = convert(Min.nrsType, -1.0)
    plusOneCmplx = convert(Min.nrsType, 1.0)
    zeroCmplx = convert(Min.nrsType, 0.0)

    npl = size(Mout.blockSizes)[1]
    
    # Extract the number of block diagonals from the dictionary
    w = div(Min.ndiag["in"] - 1, 2) 
    
    buffM1 = auxData.buffM
    buffM2 = Mout
    buffId = auxData.bIdM

    # ---------------------------------------------------------
    # UPWARD PASS (UDL Factorization & Schur Complement)
    # ---------------------------------------------------------
    
    # 1. Factorize the bottom-most element
    be_lu!(buffM1.M[npl, npl], Min.M[npl, npl], td, cd)

    # 2. Recursive block elimination
    for ix = npl-1:-1:1
        kMax = min(w, ix) 
        
        # Step A: Compute the left and right multipliers for the w-bandwidth
        for k = 1:kMax
            targetIdx = ix - k + 1
            
            # Right multiplier: buffM1[targetIdx, ix+1] = Min[targetIdx, ix+1] * (Min[ix+1, ix+1])^-1
            # Using the transpose trick to solve A^H X = B^H -> X^H = A^-H B^H
            be_ctranspose!(buffM1.M[ix+1, targetIdx], Min.M[targetIdx, ix+1], td, cd)
            be_mldivide!('C', buffM2.M[ix+1, targetIdx], buffM1.M[ix+1, targetIdx], buffM1.M[ix+1, ix+1], td, cd)
            be_ctranspose!(buffM1.M[targetIdx, ix+1], buffM2.M[ix+1, targetIdx], td, cd)

            # Left multiplier: buffM1[ix+1, targetIdx] = (Min[ix+1, ix+1])^-1 * Min[ix+1, targetIdx]
            be_mldivide!('N', buffM1.M[ix+1, targetIdx], Min.M[ix+1, targetIdx], buffM1.M[ix+1, ix+1], td, cd)
        end

        # Step B: Apply the Schur complement update to the w x w block grid
        for i = 1:kMax
            for j = 1:kMax
                targetR = ix - i + 1
                targetC = ix - j + 1
                
                be_copy_in_hw!(buffM2.M[targetR, targetC], Min.M[targetR, targetC])
                
                # Update: M_new = M_old - M_12 * (M_22^-1 * M_21)
                be_gemm!('N', 'N', minusOneCmplx, Min.M[targetR, ix+1], buffM1.M[ix+1, targetC], plusOneCmplx, buffM2.M[targetR, targetC], td, cd)
                
                # Factorize immediately if on the diagonal, else push to Min for future iterations
                if targetR == targetC && targetR == ix
                    be_lu!(buffM1.M[ix, ix], buffM2.M[ix, ix], td, cd)
                else
                    be_copy_in_hw!(Min.M[targetR, targetC], buffM2.M[targetR, targetC])
                end
            end
        end
    end

    # ---------------------------------------------------------
    # DOWNWARD PASS (Extracting bndiag(M^-1))
    # ---------------------------------------------------------
    
    # 1. Top-most element
    be_mldivide!('N', Mout.M[1, 1], buffId.M[1, 1], buffM1.M[1, 1], td, cd)

    # 2. Recursive downward substitution
    for ix = 2:npl
        kMax = min(w, ix - 1)

        for k = 1:kMax
            targetPrev = ix - k
            
            # Initialize target block accumulators
            be_fill!(Mout.M[targetPrev, ix], zeroCmplx)
            be_fill!(Mout.M[ix, targetPrev], zeroCmplx)
            
            # Accumulate sum over the bandwidth
            for k2 = 1:kMax
                m = ix - k2
                
                # Upper off-diagonals
                be_gemm!('N', 'N', minusOneCmplx, Mout.M[targetPrev, m], buffM1.M[m, ix], plusOneCmplx, Mout.M[targetPrev, ix], td, cd)
                
                # Lower off-diagonals
                be_gemm!('N', 'N', minusOneCmplx, buffM1.M[ix, m], Mout.M[m, targetPrev], plusOneCmplx, Mout.M[ix, targetPrev], td, cd)
            end
        end
        
        # Central diagonal update
        be_mldivide!('N', Mout.M[ix, ix], buffId.M[ix, ix], buffM1.M[ix, ix], td, cd)
        for k2 = 1:kMax
            m = ix - k2
            be_gemm!('N', 'N', minusOneCmplx, buffM1.M[ix, m], Mout.M[m, ix], plusOneCmplx, Mout.M[ix, ix], td, cd)
        end
    end

    # (Optional) Full inverse build logic generalized for w
    if auxData.buildFullInv == 1
        if npl > 2
            for ix = 1:npl
                for jx = (ix+w+1):npl
                    be_fill!(Mout.M[ix, jx], zeroCmplx)
                    be_fill!(Mout.M[jx, ix], zeroCmplx)
                    
                    for m = jx-w:jx-1
                        be_gemm!('N', 'N', minusOneCmplx, Mout.M[ix, m], buffM1.M[m, jx], plusOneCmplx, Mout.M[ix, jx], td, cd)
                        be_gemm!('N', 'N', minusOneCmplx, buffM1.M[jx, m], Mout.M[m, ix], plusOneCmplx, Mout.M[jx, ix], td, cd)
                    end
                end
            end
        end
    end
end