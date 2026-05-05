# testing general block n-diagonal RGF

using Random, SparseArrays

# fix seed to have reproducible tests
Random.seed!(1234)

# this factor relaxes the required relative tolerance
accFctrBare = 1.0E3
accFctr = 0.0

include("common_to_test.jl")

# values for the synthetic matrix
npl = 10
blockSize = 32

# Test for block 5-diagonal (n=5)
# You can change this to 7, 9, etc., to test wider bandwidths
nDiagVal = 5
ndiagDict = Dict("in" => nDiagVal, "out" => nDiagVal)

for k in kpoints
    for precx in precs
        
        if precx == ComplexF64
            global accFctr = accFctrBare
        else
            # extra relaxation in lower precision
            global accFctr = accFctrBare * 10.0
        end

        # Create synthetic block n-diagonal matrix using the overloaded function
        MbmSynth = bm_create_synthetic_random(npl, blockSize, precx, false, ndiagDict)
        
        # Prepare inputs and outputs for the RGF function
        Min = bm_copy(MbmSynth)
        Mout = bm_copy(MbmSynth)

        # ---------------------------------------------------------
        # FIRST: Compute inverse using the general RGF algorithm
        # ---------------------------------------------------------
        
        auxDataRGF = allocate_aux_data_general_RGF(Min)
        
        # Execute generalized RGF
        bndiag_of_inv_general_rgf!(Mout, Min, auxDataRGF, TimingData(), CountingData())

        # Convert RGF result to a dense array for numerical comparison
        rgfInvSp = bm_convert(Mout)
        rgfInv = Array(rgfInvSp)

        # ---------------------------------------------------------
        # THEN: Compute exact inverse by brute force
        # ---------------------------------------------------------
        
        # Convert original matrix to dense
        mSp = bm_convert(MbmSynth)
        mDense = Array(mSp)
        
        # Calculate the exact full inverse
        exactFullInv = LinearAlgebra.inv(mDense)
        
        # Extract ONLY the block n-diagonal part of the full inverse.
        # We map the dense result back into a BlockMatrix restricted by ndiagDict, 
        # and then immediately to a sparse/dense array.
        exactBndiagBm = bm_convert(SparseArrays.SparseMatrixCSC{precx,Int}(exactFullInv), MbmSynth.blockSizes, ndiagDict, false)
        exactBndiagSp = bm_convert(exactBndiagBm)
        exactBndiagInv = Array(exactBndiagSp)

        # ---------------------------------------------------------
        # COMPARE: Test the relative error
        # ---------------------------------------------------------
        
        relErr = LinearAlgebra.norm(rgfInv - exactBndiagInv, 2) / LinearAlgebra.norm(exactBndiagInv, 2)
        @test relErr < roundoffs[precx] * accFctr
    end
end