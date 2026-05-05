# testing general block n-diagonal RGF vs Fused block tridiagonal approach

using Test, Random, SparseArrays
import LinearAlgebra

# fix seed to have reproducible tests
Random.seed!(1234)

# this factor relaxes the required relative tolerance
accFctrBare = 1.0E3
accFctr = 0.0

@testset "CompareGeneralVsFusedRGF" begin
    include("common_to_test.jl")

    # values for the synthetic matrix
    npl = 10
    blockSize = 32

    # Set the original bandwidth for the block n-diagonal matrix (e.g., n=5)
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

            # ---------------------------------------------------------
            # SETUP: Create synthetic matrix
            # ---------------------------------------------------------
            
            # Create synthetic block n-diagonal matrix
            MbmSynth = bm_create_synthetic_random(npl, blockSize, precx, false, ndiagDict)
            
            # ---------------------------------------------------------
            # PATH 1: Compute inverse using the general n-diagonal RGF
            # ---------------------------------------------------------
            
            MinGen = bm_copy(MbmSynth)
            MoutGen = bm_copy(MbmSynth)

            auxDataGen = allocate_aux_data_general_RGF(MinGen)
            
            # Execute generalized RGF
            bndiag_of_inv_general_rgf!(MoutGen, MinGen, auxDataGen, TimingData(), CountingData())

            # Convert general RGF result to a dense array
            rgfGenInvSp = bm_convert(MoutGen)
            rgfGenInv = Array(rgfGenInvSp)

            # ---------------------------------------------------------
            # PATH 2: Compute inverse using Fused approach (n=3 RGF)
            # ---------------------------------------------------------
            
            # Fuse the blocks to recast it as a block tridiagonal matrix
            MinFused = bm_fuse_to_tridiagonal(MbmSynth)
            MoutFused = bm_copy(MinFused)

            # Allocate standard RGF (n=3) aux data for the new tridiagonal matrix
            auxDataFused = allocate_aux_data_RGF(MinFused)
            
            # Execute standard RGF
            bndiag_of_inv_rgf_local!(MoutFused, MinFused, auxDataFused, TimingData(), CountingData())

            # Convert fused RGF result to a dense array
            fusedInvSp = bm_convert(MoutFused)
            fusedDense = Array(fusedInvSp)

            # ---------------------------------------------------------
            # ALIGNMENT: Extract n-diagonal footprint from Fused result
            # ---------------------------------------------------------
            
            # We map the dense fused result back into a BlockMatrix restricted by ndiagDict 
            # and the ORIGINAL block sizes, extracting only the footprint we care about.
            fusedAlignedBm = bm_convert(SparseArrays.SparseMatrixCSC{precx,Int}(fusedDense), MbmSynth.blockSizes, ndiagDict, false)
            fusedAlignedSp = bm_convert(fusedAlignedBm)
            fusedAlignedInv = Array(fusedAlignedSp)

            # ---------------------------------------------------------
            # COMPARE: General RGF vs Fused RGF
            # ---------------------------------------------------------
            
            relErr = LinearAlgebra.norm(rgfGenInv - fusedAlignedInv, 2) / LinearAlgebra.norm(fusedAlignedInv, 2)
            @test relErr < roundoffs[precx] * accFctr
        end
    end
end