# testing Keldysh

module TestLibNEGFKeldysh

using LibNEGF, Test, SparseArrays, Random
import LinearAlgebra

# fix seed to have reproducible tests
Random.seed!(1234)

# this factor relaxes the required relative tolerance
accFctrBare = 1.0E3
accFctr = 0.0

@testset "Keldyshndiag" begin
    include("common_to_test.jl")

    # 1 from disk, 2 is random
    whereFrom = 2
    # values for the synthetic matrix
    npl = 10
    blockSize = 32

    for k in kpoints
        for precx in precs
            check_if_enough_mem_rkd(npl, blockSize, precx)

            if precx == ComplexF64
                global accFctr = accFctrBare
            else
                # extra relaxation in lower precision
                global accFctr = accFctrBare
            end

            # if whereFrom == 1
            #     # load matrices and build M
            #     listMatsToLoad = ["H", "S", "Sc"]
            #     loadedMats, blockSizes = load_matrices(systemx, E, k,
            #         listMatsToLoad, precx)
            #     Hx = loadedMats[1]
            #     Sx = loadedMats[2]
            #     Se = loadedMats[3]
            #     Mx = build_M_from_HS(Hx, Sx, Se, energVals[E])

            #     # loading blockSizes only - this is redundant, but illustrates
            #     # that this can be done without any matrix loading
            #     listMatsToLoad = Vector{String}()
            #     loadedMats, blockSizes = load_matrices(systemx, E, k,
            #         listMatsToLoad, precx)

            #     # convert to BlockMatrix
            #     MbmFromData = bm_convert(Mx, blockSizes, Dict("in" => 3, "out" => 3))

            #     # crate synthetic matrix
            #     MbmSynth = bm_create_synthetic(MbmFromData, npl, blockSize)
            # else
            #     MbmSynth = bm_create_synthetic_random(npl, blockSize, precx, false)
            # end

            MbmSynth = bm_create_synthetic_random(npl, blockSize, precx, false)

            # this is T, whose inverse is Gr
            Mbm = MbmSynth

            # make Arandbm Hermitian, this will be the middle operator in Keldysh
            Arandbm = bm_similar(Mbm, 2)
            Arandsp = bm_convert(Arandbm)
            Arandsp = (Arandsp + Arandsp') / convert(precx, 2.0)
            # BUT : we need to save Arandbm efficiently, considering that it is Hermitian,
            #       specified via the last parameter in the following line
            Arandbm = bm_convert(Arandsp, Arandbm.blockSizes, Arandbm.ndiag, true)

            # we call this Sbm, which stands for \Sigma^{n}
            Sbm = Arandbm

            # FIRST, do Keldysh via the recursive algorithm

            # pre-allocate buffer data for RKD
            auxDataKeldysh = allocate_aux_data_Keldysh(Mbm, Sbm)
            keldyshndiag!(Mbm, Sbm, auxDataKeldysh, TimingData(), CountingData())

            Gn = bm_convert(auxDataKeldysh.buffS)

            # THEN, do Keldysh by brute force

            M = bm_convert(Mbm)
            Gr = LinearAlgebra.inv(Array(M))
            S = bm_convert(Sbm)
            S = Array(S)
            GnBF = Gr * (S * Gr')
            GnBFbm = bm_convert(SparseArrays.SparseMatrixCSC{precx,Int}(GnBF), Sbm.blockSizes, Sbm.ndiag, true)
            GnBF = bm_convert(GnBFbm)

            relErr = LinearAlgebra.norm(Array(Gn) - GnBF, 2) / LinearAlgebra.norm(GnBF, 2)
            @test relErr < roundoffs[precx] * accFctr
        end
    end
end

end