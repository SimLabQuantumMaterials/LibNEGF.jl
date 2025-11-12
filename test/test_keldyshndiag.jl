# testing Keldysh

module TestLibNEGFKeldysh

using LibNEGF, Test
import LinearAlgebra

# 1 from disk, 2 is random
whereFrom = 2
# values for the synthetic matrix
npl = 10
blockSize = 32

@testset "Keldyshndiag" begin
    include("common_to_test.jl")

    for systemx in systemNames
        for E in Epoints
            for k in kpoints
                for precx in precs
                    if whereFrom == 1
                        # load matrices and build M
                        listMatsToLoad = ["H", "S", "Sc"]
                        loadedMats, blockSizes = load_matrices(systemx, E, k,
                            listMatsToLoad, precx)
                        Hx = loadedMats[1]
                        Sx = loadedMats[2]
                        Se = loadedMats[3]
                        Mx = build_M_from_HS(Hx, Sx, Se, energVals[E])

                        # loading blockSizes only - this is redundant, but illustrates
                        # that this can be done without any matrix loading
                        listMatsToLoad = Vector{String}()
                        loadedMats, blockSizes = load_matrices(systemx, E, k,
                            listMatsToLoad, precx)

                        # convert to BlockMatrix
                        MbmFromData = bm_convert(Mx, blockSizes, Dict("in" => 3, "out" => 3))

                        # crate synthetic matrix
                        MbmSynth = bm_create_synthetic(MbmFromData, npl, blockSize)
                    else
                        MbmSynth = bm_create_synthetic_random(npl, blockSize, precx, false)
                    end

                    # this is T, whose inverse is Gr
                    Mbm = MbmSynth

                    # make Arandbm Hermitian, this will be the middle operator in Keldysh
                    Arandbm = bm_similar(Mbm, 2)
                    Arandsp = bm_convert(Arandbm)
                    Arandsp = (Arandsp + Arandsp') / 2
                    # BUT : we need to save Arandbm efficiently, considering that it is Hermitian,
                    #       specified via the last parameter in the following line
                    Arandbm = bm_convert(Arandsp, Arandbm.blockSizes, Arandbm.ndiag, true)

                    # we call this Sbm, which stands for \Sigma^{n}
                    Sbm = Arandbm

                    # FIRST, do Keldysh via the recursive algorithm

                    # pre-allocate buffer data for RKD
                    auxDataKeldysh = allocate_aux_data_Keldysh(Mbm, Sbm, parse(Int, ARGS[2]), parse(Int, ARGS[3]))
                    keldyshndiag!(Mbm, Sbm, auxDataKeldysh, TimingData(), CountingData())

                    exit()

                    # # FIRST, do Keldysh 'by hand'

                    # # # pre-allocate the output matrix
                    # # MbmInvNdiag = bm_similar(Mbm, 1)
                    # # # get the block n-diagonal of M^-1 via RGF
                    # # bndiag_of_inv_ddrgf!(MbmInvNdiag, Mbm, auxDataRGF, TimingData(), CountingData())

                    # # convert back to sparse
                    # MinvSp = bm_convert(MbmInvNdiag)

                    # C1sp = MinvSp * (Arandsp * MinvSp')
                    # # but, we need to extract the bndiag part of C1sp
                    # C1bm = bm_convert(C1sp, Mbm.blockSizes, Mbm.ndiag)
                    # C1sp = bm_convert(C1bm)
                    # # for de-allocation of some matrices
                    # MinvSp = 0
                    # Arandsp = 0
                    # C1bm = 0
                    # GC.gc()

                    # # THEN, do Keldysh via its function

                    # C2bm = bm_similar(Mbm, 1)
                    # auxDataKeldysh = allocate_aux_data_Keldysh(Mbm, auxDataRGF)
                    # keldyshndiag!(C2bm, MbmInvNdiag, Mbm, Arandbm, auxDataKeldysh, TimingData(), CountingData(), "v2")
                    # C2sp = bm_convert(C2bm)

                    # relErr = LinearAlgebra.norm(Array(C1sp - C2sp), 2) / LinearAlgebra.norm(Array(C1sp), 2)
                    # @test relErr < roundoffs[precx] * 1.0E02
                end
            end
        end
    end
end

end