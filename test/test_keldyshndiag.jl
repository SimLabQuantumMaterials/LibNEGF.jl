# testing Keldysh

module TestLibNEGFKeldysh

using LibNEGF, Test
import LinearAlgebra

@testset "Keldyshndiag" begin
    include("common_to_test.jl")

    for systemx in systemNames
        for E in Epoints
            for k in kpoints
                # pre-compute the condition number in double precision
                # list of matrices to load
                listMatsToLoad = ["H", "S", "Sc"]
                loadedMats, blockSizes = load_matrices(systemx, E, k,
                    listMatsToLoad, ComplexF64)
                H = loadedMats[1]
                S = loadedMats[2]
                Se = loadedMats[3]
                M = build_M_from_HS(H, S, Se, energVals[E])

                for precx in precs
                    # load matrices and build M
                    listMatsToLoad = ["H", "S", "Sc"]
                    loadedMats, blockSizes = load_matrices(systemx, E, k,
                        listMatsToLoad, precx)
                    H = loadedMats[1]
                    S = loadedMats[2]
                    Se = loadedMats[3]
                    M = build_M_from_HS(H, S, Se, energVals[E])

                    # loading blockSizes only - this is redundant, but illustrates
                    # that this can be done without any matrix loading
                    listMatsToLoad = Vector{String}()
                    loadedMats, blockSizes = load_matrices(systemx, E, k,
                        listMatsToLoad, precx)

                    # FIRST, do Keldysh 'by hand'

                    # convert to BlockMatrix
                    Mbm = convert_S2BM_ndiag(M, blockSizes, Dict("in" => 3, "out" => 3))
                    # pre-allocate buffer data for DD-RGF
                    auxDataRGF = allocate_aux_data_DDRGF(Mbm)
                    # pre-allocate the output matrix
                    MbmInvNdiag = similar_bm_but_zero(Mbm)
                    # get the block n-diagonal of M^-1 via RGF
                    bndiag_of_inv_ddrgf!(MbmInvNdiag, Mbm, auxDataRGF, TimingData())
                    # convert back to sparse
                    MinvSp = convert_BM2S_ndiag(MbmInvNdiag)
                    Arandbm = similar_bm_but_random(Mbm)
                    Arandsp = convert_BM2S_ndiag(Arandbm)
                    C1sp = Arandsp * (MinvSp * Arandsp')
                    # but, we need to extract the bndiag part of C1sp
                    C1bm = convert_S2BM_ndiag(C1sp, Mbm.blockSizes, Mbm.ndiag)
                    C1sp = convert_BM2S_ndiag(C1bm)
                    # for de-allocation of some matrices
                    MinvSp = 0
                    Arandsp = 0
                    C1bm = 0
                    GC.gc()

                    # THEN, do Keldysh via its function

                    C2bm = similar_bm_but_zero(Mbm)
                    auxDataKeldysh = allocate_aux_data_Keldysh(Mbm, auxDataRGF)
                    keldyshndiag!(C2bm, MbmInvNdiag, Mbm, Arandbm, auxDataKeldysh, TimingData(), "v2")
                    C2sp = convert_BM2S_ndiag(C2bm)

                    relErr = LinearAlgebra.norm(Array(C1sp - C2sp), 2) / LinearAlgebra.norm(Array(C1sp), 2)
                    @test relErr < roundoffs[precx] * 1.0E01
                end
            end
        end
    end
end

end