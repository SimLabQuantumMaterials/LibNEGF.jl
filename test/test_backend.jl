module TestLibNEGFBackend

using LibNEGF, Test
import LinearAlgebra

# the two blocks passed live on the device
function check_if_equal(B1, B2, roundoff)
    B1cpu = be_copy_from_hw(B1)
    B2cpu = be_copy_from_hw(B2)
    normDen = LinearAlgebra.norm(B2cpu, 2)
    if normDen==0
        relErr = LinearAlgebra.norm(B1cpu - B2cpu, 2)
    else
        relErr = LinearAlgebra.norm(B1cpu - B2cpu, 2) / normDen
    end
    @test relErr < roundoff
end

@testset "Backend" begin
    @testset "Backend Device Transfers" begin
        # check that moving data from, to and within the selected
        # hardware device works correctly

        include("common_to_test.jl")

        # we only want to load one matrix, but we keep the for loops
        # for generality and possible extended use in the future
        for systemx in [systemNames[1]]
            for E in [Epoints[1]]
                for k in [kpoints[1]]
                    for precx in precs
                        # list of matrices to load
                        listMatsToLoad = ["H", "S", "Sc"]
                        # load in the desired precision
                        loadedMats, blockSizes = load_matrices(systemx, E, k,
                            listMatsToLoad, precx)
                        H = loadedMats[1]
                        S = loadedMats[2]
                        Se = loadedMats[3]
                        M = build_M_from_HS(H, S, Se, energVals[E])

                        # convert to BlockMatrix, this lives in the device always
                        Abm = convert_S2BM_ndiag(M, blockSizes, Dict("in" => 3, "out" => 3))

                        Bz = be_zero_array(precx, size(Abm.M[1,1]))
                        be_copy_in_hw!(Bz, Abm.M[1,1])
                        check_if_equal(Bz, Abm.M[1,1], roundoffs[precx])

                        Bz = be_copy_in_hw(Abm.M[1,1])
                        check_if_equal(Bz, Abm.M[1,1], roundoffs[precx])

                        Bz = be_zero_array(precx, size(Abm.M[1,1]))
                        check_if_equal(be_copy_to_hw(zeros(precx, size(Abm.M[1,1]))), Bz, roundoffs[precx])

                        Bzcpu = zeros(precx, size(Abm.M[1,1]))
                        Bzdev = be_zero_array(precx, size(Abm.M[1,1]))
                        be_copy_from_hw!(Bzcpu, Abm.M[1,1])
                        be_copy_to_hw!(Bzdev, Bzcpu)
                        check_if_equal(Bzdev, Abm.M[1,1], roundoffs[precx])
                    end
                end
            end
        end
    end
end

end